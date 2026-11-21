#include "esp_ssl.h"
#include <esp_log.h>
#include <esp_crt_bundle.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <sys/socket.h>
#include <cerrno>
#include <climits>

static const char *TAG = "EspSsl";
static constexpr int kIoDeadlineMs = 5000;
static constexpr int kRetryMs = 10;

EspSsl::EspSsl() {
    event_group_ = xEventGroupCreate();
}

EspSsl::~EspSsl() {
    // WebSocket/HTTP owners must outlive the entire receive callback. Failing
    // this programming contract cannot be repaired by freeing a running task.
    configASSERT(receive_task_handle_.load() != xTaskGetCurrentTaskHandle());
    Disconnect();
    if (event_group_) vEventGroupDelete(event_group_);
}

bool EspSsl::WaitForRetry() {
    return !(xEventGroupWaitBits(event_group_, ESP_SSL_EVENT_STOP, pdFALSE,
                                pdFALSE, pdMS_TO_TICKS(kRetryMs)) & ESP_SSL_EVENT_STOP);
}

void EspSsl::DestroyTls() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    socket_fd_ = -1;
    if (tls_client_) {
        // This is the only close(), preventing descriptor reuse from turning a
        // second close into an unrelated connection shutdown.
        esp_tls_conn_destroy(tls_client_);
        tls_client_ = nullptr;
    }
}

bool EspSsl::Connect(const std::string& host, int port) {
    std::lock_guard<std::mutex> owner(lifecycle_mutex_);
    if (!event_group_) { last_error_ = ENOMEM; return false; }
    if (tls_client_) return false;
    last_error_ = 0;
    xEventGroupClearBits(event_group_, ESP_SSL_EVENT_RECEIVE_TASK_EXIT | ESP_SSL_EVENT_STOP);
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        tls_client_ = esp_tls_init();
    }
    if (!tls_client_) { last_error_ = ENOMEM; return false; }

    esp_tls_cfg_t cfg = {};
    cfg.crt_bundle_attach = esp_crt_bundle_attach;
    cfg.non_block = true;
    cfg.timeout_ms = kIoDeadlineMs;
    const int64_t deadline = esp_timer_get_time() + int64_t(kIoDeadlineMs) * 1000;
    bool ready = false;
    while (!(xEventGroupGetBits(event_group_) & ESP_SSL_EVENT_STOP)) {
        int ret;
        {
            std::lock_guard<std::mutex> lock(io_mutex_);
            ret = esp_tls_conn_new_async(host.c_str(), host.size(), port, &cfg, tls_client_);
            // The descriptor appears during async setup. Keep it protected
            // against shutdown/destruction until every I/O user has exited.
            esp_tls_get_conn_sockfd(tls_client_, &socket_fd_);
        }
        if (ret == 1) { ready = true; break; }
        if (ret < 0) {
            esp_tls_error_handle_t error_handle;
            int code = 0, flags = 0;
            if (esp_tls_get_error_handle(tls_client_, &error_handle) == ESP_OK) {
                last_error_ = esp_tls_get_and_clear_last_error(error_handle, &code, &flags);
            } else last_error_ = EIO;
            break;
        }
        if (esp_timer_get_time() >= deadline) { last_error_ = ETIMEDOUT; break; }
        if (!WaitForRetry()) break;
    }
    if (!ready || (xEventGroupGetBits(event_group_) & ESP_SSL_EVENT_STOP)) {
        if (!last_error_) last_error_ = ECANCELED;
        DestroyTls();
        return false;
    }
    connected_ = true;
    TaskHandle_t task = nullptr;
    auto receive = [](void* arg) {
        auto* ssl = static_cast<EspSsl*>(arg);
        ssl->receive_task_handle_ = xTaskGetCurrentTaskHandle();
        ssl->ReceiveTask();
        xEventGroupSetBits(ssl->event_group_, ESP_SSL_EVENT_RECEIVE_TASK_EXIT);
#if CONFIG_IDF_TARGET_ESP32S3 && CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY
        vTaskDeleteWithCaps(nullptr);
#else
        vTaskDelete(nullptr);
#endif
    };
#if CONFIG_IDF_TARGET_ESP32S3 && CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY
    BaseType_t created = xTaskCreateWithCaps(receive, "ssl_receive", 4096, this, 1,
                                            &task, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
    BaseType_t created = xTaskCreate(receive, "ssl_receive", 4096, this, 1, &task);
#endif
    if (created != pdPASS) {
        last_error_ = ENOMEM;
        RequestDisconnect();
        DestroyTls();
        return false;
    }
    receive_task_handle_ = task;
    return true;
}

void EspSsl::RequestDisconnect() {
    connected_ = false;
    if (event_group_) xEventGroupSetBits(event_group_, ESP_SSL_EVENT_STOP);
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (socket_fd_ >= 0) shutdown(socket_fd_, SHUT_RDWR);
}

void EspSsl::Disconnect() {
    RequestDisconnect();
    if (receive_task_handle_.load() == xTaskGetCurrentTaskHandle()) return;
    std::lock_guard<std::mutex> owner(lifecycle_mutex_);
    if (receive_task_handle_) {
        // No I/O/send lock is held while joining callbacks. A timeout only logs;
        // it can never make destroying a still-running callback safe. A stuck
        // application callback delays destruction, without an infinite busy loop.
        while (!(xEventGroupWaitBits(event_group_, ESP_SSL_EVENT_RECEIVE_TASK_EXIT,
                                     pdFALSE, pdFALSE, pdMS_TO_TICKS(10000)) &
                 ESP_SSL_EVENT_RECEIVE_TASK_EXIT)) {
            ESP_LOGW(TAG, "Still waiting for SSL receive callback to finish");
        }
        receive_task_handle_ = nullptr;
    }
    std::lock_guard<std::mutex> send(send_mutex_);
    DestroyTls();
}

int EspSsl::Send(const std::string& data) {
    // Serialize an entire frame, including all partial-write retries.
    std::lock_guard<std::mutex> send(send_mutex_);
    if (!connected_ || data.size() > INT_MAX) return -1;
    const int64_t deadline = esp_timer_get_time() + int64_t(kIoDeadlineMs) * 1000;
    size_t sent = 0;
    while (sent < data.size() && connected_) {
        if (esp_timer_get_time() >= deadline) { last_error_ = ETIMEDOUT; break; }
        int ret;
        {
            std::lock_guard<std::mutex> io(io_mutex_);
            if (!connected_ || !tls_client_) break;
            ret = esp_tls_conn_write(tls_client_, data.data() + sent, data.size() - sent);
        }
        if (ret == ESP_TLS_ERR_SSL_WANT_WRITE || ret == ESP_TLS_ERR_SSL_WANT_READ) {
            if (!WaitForRetry()) break;
            continue;
        }
        if (ret <= 0 || size_t(ret) > data.size() - sent) { last_error_ = EIO; break; }
        sent += ret;
    }
    if (sent == data.size()) return static_cast<int>(sent);
    // A partially sent frame poisons the stream, including cancellation.
    RequestDisconnect();
    return -1;
}

void EspSsl::ReceiveTask() {
    std::string data(1500, '\0');
    while (connected_) {
        int ret;
        {
            std::lock_guard<std::mutex> io(io_mutex_);
            if (!connected_ || !tls_client_) break;
            data.resize(1500);
            ret = esp_tls_conn_read(tls_client_, data.data(), data.size());
        }
        if (ret == ESP_TLS_ERR_SSL_WANT_READ || ret == ESP_TLS_ERR_SSL_WANT_WRITE) {
            if (!WaitForRetry()) break;
            continue;
        }
        if (ret <= 0 || size_t(ret) > data.size()) {
            if (ret < 0 && connected_) last_error_ = ret;
            break;
        }
        if (connected_ && stream_callback_) {
            data.resize(ret);
            stream_callback_(data);
        }
    }
    RequestDisconnect();
    if (disconnect_callback_) disconnect_callback_();
}

int EspSsl::GetLastError() { return last_error_.load(); }
