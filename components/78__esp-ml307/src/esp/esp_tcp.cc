#include "esp_tcp.h"

#include <esp_log.h>
#include <unistd.h>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <errno.h>
#include <freertos/idf_additions.h>
#include <esp_heap_caps.h>

static const char *TAG = "EspTcp";

EspTcp::EspTcp() {
    event_group_ = xEventGroupCreate();
}

EspTcp::~EspTcp() {
    // The owner must destroy this object, never one of its receive callbacks.
    configASSERT(receive_task_handle_ == nullptr ||
                 xTaskGetCurrentTaskHandle() != receive_task_handle_);
    Disconnect();

    if (event_group_ != nullptr) {
        vEventGroupDelete(event_group_);
        event_group_ = nullptr;
    }
}

bool EspTcp::Connect(const std::string& host, int port) {
    // A peer can have disconnected while its receive callback is still running.
    // Join the old task even when connected_ has already become false.
    Disconnect();
    last_error_ = 0;

    struct sockaddr_in server_addr;
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    // host is domain
    struct hostent *server = gethostbyname(host.c_str());
    if (server == NULL) {
        last_error_ = h_errno;
        ESP_LOGE(TAG, "Failed to get host by name");
        return false;
    }
    memcpy(&server_addr.sin_addr, server->h_addr, server->h_length);

    tcp_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd_ < 0) {
        last_error_ = errno;
        ESP_LOGE(TAG, "Failed to create socket");
        return false;
    }
    int no_delay = 1;
    setsockopt(tcp_fd_, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
    struct timeval send_timeout = {.tv_sec = 5, .tv_usec = 0};
    setsockopt(tcp_fd_, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));

    int ret = connect(tcp_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (ret < 0) {
        last_error_ = errno;
        ESP_LOGE(TAG, "Failed to connect to %s:%d, code=0x%x", host.c_str(), port, last_error_.load());
        close(tcp_fd_);
        tcp_fd_ = -1;
        return false;
    }

    connected_ = true;

    xEventGroupClearBits(event_group_, ESP_TCP_EVENT_RECEIVE_TASK_EXIT);
#if CONFIG_IDF_TARGET_ESP32S3 && CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY
    BaseType_t created = xTaskCreateWithCaps([](void* arg) {
        EspTcp* tcp = (EspTcp*)arg;
        tcp->ReceiveTask();
        xEventGroupSetBits(tcp->event_group_, ESP_TCP_EVENT_RECEIVE_TASK_EXIT);
        vTaskDeleteWithCaps(NULL);
    }, "tcp_receive", 4096, this, 1, &receive_task_handle_, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
    BaseType_t created = xTaskCreate([](void* arg) {
        EspTcp* tcp = (EspTcp*)arg;
        tcp->ReceiveTask();
        xEventGroupSetBits(tcp->event_group_, ESP_TCP_EVENT_RECEIVE_TASK_EXIT);
        vTaskDelete(NULL);
    }, "tcp_receive", 4096, this, 1, &receive_task_handle_);
#endif
    if (created != pdPASS) {
        ESP_LOGE(TAG, "Failed to start TCP receive task");
        connected_ = false;
        close(tcp_fd_);
        tcp_fd_ = -1;
        receive_task_handle_ = nullptr;
        return false;
    }
    return true;
}

void EspTcp::Disconnect() {
    RequestDisconnect();
    if (receive_task_handle_ != nullptr &&
        xTaskGetCurrentTaskHandle() == receive_task_handle_) {
        // A callback may request shutdown. Its owner joins before destruction.
        return;
    }
    if (receive_task_handle_ != nullptr) {
        // Shutdown wakes recv(). Never free callbacks/event_group_ on timeout
        // while the task could still use them; keep waiting for actual exit.
        while (!(xEventGroupWaitBits(event_group_, ESP_TCP_EVENT_RECEIVE_TASK_EXIT,
                                     pdFALSE, pdFALSE, pdMS_TO_TICKS(10000)) &
                 ESP_TCP_EVENT_RECEIVE_TASK_EXIT)) {
            ESP_LOGW(TAG, "Still waiting for TCP receive callback to finish");
        }
        receive_task_handle_ = nullptr;
    }
    // shutdown() also releases a blocked sender. Wait for it before closing the
    // descriptor so another socket cannot reuse it during the send operation.
    std::lock_guard<std::mutex> lock(send_mutex_);
    int fd = tcp_fd_.exchange(-1);
    if (fd != -1) {
        close(fd);
    }
}

void EspTcp::RequestDisconnect() {
    connected_ = false;
    int fd = tcp_fd_.load();
    if (fd != -1) {
        shutdown(fd, SHUT_RDWR);
    }
}

int EspTcp::Send(const std::string& data) {
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (!connected_) {
        ESP_LOGE(TAG, "Not connected");
        return -1;
    }

    size_t total_sent = 0;
    size_t data_size = data.size();
    const char* data_ptr = data.data();

    while (total_sent < data_size) {
        int ret = send(tcp_fd_, data_ptr + total_sent, data_size - total_sent, 0);

        if (ret <= 0) {
            ESP_LOGE(TAG, "Send failed: ret=%d, errno=%d", ret, errno);
            last_error_ = ret < 0 ? errno : ECONNRESET;
            // A partly sent WebSocket frame cannot be followed by another one.
            // Do not join here: the receiver may be waiting for the WS send lock.
            RequestDisconnect();
            return -1;
        }

        total_sent += ret;
    }

    return total_sent;
}

void EspTcp::ReceiveTask() {
    std::string data;
    while (connected_) {
        data.resize(1500);
        int ret = recv(tcp_fd_, data.data(), data.size(), 0);
        if (ret <= 0) {
            if (ret < 0) {
                ESP_LOGE(TAG, "TCP receive failed: %d", ret);
            }
            if (ret < 0 && connected_) last_error_ = errno;
            RequestDisconnect();
            break;
        }

        if (connected_ && stream_callback_) {
            data.resize(ret);
            stream_callback_(data);
        }
    }
    // Notify exactly once, from the receive task, before its exit event. Owners
    // can therefore join and destroy every callback capture safely.
    if (disconnect_callback_) {
        disconnect_callback_();
    }
}

int EspTcp::GetLastError() {
    return last_error_;
}
