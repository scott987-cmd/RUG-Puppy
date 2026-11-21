#ifndef _ESP_SSL_H_
#define _ESP_SSL_H_

#include "tcp.h"
#include <esp_tls.h>

#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>
#include <freertos/task.h>
#include <mutex>

#define ESP_SSL_EVENT_RECEIVE_TASK_EXIT 1
#define ESP_SSL_EVENT_STOP 2

class EspSsl : public Tcp {
public:
    EspSsl();
    ~EspSsl();

    bool Connect(const std::string& host, int port) override;
    void Disconnect() override;
    int Send(const std::string& data) override;

    int GetLastError() override;

private:
    // Connect/destruction belong to the owner, never a receive callback.
    // Callbacks may Disconnect() but must return before the owner can destroy.
    esp_tls_t* tls_client_ = nullptr;
    std::mutex lifecycle_mutex_;
    std::mutex io_mutex_;
    std::mutex send_mutex_;
    EventGroupHandle_t event_group_ = nullptr;
    std::atomic<TaskHandle_t> receive_task_handle_{nullptr};
    std::atomic<int> last_error_{0};
    int socket_fd_ = -1; // io_mutex_; only esp_tls_conn_destroy closes it.

    void ReceiveTask();
    void RequestDisconnect();
    bool WaitForRetry();
    void DestroyTls(); // Owner only, after joining all receive/send operations.
};

#endif // _ESP_SSL_H_
