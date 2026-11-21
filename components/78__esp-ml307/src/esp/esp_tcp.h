#ifndef _ESP_TCP_H_
#define _ESP_TCP_H_

#include "tcp.h"

#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>
#include <freertos/task.h>
#include <mutex>

#define ESP_TCP_EVENT_RECEIVE_TASK_EXIT 1

class EspTcp : public Tcp {
public:
    EspTcp();
    ~EspTcp();

    bool Connect(const std::string& host, int port) override;
    void Disconnect() override;
    int Send(const std::string& data) override;

    int GetLastError() override;

private:
    std::atomic<int> tcp_fd_{-1};
    std::mutex send_mutex_;
    EventGroupHandle_t event_group_ = nullptr;
    TaskHandle_t receive_task_handle_ = nullptr;
    std::atomic<int> last_error_{0};

    void ReceiveTask();
    // Stop I/O without joining: safe inside a receive callback or a send lock.
    void RequestDisconnect();
};

#endif // _ESP_TCP_H_
