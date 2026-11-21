// Modified by the RUG Puppy project. See NOTICE.
#include "wifi_board.h"
#include "codecs/no_audio_codec.h"
#include "application.h"
#include "audio_service.h"
#include "ble_remote_control.h"
#include "button.h"
#include "config.h"
#include "mcp_server.h"
#include "esp32_camera.h"
// #include "camera_web_server.h"  // TODO: Migrate camera web server if needed
// #include "lulu_ble.h"  // 暂时屏蔽，启用 BluFi 配网
#include "assets/lang_config.h"
#include "board_config.h"
#include "settings.h"
#include "system_info.h"

#include "display/emote_display.h"

#include <esp_log.h>
#include <esp_heap_caps.h>
#include <cJSON.h>
#include <esp_lcd_panel_vendor.h>
#include <esp_lcd_panel_io.h>
#include <esp_lcd_panel_ops.h>
#include <driver/spi_common.h>
#include <driver/uart.h>
#include <driver/gpio.h>
#include <driver/rtc_io.h>
#include <esp_sleep.h>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <sys/time.h>
#include <utility>
#include <freertos/idf_additions.h>

#include "esp_lcd_gc9a01.h"
#include "xgo.h"
#include "xgo_action.h"
#include "imu.h"
#include "offline_actions.h"
#include "sleep_policy.h"
#include "gateway_config.h"

#define TAG "PUPPY"

static constexpr int kHoldToStopMs = 1000;
static constexpr int kHoldToRebootMs = 5000;
static constexpr int64_t kOfflineConfirmWindowUs = 10000000;
#ifdef CONFIG_PUPPY_STATIONARY_PLAY
static constexpr bool kStationaryPlay = true;
#else
static constexpr bool kStationaryPlay = false;
#endif

class PuppyBoard : public WifiBoard {
public:
    bool UsesLocalCompanion() const override { return true; }
private:
    static_assert(sizeof("sleep_auto") - 1 <= 15, "ESP32 NVS keys are at most 15 bytes");
    struct MotionReceipt {
        char id[65];
        char command[24];
        bool ok;
        const char* error;
        const char* phase;
        bool pending;
        bool acceptance_pending;
    };
    static constexpr size_t kMotionReceiptCount = 64;
    inline static std::atomic<uint8_t> stalled_motors_{0};
    std::unique_ptr<MotionReceipt, decltype(&heap_caps_free)> motion_receipts_{nullptr, heap_caps_free};
    size_t next_motion_receipt_ = 0;
    std::mutex motion_mutex_;
    std::mutex motion_send_mutex_;
    MotionReceipt* active_motion_ = nullptr;
    enum class MotionSource { LocalButton, Gateway };
    MotionSource motion_source_ = MotionSource::LocalButton;
    // A safety cutoff, never a substitute for the action engine's ACTION_DONE.
    // The allowed sequences contain at most 800 control steps (Hug).
    static constexpr int64_t kMotionTimeoutUs = 15000000;
    int64_t motion_deadline_us_ = 0;
    int64_t last_stationary_voltage_poll_us_ = 0;
    // Protected by motion_mutex_. The owning control task clears this before
    // destroying the socket; IsConnected() only reads its atomic flag.
    WebSocket* companion_socket_ = nullptr;
    puppy::OfflineLibrary offline_library_{};
    struct OfflineSequence {
        puppy::OfflineAction action{};
        bool active = false;
        bool step_running = false;
        uint8_t index = 0;
        int64_t wait_until_us = 0;
        int64_t deadline_us = 0;
    } offline_sequence_;
    struct LibrarySaveJob {
        std::string id;
        puppy::OfflineLibrary library{};
        std::string encoded;
        std::atomic<bool> ready{false};
        bool ok = false;
        const char* error = nullptr;
    };
    std::mutex library_job_mutex_;
    std::shared_ptr<LibrarySaveJob> library_save_job_;
    std::atomic<bool> companion_online_{false};
    std::string pet_name_ = "小狗";  // Main application task only.
    struct PhotoJob {
        explicit PhotoJob(std::string request_id) : id(std::move(request_id)), started_us(esp_timer_get_time()) {}
        ~PhotoJob() { heap_caps_free(jpeg); }
        std::string id;
        int64_t started_us;
        std::atomic<bool> ready{false};
        std::atomic<size_t> acknowledged{0};
        uint8_t* jpeg = nullptr;
        size_t length = 0;
        PowerSaveLevel restore_power = PowerSaveLevel::LOW_POWER;
    };
    struct GatewaySaveJob {
        std::string id;
        puppy::GatewayConfig config;
        std::atomic<bool> ready{false};
        bool ok = false;
        const char* error = nullptr;
    };
    puppy::GatewayConfig boot_gateway_;
    std::mutex gateway_job_mutex_;
    std::shared_ptr<GatewaySaveJob> gateway_save_job_;
    bool gateway_recovery_pending_ = false;  // gateway_job_mutex_
    bool gateway_recovery_notice_ = false;  // Main task only.
    Button boot_button_;
    emote::EmoteDisplay* display_ = nullptr;  // AAF动画显示
    Esp32Camera* camera_ = nullptr;  // 初始化为nullptr
    TaskHandle_t xgo_task_handle_ = nullptr;
    TaskHandle_t xgo_rx_task_handle_ = nullptr;
    TaskHandle_t companion_task_handle_ = nullptr;
    std::string companion_control_url_;
    std::string companion_control_token_;
    std::mutex photo_mutex_;
    std::shared_ptr<PhotoJob> photo_job_;
    std::atomic<int64_t> button_press_start_time_{0};
    esp_timer_handle_t long_press_timer_ = nullptr;
    esp_timer_handle_t lamp_test_timer_ = nullptr;
    std::atomic<bool> stop_triggered_{false};
    std::atomic<bool> reboot_requested_{false};
    std::atomic<int64_t> last_user_activity_us_{0};
    std::atomic<bool> entering_deep_sleep_{false};
    std::atomic<bool> auto_sleep_enabled_{false};
    std::atomic<bool> user_interaction_since_boot_{false};
    std::atomic<int64_t> sleep_enabled_at_us_{0};
    const bool woke_by_button_ = esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0;
    bool ignore_wake_button_click_ = woke_by_button_;
    std::atomic<int64_t> offline_confirm_until_us_{0};
    std::atomic<uint32_t> offline_preview_generation_{0};
    std::atomic<uint32_t> offline_confirm_generation_{0};
    uint8_t offline_confirm_index_ = 0;  // Button callback task only.
    uint32_t offline_confirm_version_ = 0;  // Button callback task only.
    uint8_t local_action_index_ = 0;

    void MarkUserActivity() {
        last_user_activity_us_.store(esp_timer_get_time());
    }

    uint8_t SafeActionId(const std::string& name) const {
        if (name == "wave") return Wave_ID;
        if (name == "sit") return Sit_ID;
        if (name == "stretch") return Stretch_ID;
        if (name == "hug") return Hug_ID;
        return 0;
    }

    void FinishActiveMotionLocked(const char* phase, const char* error = nullptr) {
        motion_deadline_us_ = 0;
        motion_source_ = MotionSource::LocalButton;
        if (!active_motion_) return;
        active_motion_->phase = phase;
        active_motion_->ok = strcmp(phase, "completed") == 0;
        active_motion_->error = error;
        active_motion_->pending = true;
        active_motion_ = nullptr;
    }

    bool StartActionLocked(uint8_t action, MotionReceipt* receipt = nullptr,
                           MotionSource source = MotionSource::LocalButton, bool sequence_step = false) {
        if (kStationaryPlay ||
            (action != Wave_ID && action != Sit_ID && action != Stretch_ID && action != Hug_ID) ||
            stalled_motors_.load() != 0 || calibrate_mode != 0 || entering_deep_sleep_.load()) return false;
        if (source == MotionSource::Gateway && (!companion_socket_ || !companion_socket_->IsConnected())) return false;
        if (!sequence_step) offline_sequence_ = {};
        FinishActiveMotionLocked("cancelled", "replaced by another action");
        Clear_State(2);
        active_motion_ = receipt;
        motion_source_ = source;
        motion_deadline_us_ = esp_timer_get_time() + kMotionTimeoutUs;
        Action_ID = action;
        return true;
    }

    bool StartSafeAction(const std::string& name, MotionSource source = MotionSource::LocalButton) {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        return StartActionLocked(SafeActionId(name), nullptr, source);
    }

    void StopMotionLocked(const char* phase = "cancelled", const char* error = "cancelled by stop") {
        offline_sequence_ = {};
        if (stalled_motors_.load()) FinishActiveMotionLocked("failed", "motor stall latched; motion disabled");
        else FinishActiveMotionLocked(phase, error);
        Clear_State(2);
        gpio_set_level(LASER_GPIO, 0);
    }

    bool StartOfflineActionLocked(size_t index) {
        if (kStationaryPlay || index >= offline_library_.count || stalled_motors_.load() ||
            calibrate_mode != 0) return false;
        StopMotionLocked();
        // Copy the approved action: later library saves cannot change this run.
        offline_sequence_.action = offline_library_.actions[index];
        offline_sequence_.active = true;
        offline_sequence_.deadline_us = esp_timer_get_time() + 90000000;
        return true;
    }

    void InstallOfflineLibrary(const puppy::OfflineLibrary& library) {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        offline_library_ = library;
    }

    void StartNextLocalAction() {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        static constexpr uint8_t presets[] = {Wave_ID, Sit_ID, Stretch_ID, Hug_ID};
        const size_t count = 4 + offline_library_.count;
        local_action_index_ %= count;
        size_t index = local_action_index_;
        bool started = index < 4 ? StartActionLocked(presets[index]) : StartOfflineActionLocked(index - 4);
        if (started) local_action_index_ = (index + 1) % count;
    }

    bool StartConfirmedLocalAction(uint8_t index, uint32_t library_version) {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        if (Action_ID || offline_sequence_.active || stalled_motors_.load() || calibrate_mode != 0 ||
            offline_library_.version != library_version) return false;
        static constexpr uint8_t presets[] = {Wave_ID, Sit_ID, Stretch_ID, Hug_ID};
        const size_t count = 4 + offline_library_.count;
        if (index >= count || local_action_index_ % count != index) return false;
        const bool started = index < 4 ? StartActionLocked(presets[index]) :
                                               StartOfflineActionLocked(index - 4);
        if (started) local_action_index_ = (index + 1) % count;
        return started;
    }

    void AdvanceOfflineSequenceLocked(int64_t now) {
        if (!offline_sequence_.active) return;
        auto& sequence = offline_sequence_;
        if (sequence.step_running) {
            if (motion_deadline_us_) return;
            if (ACTION_DONE != 1) { StopMotionLocked(); return; }
            sequence.step_running = false;
            sequence.wait_until_us = now + int64_t(sequence.action.steps[sequence.index].wait_ms) * 1000;
            ++sequence.index;
        }
        if (now < sequence.wait_until_us) return;
        if (sequence.index >= sequence.action.count) { sequence = {}; return; }
        const auto& step = sequence.action.steps[sequence.index];
        if (step.action) {
            if (!StartActionLocked(step.action, nullptr, MotionSource::LocalButton, true)) { StopMotionLocked(); return; }
            sequence.step_running = true;
        } else {
            sequence.wait_until_us = now + int64_t(step.wait_ms) * 1000;
            ++sequence.index;
        }
    }

    void SetCompanionSocket(WebSocket* socket) {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        companion_socket_ = socket;
        if (!socket && motion_deadline_us_ && motion_source_ == MotionSource::Gateway) {
            StopMotionLocked("cancelled", "gateway disconnected");
        }
    }

    void RunMotorControlStep() {
        // All local-companion action writers share this lock. It is never held
        // during a socket send, camera capture, or application task wait.
        std::lock_guard<std::mutex> lock(motion_mutex_);
        if (entering_deep_sleep_.load()) return;
        int64_t now = esp_timer_get_time();
        if (kStationaryPlay) {
            if (now - last_stationary_voltage_poll_us_ >= 60000000) {
                ReadServoVoltage(1);
                last_stationary_voltage_poll_us_ = now;
            }
            return;
        }
        if (offline_sequence_.active && now >= offline_sequence_.deadline_us) {
            StopMotionLocked("failed", "offline action sequence deadline exceeded");
        }
        if (motion_deadline_us_) {
            if (motion_source_ == MotionSource::Gateway &&
                (!companion_socket_ || !companion_socket_->IsConnected())) {
                StopMotionLocked("cancelled", "gateway disconnected");
            } else if (esp_timer_get_time() >= motion_deadline_us_) {
                StopMotionLocked("failed", "action deadline exceeded");
            }
        }
        if (stalled_motors_.load() == 0) xgo_control();
        if (stalled_motors_.load() != 0) {
            offline_sequence_ = {};
            FinishActiveMotionLocked("failed", "motor stall latched; motion disabled");
            Clear_State(2);
            // This task owns UART motor writes during normal operation.
            // Reassert torque-off; no timer or new action can re-enable it.
            for (uint8_t id = 1; id <= MOTOR_NUM; ++id) EnableMotor(id, 0);
        } else if (motion_deadline_us_ && Action_ID == 0) {
            // The action engine sets ACTION_DONE only at its final sequence
            // step. This confirms sequence completion, not measured arrival
            // of every servo at its target angle.
            if (ACTION_DONE) FinishActiveMotionLocked("completed");
            else FinishActiveMotionLocked("cancelled", "motion reset before completion");
        }
        AdvanceOfflineSequenceLocked(esp_timer_get_time());
    }

    bool SendControlReceipt(WebSocket& socket, const char* id, bool ok, const char* error = nullptr,
                            const char* phase = nullptr) {
        cJSON* receipt = cJSON_CreateObject();
        if (!receipt) return false;
        cJSON_AddStringToObject(receipt, "type", "receipt");
        cJSON_AddStringToObject(receipt, "id", id);
        cJSON_AddBoolToObject(receipt, "ok", ok);
        if (phase) cJSON_AddStringToObject(receipt, "phase", phase);
        if (!ok) cJSON_AddStringToObject(receipt, "error", error ? error : "command failed");
        char* payload = cJSON_PrintUnformatted(receipt);
        bool sent = payload && socket.Send(payload);
        cJSON_free(payload);
        cJSON_Delete(receipt);
        return sent;
    }

    MotionReceipt* FindMotionReceiptLocked(const char* id) {
        if (!motion_receipts_) return nullptr;
        for (size_t i = 0; i < kMotionReceiptCount; ++i) {
            if (strcmp(motion_receipts_.get()[i].id, id) == 0) return &motion_receipts_.get()[i];
        }
        return nullptr;
    }

    bool MotionIdUsed(const char* id) {
        std::lock_guard<std::mutex> lock(motion_mutex_);
        return FindMotionReceiptLocked(id) != nullptr;
    }

    bool SendSavedMotionReceipt(WebSocket& socket, const MotionReceipt& requested) {
        // Serialize receipt output without preventing the receive task from
        // applying a stop, or the motor task from finishing/unloading motors.
        std::lock_guard<std::mutex> send_lock(motion_send_mutex_);
        MotionReceipt receipt = requested;
        {
            std::lock_guard<std::mutex> lock(motion_mutex_);
            auto* saved = FindMotionReceiptLocked(requested.id);
            if (saved && strcmp(saved->command, requested.command) == 0) receipt = *saved;
            if (receipt.acceptance_pending) {
                receipt.phase = "accepted";
                receipt.ok = true;
                receipt.error = nullptr;
            }
        }
        if (!SendControlReceipt(socket, receipt.id, receipt.ok, receipt.error, receipt.phase)) return false;
        std::lock_guard<std::mutex> lock(motion_mutex_);
        auto* saved = FindMotionReceiptLocked(receipt.id);
        if (saved && strcmp(saved->command, receipt.command) == 0) {
            if (strcmp(receipt.phase, "accepted") == 0) saved->acceptance_pending = false;
            if (strcmp(saved->phase, receipt.phase) == 0) saved->pending = false;
        }
        return true;
    }

    void FlushMotionReceipts(WebSocket& socket) {
        // At most an acceptance and a terminal receipt for each retained ID.
        for (size_t count = 0; count < 2 * kMotionReceiptCount; ++count) {
            MotionReceipt response{};
            {
                std::lock_guard<std::mutex> lock(motion_mutex_);
                if (!motion_receipts_) return;
                for (size_t i = 0; i < kMotionReceiptCount; ++i) {
                    if (motion_receipts_.get()[i].pending) {
                        response = motion_receipts_.get()[i];
                        break;
                    }
                }
                if (!response.id[0]) return;
                // If completion raced the first send, preserve accepted before
                // terminal order, including after reconnect or a failed send.
                if (response.acceptance_pending) {
                    response.phase = "accepted";
                    response.ok = true;
                    response.error = nullptr;
                }
            }
            if (!SendSavedMotionReceipt(socket, response)) return;
        }
    }

    void HandleMotionCommand(WebSocket& socket, const cJSON* message, const char* id, const char* type) {
        bool is_action = strcmp(type, "action") == 0;
        const cJSON* name = cJSON_GetObjectItemCaseSensitive(message, "name");
        // A small strict schema makes the canonical command independent of JSON
        // property order and prevents unrecorded arguments changing its meaning.
        bool valid = is_action ? cJSON_IsString(name) && strlen(name->valuestring) <= 16 : name == nullptr;
        unsigned id_count = 0, type_count = 0, name_count = 0;
        const cJSON* field = nullptr;
        cJSON_ArrayForEach(field, message) {
            if (!field->string) valid = false;
            else if (strcmp(field->string, "id") == 0) ++id_count;
            else if (strcmp(field->string, "type") == 0) ++type_count;
            else if (strcmp(field->string, "name") == 0) ++name_count;
            else valid = false;
        }
        valid = valid && id_count == 1 && type_count == 1 && name_count == (is_action ? 1u : 0u);
        char command[24] = {};
        if (valid) snprintf(command, sizeof(command), "%s:%s", type, is_action ? name->valuestring : "");
        std::unique_lock<std::mutex> lock(motion_mutex_);
        if (MotionReceipt* prior = FindMotionReceiptLocked(id)) {
            bool same = valid && strcmp(prior->command, command) == 0;
            if (same) prior->pending = true;
            lock.unlock();
            if (!same) SendControlReceipt(socket, id, false, "request id already used for different content", "failed");
            return;
        }
        if (!valid) {
            lock.unlock();
            SendControlReceipt(socket, id, false, "invalid motion command fields", "failed");
            return;
        }
        if (!motion_receipts_) {
            motion_receipts_.reset(static_cast<MotionReceipt*>(heap_caps_calloc(
                kMotionReceiptCount, sizeof(MotionReceipt), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
        }
        MotionReceipt* saved = nullptr;
        if (motion_receipts_) {
            for (size_t offset = 0; offset < kMotionReceiptCount; ++offset) {
                size_t index = (next_motion_receipt_ + offset) % kMotionReceiptCount;
                auto* candidate = &motion_receipts_.get()[index];
                // Retain active requests and unsent terminal notifications.
                if (candidate != active_motion_ && !candidate->pending) {
                    saved = candidate;
                    next_motion_receipt_ = (index + 1) % kMotionReceiptCount;
                    break;
                }
            }
        }
        bool stop = !is_action || strcmp(name->valuestring, "stop") == 0;
        if (!saved) {
            // Physical stop remains available even when receipt storage is full.
            if (stop) StopMotionLocked();
            lock.unlock();
            SendControlReceipt(socket, id, false, "motion receipt cache unavailable", "failed");
            return;
        }
        *saved = MotionReceipt{};
        strcpy(saved->id, id);
        strcpy(saved->command, command);
        saved->pending = true;
        if (stop) {
            StopMotionLocked();
            saved->ok = true;
            saved->phase = "completed";
        } else {
            saved->ok = StartActionLocked(SafeActionId(name->valuestring), saved, MotionSource::Gateway);
            saved->phase = saved->ok ? "accepted" : "failed";
            saved->acceptance_pending = saved->ok;
            if (!saved->ok) {
                if (kStationaryPlay) saved->error = "movement disabled in stationary play";
                else if (stalled_motors_.load()) saved->error = "motor stall latched; motion disabled";
                else saved->error = "action not allowed";
            }
        }
        // The control sender drains this outbox. Return from the TCP receive
        // callback immediately so a stop can be parsed even during send
        // backpressure. The motor task can finish while acceptance is queued.
    }

    void SendControlStatus(WebSocket& socket) {
        struct Snapshot {
            std::atomic<bool> ready{false};
            std::string json;
        };
        auto snapshot = std::make_shared<Snapshot>();
        Application::GetInstance().Schedule([this, snapshot]() {
            cJSON* status = cJSON_Parse(GetDeviceStatusJson().c_str());
            if (status) {
                Settings companion("companion", false);
                cJSON_AddStringToObject(status, "pet_name", companion.GetString("pet_name", "小狗").c_str());
                cJSON_AddBoolToObject(status, "stationary_play", kStationaryPlay);
                char* result = cJSON_PrintUnformatted(status);
                if (result) {
                    snapshot->json = result;
                    cJSON_free(result);
                }
                cJSON_Delete(status);
            }
            snapshot->ready.store(true);
        });
        for (int retry = 0; retry < 100 && !snapshot->ready.load(); ++retry) {
            vTaskDelay(pdMS_TO_TICKS(20));
        }
        // A camera capture can keep the main task busy longer than this wait.
        // Do not read the string until its producer has published completion.
        if (!snapshot->ready.load() || snapshot->json.empty()) return;
        cJSON* message = cJSON_CreateObject();
        cJSON_AddStringToObject(message, "type", "status");
        cJSON_AddItemToObject(message, "status", cJSON_Parse(snapshot->json.c_str()));
        char* payload = cJSON_PrintUnformatted(message);
        if (payload) {
            socket.Send(payload);
            cJSON_free(payload);
        }
        cJSON_Delete(message);
    }

    void SendPhotoReceipt(WebSocket& socket, const PhotoJob& job, bool ok, const char* error = nullptr) {
        cJSON* receipt = cJSON_CreateObject();
        cJSON_AddStringToObject(receipt, "type", "receipt");
        cJSON_AddStringToObject(receipt, "id", job.id.c_str());
        cJSON_AddBoolToObject(receipt, "ok", ok);
        if (ok) cJSON_AddStringToObject(receipt, "photo_id", job.id.c_str());
        if (!ok) cJSON_AddStringToObject(receipt, "error", error ? error : "photo failed");
        char* payload = cJSON_PrintUnformatted(receipt);
        if (payload) socket.Send(payload);
        cJSON_free(payload);
        cJSON_Delete(receipt);
    }

    void SendPhoto(WebSocket& socket, const PhotoJob& job) {
        if (!job.jpeg || job.length == 0 || job.length > 512000) {
            SendPhotoReceipt(socket, job, false, "camera capture failed");
            return;
        }
        SetPowerSaveLevel(PowerSaveLevel::PERFORMANCE);
        cJSON* begin = cJSON_CreateObject();
        cJSON_AddStringToObject(begin, "type", "photo_begin");
        cJSON_AddStringToObject(begin, "id", job.id.c_str());
        cJSON_AddNumberToObject(begin, "length", job.length);
        char* payload = cJSON_PrintUnformatted(begin);
        bool ok = payload && socket.Send(payload);
        cJSON_free(payload);
        cJSON_Delete(begin);
        for (size_t offset = 0; ok && offset < job.length; offset += 1024) {
            size_t count = std::min<size_t>(1024, job.length - offset);
            ESP_LOGI(TAG, "Photo chunk %u/%u", (unsigned)offset, (unsigned)job.length);
            ok = socket.Send(job.jpeg + offset, count, true);
            size_t expected = offset + count;
            for (int retry = 0; ok && retry < 500 && job.acknowledged.load() < expected; ++retry) {
                FlushMotionReceipts(socket);
                vTaskDelay(pdMS_TO_TICKS(10));
            }
            if (ok && job.acknowledged.load() < expected) {
                ESP_LOGE(TAG, "Photo chunk acknowledgement timed out at %u", (unsigned)expected);
                ok = false;
            }
        }
        if (ok) {
            cJSON* end = cJSON_CreateObject();
            cJSON_AddStringToObject(end, "type", "photo_end");
            cJSON_AddStringToObject(end, "id", job.id.c_str());
            char* end_payload = cJSON_PrintUnformatted(end);
            ok = end_payload && socket.Send(end_payload);
            cJSON_free(end_payload);
            cJSON_Delete(end);
        }
        SendPhotoReceipt(socket, job, ok, "photo transfer failed");
        SetPowerSaveLevel(job.restore_power);
    }

    void HandleControlMessage(WebSocket& socket, const char* data, size_t len) {
        // cJSON uses recursive parsing. Bound nesting before entering it on
        // the receive task's small stack (offline libraries need five levels).
        if (len > 8192 || !puppy::BoundedControlJson(data, len)) return;
        const char* end = nullptr;
        cJSON* message = cJSON_ParseWithLengthOpts(data, len, &end, false);
        while (end && end < data + len && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) ++end;
        if (!cJSON_IsObject(message) || end != data + len) {
            cJSON_Delete(message);
            return;
        }
        const cJSON* id = cJSON_GetObjectItem(message, "id");
        const cJSON* type = cJSON_GetObjectItem(message, "type");
        if (!cJSON_IsString(id) || !cJSON_IsString(type) || strlen(id->valuestring) == 0 || strlen(id->valuestring) > 64) {
            cJSON_Delete(message);
            return;
        }
        if (strcmp(type->valuestring, "set_clock") == 0) {
            const cJSON* epoch_ms = cJSON_GetObjectItemCaseSensitive(message, "epoch_ms");
            const double value = cJSON_IsNumber(epoch_ms) ? epoch_ms->valuedouble : 0;
            const bool in_range = std::isfinite(value) && value >= double(puppy::kEarliestTrustedEpoch) * 1000 &&
                                  value < double(puppy::kLatestTrustedEpoch) * 1000;
            const int64_t milliseconds = in_range ? static_cast<int64_t>(value) : 0;
            const bool valid = in_range && value == double(milliseconds) &&
                puppy::HasTrustedClock(static_cast<time_t>(milliseconds / 1000));
            if (valid) {
                const timeval tv = {static_cast<time_t>(milliseconds / 1000),
                                    static_cast<suseconds_t>((milliseconds % 1000) * 1000)};
                if (settimeofday(&tv, nullptr) == 0) {
                    SendControlReceipt(socket, id->valuestring, true, nullptr, "synchronized");
                } else SendControlReceipt(socket, id->valuestring, false, "clock sync failed");
            } else SendControlReceipt(socket, id->valuestring, false, "invalid gateway time");
            cJSON_Delete(message);
            return;
        }
        if (entering_deep_sleep_.load() && strcmp(type->valuestring, "status") != 0 &&
            strcmp(type->valuestring, "stop") != 0) {
            SendControlReceipt(socket, id->valuestring, false, "device entering sleep");
            cJSON_Delete(message);
            return;
        }
        if (strcmp(type->valuestring, "photo_ack") == 0) {
            const cJSON* received = cJSON_GetObjectItem(message, "received");
            std::lock_guard<std::mutex> lock(photo_mutex_);
            if (photo_job_ && photo_job_->id == id->valuestring && cJSON_IsNumber(received) &&
                received->valuedouble >= 0 && received->valuedouble <= 512000) {
                photo_job_->acknowledged.store((size_t)received->valuedouble);
            }
            cJSON_Delete(message);
            return;
        }
        if (strcmp(type->valuestring, "action") == 0 || strcmp(type->valuestring, "stop") == 0 ||
            strcmp(type->valuestring, "photo") == 0 || strcmp(type->valuestring, "set_volume") == 0 ||
            strcmp(type->valuestring, "lamp_test") == 0) {
            MarkUserActivity();
        }
        if (strcmp(type->valuestring, "action") == 0 || strcmp(type->valuestring, "stop") == 0) {
            HandleMotionCommand(socket, message, id->valuestring, type->valuestring);
            cJSON_Delete(message);
            return;
        }
        if (MotionIdUsed(id->valuestring)) {
            SendControlReceipt(socket, id->valuestring, false, "request id already used for different content");
            cJSON_Delete(message);
            return;
        }
        bool ok = false;
        const char* error = "unsupported command";
        if (strcmp(type->valuestring, "lamp_test") == 0) {
            const cJSON* duration = cJSON_GetObjectItemCaseSensitive(message, "duration_ms");
            const int duration_ms = duration ? (cJSON_IsNumber(duration) ? duration->valueint : -1) : 1000;
            if (duration_ms < 1000 || duration_ms > 10000 ||
                (duration && duration->valuedouble != static_cast<double>(duration_ms))) {
                error = "lamp duration must be 1000 to 10000 ms";
            } else if (Application::GetInstance().GetDeviceState() != kDeviceStateIdle ||
                       entering_deep_sleep_.load() || !lamp_test_timer_) {
                error = "lamp test unavailable while busy";
            } else {
                esp_timer_stop(lamp_test_timer_);
                SetLaser(true);
                ok = esp_timer_start_once(lamp_test_timer_, duration_ms * 1000) == ESP_OK;
                if (!ok) {
                    SetLaser(false);
                    error = "lamp test timer failed";
                }
            }
        } else if (strcmp(type->valuestring, "set_sleep_policy") == 0) {
            const cJSON* enabled = cJSON_GetObjectItemCaseSensitive(message, "enabled");
            if (!cJSON_IsBool(enabled)) {
                error = "enabled must be boolean";
            } else {
                const bool requested = cJSON_IsTrue(enabled);
                auto done = std::make_shared<std::atomic<bool>>(false);
                Application::GetInstance().Schedule([this, done, requested]() {
                    Settings companion("companion", true);
                    companion.SetBool("sleep_auto", requested);
                    sleep_enabled_at_us_.store(esp_timer_get_time());
                    auto_sleep_enabled_.store(requested);
                    done->store(true);
                });
                for (int retry = 0; retry < 100 && !done->load(); ++retry) vTaskDelay(pdMS_TO_TICKS(20));
                ok = done->load();
                if (!ok) error = "sleep policy save timed out";
            }
        } else if (strcmp(type->valuestring, "set_gateway_config") == 0) {
            auto job = std::make_shared<GatewaySaveJob>();
            job->id = id->valuestring;
            if (!puppy::ParseGatewayConfig(data, len, true, job->config, error)) {
                SendControlReceipt(socket, id->valuestring, false, error);
            } else {
                bool queued = false;
                {
                    std::lock_guard<std::mutex> lock(gateway_job_mutex_);
                    if (!gateway_save_job_ && !gateway_recovery_pending_) {
                        gateway_save_job_ = job;
                        // Queue under the same lock as physical recovery, so a
                        // pending save cannot overtake a later recovery erase.
                        Application::GetInstance().Schedule([job]() {
                            // Flash writes require the main task's internal stack.
                            // Keep both transports on their boot-time snapshot.
                            job->ok = puppy::SaveGatewayConfig(job->config, job->error);
                            job->ready.store(true);
                        });
                        queued = true;
                    }
                }
                if (!queued) SendControlReceipt(socket, id->valuestring, false, "gateway save already pending");
            }
            cJSON_Delete(message);
            return;
        } else if (strcmp(type->valuestring, "set_offline_actions") == 0) {
            auto job = std::make_shared<LibrarySaveJob>();
            job->id = id->valuestring;
            error = "invalid offline action library";
            if (puppy::HasJsonNulEscape(data, len) ||
                !puppy::ParseOfflineLibrary(message, true, job->library, job->encoded, error)) {
                SendControlReceipt(socket, id->valuestring, false, error);
            } else {
                bool queued = false;
                {
                    std::lock_guard<std::mutex> lock(library_job_mutex_);
                    if (!library_save_job_) { library_save_job_ = job; queued = true; }
                }
                if (!queued) SendControlReceipt(socket, id->valuestring, false, "offline library save already pending");
                else Application::GetInstance().Schedule([this, job]() {
                    job->ok = puppy::SaveOfflineLibrary(job->library, job->encoded, job->error);
                    if (job->ok) InstallOfflineLibrary(job->library);
                    job->ready.store(true);
                });
            }
            cJSON_Delete(message);
            return;
        } else if (strcmp(type->valuestring, "status") == 0) {
            SendControlStatus(socket);
            ok = true;
        } else if (strcmp(type->valuestring, "set_name") == 0) {
            const cJSON* name = cJSON_GetObjectItem(message, "name");
            if (cJSON_IsString(name) && strlen(name->valuestring) > 0 && strlen(name->valuestring) <= 36) {
                // NVS writes disable the flash cache. Run them on the main
                // task's internal stack, never on this PSRAM-backed TCP task.
                auto done = std::make_shared<std::atomic<bool>>(false);
                Application::GetInstance().Schedule([this, done, pet_name = std::string(name->valuestring)]() {
                    {
                        Settings companion("companion", true);
                        companion.SetString("pet_name", pet_name);
                    }
                    pet_name_ = pet_name;
                    display_->SetPetIdentity(pet_name_, companion_online_.load());
                    done->store(true);
                });
                for (int retry = 0; retry < 100 && !done->load(); ++retry) {
                    vTaskDelay(pdMS_TO_TICKS(20));
                }
                ok = done->load();
                if (!ok) error = "name save timed out";
            } else {
                error = "invalid pet name";
            }
        } else if (strcmp(type->valuestring, "set_volume") == 0) {
            const cJSON* value = cJSON_GetObjectItem(message, "value");
            int volume = cJSON_IsNumber(value) ? value->valueint : -1;
            if (cJSON_IsNumber(value) && value->valuedouble == volume && volume >= 0 && volume <= 60) {
                auto done = std::make_shared<std::atomic<bool>>(false);
                Application::GetInstance().Schedule([this, done, volume]() {
                    GetAudioCodec()->SetOutputVolume(volume);
                    Settings companion("companion", true);
                    companion.SetBool("volume_user_set", true);
                    done->store(true);
                });
                for (int retry = 0; retry < 100 && !done->load(); ++retry) {
                    vTaskDelay(pdMS_TO_TICKS(20));
                }
                ok = done->load();
                if (!ok) error = "volume save timed out";
            } else {
                error = "volume must be 0-60";
            }
        } else if (strcmp(type->valuestring, "photo") == 0) {
            std::shared_ptr<PhotoJob> job;
            {
                std::lock_guard<std::mutex> lock(photo_mutex_);
                if (!photo_job_ && !entering_deep_sleep_.load()) {
                    job = std::make_shared<PhotoJob>(id->valuestring);
                    photo_job_ = job;
                }
            }
            if (job) {
                // The receive task must return immediately so TCP and WebSocket
                // frames keep being consumed while the control task sends JPEG.
                Application::GetInstance().Schedule([this, job]() {
                    job->restore_power = Application::GetUserPowerSaveLevel();
                    if (!camera_) InitializeCamera();
                    if (camera_) {
                        try {
                            bool captured = camera_->Capture();
                            if (!captured) {
                                delete camera_;
                                camera_ = nullptr;
                                vTaskDelay(pdMS_TO_TICKS(100));
                                InitializeCamera();
                                captured = camera_ && camera_->Capture();
                            }
                            if (captured) {
                                job->jpeg = camera_->EncodeJpeg(job->length);
                            }
                        } catch (const std::exception& ex) {
                            ESP_LOGE(TAG, "Photo failed: %s", ex.what());
                            if (camera_) camera_->ReleaseFrame();
                        }
                        delete camera_;
                        camera_ = nullptr;
                    }
                    job->ready.store(true);
                });
                cJSON_Delete(message);
                return;
            }
            error = "photo already in progress";
        }
        SendControlReceipt(socket, id->valuestring, ok, error);
        cJSON_Delete(message);
    }

    void FlushLibraryReceipt(WebSocket& socket) {
        std::shared_ptr<LibrarySaveJob> job;
        {
            std::lock_guard<std::mutex> lock(library_job_mutex_);
            job = library_save_job_;
        }
        if (!job || !job->ready.load()) return;
        cJSON* receipt = cJSON_CreateObject();
        if (!receipt) return;
        cJSON_AddStringToObject(receipt, "type", "receipt");
        cJSON_AddStringToObject(receipt, "id", job->id.c_str());
        cJSON_AddBoolToObject(receipt, "ok", job->ok);
        cJSON_AddBoolToObject(receipt, "executed", false);
        if (job->ok) {
            cJSON_AddStringToObject(receipt, "phase", "saved");
            cJSON_AddNumberToObject(receipt, "version", job->library.version);
            cJSON_AddNumberToObject(receipt, "action_count", job->library.count);
        } else cJSON_AddStringToObject(receipt, "error", job->error ? job->error : "offline library save failed");
        char* text = cJSON_PrintUnformatted(receipt);
        bool sent = text && socket.Send(text);
        cJSON_free(text);
        cJSON_Delete(receipt);
        if (sent) {
            std::lock_guard<std::mutex> lock(library_job_mutex_);
            if (library_save_job_ == job) library_save_job_.reset();
        }
    }

    void NotifyCompanionDisplay(bool online) {
        if (companion_online_.exchange(online) == online) return;
        Application::GetInstance().Schedule([this]() {
            if (!gateway_recovery_notice_) display_->SetPetIdentity(pet_name_, companion_online_.load());
        });
    }

    void RequestGatewayRecovery() {
        if (calibrate_mode == 1) return;
        {
            std::lock_guard<std::mutex> lock(motion_mutex_);
            if (Action_ID || offline_sequence_.active) return;
        }
        std::lock_guard<std::mutex> lock(gateway_job_mutex_);
        if (gateway_recovery_pending_) return;
        gateway_recovery_pending_ = true;
        Application::GetInstance().Schedule([this]() {
            const char* error = nullptr;
            const bool ok = puppy::ClearGatewayOverride(error);
            {
                std::lock_guard<std::mutex> lock(gateway_job_mutex_);
                // A save queued before this physical request has now finished.
                // Do not subsequently report that cleared override as installed.
                if (ok) gateway_save_job_.reset();
                gateway_recovery_pending_ = false;
            }
            gateway_recovery_notice_ = true;
            display_->SetPetIdentity(pet_name_ + (ok ? " 网关已恢复，请重启" : " 网关恢复失败"), companion_online_.load());
            if (!ok) ESP_LOGW(TAG, "%s", error ? error : "gateway recovery failed");
        });
    }

    void FlushGatewayReceipt(WebSocket& socket) {
        std::shared_ptr<GatewaySaveJob> job;
        {
            std::lock_guard<std::mutex> lock(gateway_job_mutex_);
            job = gateway_save_job_;
        }
        if (!job || !job->ready.load()) return;
        auto* receipt = cJSON_CreateObject();
        if (!receipt) return;
        cJSON_AddStringToObject(receipt, "type", "receipt");
        cJSON_AddStringToObject(receipt, "id", job->id.c_str());
        cJSON_AddBoolToObject(receipt, "ok", job->ok);
        cJSON_AddBoolToObject(receipt, "executed", false);
        if (job->ok) {
            cJSON_AddStringToObject(receipt, "phase", "saved");
            cJSON_AddNumberToObject(receipt, "version", job->config.version);
            cJSON_AddBoolToObject(receipt, "restart_required", true);
        } else cJSON_AddStringToObject(receipt, "error", job->error ? job->error : "gateway save failed");
        char* text = cJSON_PrintUnformatted(receipt);
        const bool sent = text && socket.Send(text);
        cJSON_free(text);
        cJSON_Delete(receipt);
        if (sent) {
            std::lock_guard<std::mutex> lock(gateway_job_mutex_);
            if (gateway_save_job_ == job) gateway_save_job_.reset();
        }
    }

    void CompanionControlLoop() {
        for (;;) {
            auto socket = GetNetwork()->CreateWebSocket(2);
            if (socket) {
                socket->SetHeader("Authorization", ("Bearer " + companion_control_token_).c_str());
                socket->SetHeader("Device-Id", SystemInfo::GetMacAddress().c_str());
                socket->OnData([this, ptr = socket.get()](const char* data, size_t len, bool binary) {
                    if (!binary) HandleControlMessage(*ptr, data, len);
                });
                socket->OnDisconnected([this]() { SetCompanionSocket(nullptr); NotifyCompanionDisplay(false); });
                SetCompanionSocket(socket.get());
                ESP_LOGW(TAG, "Connecting companion control: %s", companion_control_url_.c_str());
                if (socket->Connect(companion_control_url_.c_str())) {
                    NotifyCompanionDisplay(socket->IsConnected());
                    ESP_LOGW(TAG, "Companion control connected");
                    int64_t last_status_us = 0;
                    while (socket->IsConnected()) {
                        FlushMotionReceipts(*socket);
                        FlushLibraryReceipt(*socket);
                        FlushGatewayReceipt(*socket);
                        int64_t now_us = esp_timer_get_time();
                        if (now_us - last_status_us >= 10000000) {
                            SendControlStatus(*socket);
                            last_status_us = now_us;
                        }
                        std::shared_ptr<PhotoJob> job;
                        {
                            std::lock_guard<std::mutex> lock(photo_mutex_);
                            job = photo_job_;
                        }
                        if (job && job->ready.load()) {
                            SendPhoto(*socket, *job);
                            std::lock_guard<std::mutex> lock(photo_mutex_);
                            if (photo_job_ == job) photo_job_.reset();
                        } else if (job && now_us - job->started_us > 25000000) {
                            SendPhotoReceipt(*socket, *job, false, "photo capture timed out");
                            std::lock_guard<std::mutex> lock(photo_mutex_);
                            if (photo_job_ == job) photo_job_.reset();
                        }
                        vTaskDelay(pdMS_TO_TICKS(100));
                    }
                    std::lock_guard<std::mutex> lock(photo_mutex_);
                    photo_job_.reset();
                } else {
                    ESP_LOGW(TAG, "Companion control connection failed: %d", socket->GetLastError());
                }
                // TCP callbacks are joined by the socket destructor before the
                // next iteration. Repeated disconnect cleanup preserves local
                // button actions and cannot overwrite a terminal receipt.
                SetCompanionSocket(nullptr);
                NotifyCompanionDisplay(false);
            }
            vTaskDelay(pdMS_TO_TICKS(5000));
        }
    }

    void InitializeUart() {
        uart_config_t uart_cfg = {
            .baud_rate = 1000000,
            .data_bits = UART_DATA_8_BITS,
            .parity = UART_PARITY_DISABLE,
            .stop_bits = UART_STOP_BITS_1,
            .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        };
        uart_driver_install(UART_NUM_2, 1024, 1024, 0, NULL, 0);
        uart_param_config(UART_NUM_2, &uart_cfg);
        uart_set_pin(UART_NUM_2, XGO_UART_TX_PIN, XGO_UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    }

    void InitializeLaser() {
        esp_rom_gpio_pad_select_gpio(LASER_GPIO);
        gpio_reset_pin(LASER_GPIO);
        gpio_config_t io_conf = {
            .pin_bit_mask = (1ULL << LASER_GPIO),
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        gpio_config(&io_conf);
        const esp_timer_create_args_t lamp_timer = {
            .callback = [](void* arg) {
                auto* board = static_cast<PuppyBoard*>(arg);
                board->SetLaser(false);
            },
            .arg = this,
            .dispatch_method = ESP_TIMER_TASK,
            .name = "lamp_test",
        };
        ESP_ERROR_CHECK(esp_timer_create(&lamp_timer, &lamp_test_timer_));
    }

    void InitializeBootButton() {
        esp_rom_gpio_pad_select_gpio(GPIO_NUM_0);
        gpio_reset_pin(GPIO_NUM_0);
        gpio_config_t io_conf = {
            .pin_bit_mask = (1ULL << GPIO_NUM_0),
            .mode = GPIO_MODE_INPUT,
            .pull_up_en = GPIO_PULLUP_ENABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        ESP_ERROR_CHECK(gpio_config(&io_conf));
    }

    void InitializeSpi() {
        spi_bus_config_t buscfg = {};
        buscfg.mosi_io_num = DISPLAY_MOSI_PIN;
        buscfg.miso_io_num = GPIO_NUM_NC;
        buscfg.sclk_io_num = DISPLAY_CLK_PIN;
        buscfg.quadwp_io_num = GPIO_NUM_NC;
        buscfg.quadhd_io_num = GPIO_NUM_NC;
        buscfg.max_transfer_sz = DISPLAY_WIDTH * DISPLAY_HEIGHT * sizeof(uint16_t);
        ESP_ERROR_CHECK(spi_bus_initialize(SPI3_HOST, &buscfg, SPI_DMA_CH_AUTO));
    }

    void InitializeLcdDisplay() {
        esp_lcd_panel_io_handle_t panel_io = nullptr;
        esp_lcd_panel_handle_t panel = nullptr;

        ESP_LOGI(TAG, "Install panel IO");
        esp_lcd_panel_io_spi_config_t io_config = {};
        io_config.cs_gpio_num = DISPLAY_CS_PIN;
        io_config.dc_gpio_num = DISPLAY_DC_PIN;
        io_config.spi_mode = DISPLAY_SPI_MODE;
        io_config.pclk_hz = 80 * 1000 * 1000;  // 80MHz SPI
        io_config.trans_queue_depth = 10;
        io_config.lcd_cmd_bits = 8;
        io_config.lcd_param_bits = 8;
        ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(SPI3_HOST, &io_config, &panel_io));

        ESP_LOGI(TAG, "Install GC9A01 LCD driver");
        esp_lcd_panel_dev_config_t panel_config = {};
        panel_config.reset_gpio_num = DISPLAY_RST_PIN;
        panel_config.rgb_ele_order = DISPLAY_RGB_ORDER;
        panel_config.bits_per_pixel = 16;
        // 使用默认GC9A01初始化，不使用自定义gc9107命令
        ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(panel_io, &panel_config, &panel));
        esp_lcd_panel_reset(panel);
        esp_lcd_panel_init(panel);
        esp_lcd_panel_invert_color(panel, DISPLAY_INVERT_COLOR);
        esp_lcd_panel_swap_xy(panel, DISPLAY_SWAP_XY);
        esp_lcd_panel_mirror(panel, DISPLAY_MIRROR_X, DISPLAY_MIRROR_Y);
        esp_lcd_panel_disp_on_off(panel, true);  // 打开显示

        // 使用 EmoteDisplay 播放 AAF 动画
        display_ = new emote::EmoteDisplay(panel, panel_io, DISPLAY_WIDTH, DISPLAY_HEIGHT);
        ESP_LOGI(TAG, "Using EmoteDisplay for AAF animations");
    }

    void InitializeCamera() {
        camera_config_t camera_config = {
            .pin_pwdn = CAMERA_PIN_PWDN,
            .pin_reset = CAMERA_PIN_RESET,
            .pin_xclk = CAMERA_PIN_XCLK,
            .pin_sccb_sda = CAMERA_PIN_SIOD,
            .pin_sccb_scl = CAMERA_PIN_SIOC,
            .pin_d7 = CAMERA_PIN_D7,
            .pin_d6 = CAMERA_PIN_D6,
            .pin_d5 = CAMERA_PIN_D5,
            .pin_d4 = CAMERA_PIN_D4,
            .pin_d3 = CAMERA_PIN_D3,
            .pin_d2 = CAMERA_PIN_D2,
            .pin_d1 = CAMERA_PIN_D1,
            .pin_d0 = CAMERA_PIN_D0,
            .pin_vsync = CAMERA_PIN_VSYNC,
            .pin_href = CAMERA_PIN_HREF,
            .pin_pclk = CAMERA_PIN_PCLK,
            .xclk_freq_hz = XCLK_FREQ_HZ,
            .ledc_timer = LEDC_TIMER_0,
            .ledc_channel = LEDC_CHANNEL_0,
            .pixel_format = PIXFORMAT_RGB565,
            .frame_size = FRAMESIZE_240X240,
            .jpeg_quality = 12,
            .fb_count = 1,
            .fb_location = CAMERA_FB_IN_PSRAM,
            .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
            .sccb_i2c_port = 1,  // 摄像头用GPIO 4/5，IMU用GPIO 48/14，必须用不同端口
        };

        // 先检查摄像头是否可用
        sensor_t* sensor = esp_camera_sensor_get();
        if (sensor != nullptr) {
            // 传感器已存在，不需要重复初始化
            ESP_LOGW(TAG, "Camera sensor already initialized");
            camera_ = nullptr;
            return;
        }

        // DMA straight into PSRAM. An internal 23 KB DMA buffer cannot be
        // allocated after Wi-Fi and the control channel have started.
        esp_camera_set_psram_mode(true);
        camera_ = new Esp32Camera(camera_config);
        
        // 检查摄像头传感器是否成功初始化
        sensor = esp_camera_sensor_get();
        if (sensor != nullptr) {
            ESP_LOGI(TAG, "Camera initialized successfully, sensor PID: 0x%x", sensor->id.PID);
        } else {
            ESP_LOGW(TAG, "Camera initialization failed, camera tools will not be available");
            delete camera_;
            camera_ = nullptr;
        }
    }

    void SetAngle(float a1, float a2, float a3, float a4, float a5, int t) {
        control_mode = 1;
        angle1 = a1;
        angle2 = a2;
        angle3 = a3;
        angle4 = a4;
        angle5 = a5;
        if (t > 0) {
            vTaskDelay(pdMS_TO_TICKS(t));
        }
    }

    void StopMotion() {
        {
            std::lock_guard<std::mutex> lock(motion_mutex_);
            StopMotionLocked();
        }
        auto& app = Application::GetInstance();
        app.Schedule([this]() {
            if (display_) {
                display_->SetEmotion("neutral");
                display_->ShowNotification("已停止");
            }
        });
    }

    void ToggleMute() {
        Application::GetInstance().Schedule([this]() {
            auto* codec = GetAudioCodec();
            Settings companion("companion", true);
            const int current = codec->output_volume();
            if (current > 0) {
                companion.SetInt("unmuted_volume", std::min(current, 60));
                codec->SetOutputVolume(0);
                display_->ShowNotification("已静音");
            } else {
                int restore = companion.GetInt("unmuted_volume", 20);
                if (restore < 1 || restore > 60) restore = 20;
                codec->SetOutputVolume(restore);
                display_->ShowNotification("声音已开启");
            }
            companion.SetBool("volume_user_set", true);
        });
    }

    void HandleOfflineDoubleClick() {
        if (stop_triggered_) return;
        if (kStationaryPlay) {
            display_->ShowNotification("安静陪玩，动作已关闭");
            return;
        }
        const int64_t now = esp_timer_get_time();
        bool busy = false;
        bool confirm = false;
        uint8_t index = 0;
        uint32_t version = 0;
        std::string name;
        {
            std::lock_guard<std::mutex> lock(motion_mutex_);
            busy = Action_ID || offline_sequence_.active || stalled_motors_.load() || calibrate_mode != 0;
            if (!busy) {
                static constexpr const char* names[] = {"挥手", "坐下", "伸展", "抱抱"};
                const size_t count = 4 + offline_library_.count;
                local_action_index_ %= count;
                index = local_action_index_;
                version = offline_library_.version;
                name = index < 4 ? names[index] : offline_library_.actions[index - 4].name;
                confirm = now <= offline_confirm_until_us_.load() &&
                          offline_confirm_generation_.load() == offline_preview_generation_.load() &&
                          index == offline_confirm_index_ && version == offline_confirm_version_;
                if (!confirm) {
                    offline_confirm_index_ = index;
                    offline_confirm_version_ = version;
                }
            }
        }
        if (busy) {
            offline_confirm_until_us_ = 0;
            ++offline_preview_generation_;
            Application::GetInstance().Schedule([this]() {
                display_->ShowNotification("动作进行中");
            });
            return;
        }
        if (!confirm) {
            offline_confirm_until_us_ = 0;
            const uint32_t generation = ++offline_preview_generation_;
            Application::GetInstance().Schedule([this, name, generation]() {
                if (offline_preview_generation_.load() != generation) return;
                display_->ShowNotification(("再双击：" + name).c_str(), 10000);
                offline_confirm_until_us_ = esp_timer_get_time() + kOfflineConfirmWindowUs;
                offline_confirm_generation_ = generation;
            });
            return;
        }
        offline_confirm_until_us_ = 0;
        ++offline_preview_generation_;
        const bool started = StartConfirmedLocalAction(index, version);
        Application::GetInstance().Schedule([this, started, name]() {
            display_->ShowNotification(started ? ("已开始：" + name).c_str() : "离线动作不可用");
        });
    }

    void InitializeButtons() {
        // Stop motors at one second even without Wi-Fi. At five seconds arm a
        // reboot, but wait for release: GPIO0 held low during reset enters the
        // ROM download mode instead of starting the application.
        esp_timer_create_args_t timer_args = {
            .callback = [](void* arg) {
                auto board = static_cast<PuppyBoard*>(arg);
                if (board->button_press_start_time_.load() <= 0 ||
                    gpio_get_level(BOOT_BUTTON_GPIO) != 0) return;
                if (!board->stop_triggered_.exchange(true)) {
                    board->offline_confirm_until_us_ = 0;
                    ++board->offline_preview_generation_;
                    board->StopMotion();
                    ESP_LOGW(TAG, "Button hold: motion stopped");
                    if (board->button_press_start_time_.load() > 0 &&
                        gpio_get_level(BOOT_BUTTON_GPIO) == 0 &&
                        esp_timer_start_once(board->long_press_timer_,
                                             (kHoldToRebootMs - kHoldToStopMs) * 1000) != ESP_OK) {
                        ESP_LOGE(TAG, "Could not arm button reboot timer");
                    }
                } else {
                    board->reboot_requested_ = true;
                    Application::GetInstance().Schedule([board]() {
                        board->display_->ShowNotification("松开后重启");
                    });
                }
            },
            .arg = this,
            .dispatch_method = ESP_TIMER_TASK,
            .name = "stop_timer",
        };
        esp_timer_create(&timer_args, &long_press_timer_);

        // 记录按键按下时间，并启动定时器
        boot_button_.OnPressDown([this]() {
            MarkUserActivity();
            user_interaction_since_boot_.store(true);
            button_press_start_time_ = esp_timer_get_time() / 1000;  // 转换为毫秒
            stop_triggered_ = false;
            reboot_requested_ = false;
            ESP_LOGI(TAG, "Button pressed down");
            esp_timer_start_once(long_press_timer_, kHoldToStopMs * 1000);
        });

        boot_button_.OnPressUp([this]() {
            esp_timer_stop(long_press_timer_);
            button_press_start_time_ = 0;
            if (reboot_requested_.exchange(false) && gpio_get_level(BOOT_BUTTON_GPIO) != 0) {
                Application::GetInstance().Schedule([this]() {
                    display_->ShowNotification("正在重启");
                    Application::GetInstance().Reboot();
                });
            }
        });

        // Everyday conversation is hands-free. A short press only controls
        // whether the speaker is audible; the microphone/wake word stays on.
        boot_button_.OnClick([this]() {
            offline_confirm_until_us_ = 0;
            ++offline_preview_generation_;
            if (ignore_wake_button_click_) {
                ignore_wake_button_click_ = false;
                return;
            }
            if (!stop_triggered_) ToggleMute();
        });

        // First double click previews a local action. Only a second double
        // click inside the confirmation window can start it; speech stays
        // hands-free and one mistaken double click never moves the motors.
        boot_button_.OnDoubleClick([this]() {
            HandleOfflineDoubleClick();
        });

        // No triple-click shortcuts or network/family-data reset.
    }

    void SetDogSpeed(int dog_vx, int dog_vyaw, int time) {
        ESP_LOGI(TAG, "SetDogSpeed: vx=%d, vyaw=%d, time=%d", dog_vx, dog_vyaw, time);
        control_mode = 0;
        motor_speed = 0;
        vx = int(2.2 * dog_vx);
        vyaw = int(2.8 * dog_vyaw);
        if (time > 0) {
            vTaskDelay(pdMS_TO_TICKS(time));
        }
        vx = 0.0;
        vyaw = 0.0;
        ESP_LOGI(TAG, "SetDogSpeed: done");
    }

    void Calibrate(int mode) {
        short mid_pos[] = {1500, 1500, 1500, 1500, 1500};
        if(mode==1 && calibrate_mode==0){
            //printf("Enter calibration mode\n");
            calibrate_mode = 1;
            for(int i=0;i<10;i++){
                SetMotorAngle(mid_pos, 0);
                vTaskDelay(pdMS_TO_TICKS(200));
            }
            EnableAllMotor(0);
            // Show calibration mode emotion
            if (display_) {
                display_->SetEmotion("calibration");
            }
        }
        if(mode==0 && calibrate_mode==1){
            WriteZeroPos();
            EnableAllMotor(1);
            calibrate_mode = 0;
            // Reset to normal emotion when exiting calibration mode
            if (display_) {
                display_->SetEmotion("neutral");
            }
        }
    }

    enum class GpioMode {
        Off = 0,
        On = 1,
        Toggle = 2
    };

    void ControlLaser(GpioMode mode) {
        switch (mode) {
            case GpioMode::Off:
                gpio_set_level(LASER_GPIO, 0);
                break;
            case GpioMode::On:
                gpio_set_level(LASER_GPIO, 1);
                break;
            case GpioMode::Toggle:
                gpio_set_level(LASER_GPIO, 0);
                ESP_LOGI(TAG, "Switch lighting modes");
                gpio_set_level(LASER_GPIO, 1);
                break;
        }
    }

    void InitializeTools() {
        auto& mcp_server = McpServer::GetInstance();

        if (UsesLocalCompanion()) {
            if (!kStationaryPlay) {
                const std::pair<const char*, const char*> actions[] = {
                    {"self.dog.Wave", "wave"}, {"self.dog.Sit", "sit"},
                    {"self.dog.Stretch", "stretch"}, {"self.dog.Hug", "hug"},
                };
                for (const auto& [tool, action] : actions) {
                    mcp_server.AddTool(tool, "Run one permitted pet action; disabled after a motor stall.",
                        PropertyList(), [this, action](const PropertyList&) -> ReturnValue {
                            return StartSafeAction(action, MotionSource::Gateway);
                        });
                }
            }
            mcp_server.AddTool("self.dog.stop", "Stop pet motion and action loops.",
                PropertyList(), [this](const PropertyList&) -> ReturnValue {
                    StopMotion();
                    return true;
                });
            // Laser, raw angles, walking, calibration, BLE and looping tools are
            // not registered on the local child companion, even via old MCP.
            return;
        }

        mcp_server.AddTool("self.dog.move",
            "机器狗移动(vx,vyaw,time),前后移动速度vx(前正后负,0停下)和转向速度vyaw(左转正值,右转负值,0停下),time为移动时间(毫秒),time=0时持续移动,否则移动time毫秒后停止,默认70%幅度",
            PropertyList({
                Property("dog_vx", kPropertyTypeInteger, -100, 100),
                Property("dog_vyaw", kPropertyTypeInteger, -100, 100),
                Property("time", kPropertyTypeInteger, 0, 10000),
            }), [this](const PropertyList& properties) -> ReturnValue {
                int dog_vx = properties["dog_vx"].value<int>();
                int dog_vyaw = properties["dog_vyaw"].value<int>();
                int time = properties["time"].value<int>();
                SetDogSpeed(dog_vx, dog_vyaw, time);
                return true;
            });

        mcp_server.AddTool("self.dog.calibrate",
            "标定机器狗,1为进入标定,0为退出/完成标定",
            PropertyList({
                Property("mode", kPropertyTypeInteger, 0, 1),
            }), [this](const PropertyList& properties) -> ReturnValue {
                int mode = properties["mode"].value<int>();
                ESP_LOGI(TAG, "Calibrate called with mode=%d", mode);
                Calibrate(mode);
                return true;
            });

        mcp_server.AddTool("self.dog.Wave",
            "执行打招呼动作 / Wave hello",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Wave_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Naughty",
            "执行撒娇动作 / Act cute",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Naughty_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Swing",
            "执行前后运动动作 / Swing back and forth",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Swing_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Lookup",
            "执行祈求/抬头动作 / Look up",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Lookup_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Rolling",
            "执行左右摇摆动作 / Roll side to side",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Rolling_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Angry",
            "执行懊悔/生气动作 / Show anger",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Angry_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Swimming",
            "执行游泳动作 / Swimming motion",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Swimming_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Pee",
            "执行撒尿动作 / Pee action",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Pee_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Stretch",
            "执行伸懒腰动作 / Stretch",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Stretch_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Bouncing",
            "执行上下蹲起动作 / Bounce up and down",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Bouncing_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Shaking",
            "执行摇头晃脑动作 / Shake head",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Shaking_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Sit",
            "执行坐下动作 / Sit down",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Sit_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Scratch",
            "执行挠痒动作 / Scratch",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Scratch_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Hug",
            "执行抱抱动作 / Hug",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Action_ID = Hug_ID;
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.Reset",
            "执行复位动作 / Reset to standing",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList& properties) -> ReturnValue {
                Clear_State(2);
                vTaskDelay(pdMS_TO_TICKS(1000));  // 动作间隔
                return true;
            });

        mcp_server.AddTool("self.dog.stop",
            "立即停止移动和动作循环 / Stop all motion",
            PropertyList(std::vector<Property>{}),
            [this](const PropertyList&) -> ReturnValue {
                StopMotion();
                return true;
            });

        mcp_server.AddTool("self.dog.action_loop",
            "设定表演模式/动作循环,1为开始,0为停止",
            PropertyList({
                Property("flag", kPropertyTypeInteger, 0, 1),
            }), [this](const PropertyList& properties) -> ReturnValue {
                int flag = properties["flag"].value<int>();
                set_action_loop_flag(flag);
                return true;
            });

        mcp_server.AddTool("self.laser.control",
            "激光剑控制: 0=关闭, 1=打开, 2=切换激光剑模式",
            PropertyList({
                Property("mode", kPropertyTypeInteger, 0, 2),
            }),
            [this](const PropertyList& properties) -> ReturnValue {
                int modeValue = properties["mode"].value<int>();
                if (modeValue < 0 || modeValue > 2) {
                    ESP_LOGE(TAG, "Invalid mode value: %d", modeValue);
                    return false;
                }
                ControlLaser(static_cast<GpioMode>(modeValue));
                return true;
            });

        mcp_server.AddTool("self.dog.set_motor_angle",
            "即兴动作，设定左前，右前，左后，右后，腰部的五个角度，time为持续时间(毫秒)，前四个角度为0时，四条腿垂直，狗站立，为90时，四条腿水平，狗爬下，第五个角度一般为0，正值上半身向左转，负值上半身向右转。例如默认站立的动作为（40,40,40,40,0）,打招呼动作抬起右前腿(80,120,40,40,30)，如果直接让右前120度会倒，所以左前伸出，且腰向左扭，要让它更加动态可以延时150毫秒，然后多次发右前腿在100、120来回变动的指令。这套随机动作的指令一次最好不要超过15条，需要考虑动作之间延时，最好在100-500毫秒之间，尽可能科学准确地完成用户下达的即兴动作指令。",
            PropertyList({
                Property("angle1", kPropertyTypeInteger, -135.0, 135.0),
                Property("angle2", kPropertyTypeInteger, -135.0, 135.0),
                Property("angle3", kPropertyTypeInteger, -135.0, 135.0),
                Property("angle4", kPropertyTypeInteger, -135.0, 135.0),
                Property("angle5", kPropertyTypeInteger, -30.0, 30.0),
                Property("time", kPropertyTypeInteger, 0, 10000),
            }), [this](const PropertyList& properties) -> ReturnValue {
                float a1 = float(properties["angle1"].value<int>());
                float a2 = float(properties["angle2"].value<int>());
                float a3 = float(properties["angle3"].value<int>());
                float a4 = float(properties["angle4"].value<int>());
                float a5 = float(properties["angle5"].value<int>());
                int time = properties["time"].value<int>();
                SetAngle(a1, a2, a3, a4, a5, time);
                return true;
            });

        // BLE 遥控模式
        mcp_server.AddTool("self.ble.remote_control",
            "开启/关闭蓝牙遥控模式。开启后可用小程序或 APP 遥控机器狗（蓝牙名称与配网时相同）。"
            "enable=1 开启遥控模式, enable=0 关闭遥控模式",
            PropertyList({
                Property("enable", kPropertyTypeInteger, 0, 1),
            }),
            [this](const PropertyList& properties) -> ReturnValue {
                int enable = properties["enable"].value<int>();
                if (enable) {
                    if (!ble_remote_is_running()) {
                        ESP_LOGI(TAG, "Starting BLE remote control mode");
                        // 显示遥控模式表情
                        if (display_) {
                            display_->SetEmotion("remote_mode");
                        }
                        Application::GetInstance().PlaySound(Lang::Sounds::OGG_ENTER_REMOTE());
                        
                        bool success = ble_remote_init();
                        if (success) {
                            ESP_LOGI(TAG, "BLE remote control started");
                            return std::string("蓝牙遥控模式已开启，请用小程序或 APP 连接");
                        } else {
                            ESP_LOGE(TAG, "Failed to start BLE remote control");
                            return std::string("蓝牙遥控模式启动失败");
                        }
                    }
                    return std::string("蓝牙遥控模式已经开启");
                } else {
                    if (ble_remote_is_running()) {
                        ble_remote_deinit();
                        ESP_LOGI(TAG, "BLE remote control stopped");
                        // 恢复表情并播放退出语音
                        if (display_) {
                            display_->SetEmotion("neutral");
                        }
                        Application::GetInstance().PlaySound(Lang::Sounds::OGG_EXIT_REMOTE());
                    }
                    return std::string("蓝牙遥控模式已关闭");
                }
            });
    }

public:
    PuppyBoard() : boot_button_(BOOT_BUTTON_GPIO) {
        InitializeSpi();
        InitializeLcdDisplay();
        InitializeButtons();
        InitializeTools();
        
        InitializeUart();
        InitializeLaser();
        gpio_set_level(LASER_GPIO, 0);
        
        // 初始化舵机零点（不阻塞，标定检查在 CheckCalibration 中进行）
        if (kStationaryPlay) {
            EnableAllMotor(0);
            ESP_LOGI(TAG, "Stationary play: motor output disabled");
        } else {
            InitZeroPos();
        }

        // 立即读取一次电池电压，避免等待 60 秒才有电量数据
        ReadServoVoltage(1);
        last_stationary_voltage_poll_us_ = esp_timer_get_time();
        
        // 注册舵机堵转检测回调
        SetMotorStallCallback([](uint8_t motor_id) {
            ESP_LOGW(TAG, "Motor %d stall event triggered!", motor_id);
            if (motor_id < 1 || motor_id > MOTOR_NUM) return;
            uint8_t previous = stalled_motors_.fetch_or(1u << (motor_id - 1));
            gpio_set_level(LASER_GPIO, 0);
            // Unloading is performed by the independent motor task below, not
            // queued behind a camera capture on the application task.
            if (previous == 0) {
                Application::GetInstance().Schedule([]() {
                    auto display = Board::GetInstance().GetDisplay();
                    if (display) {
                        display->SetEmotion("sad");
                        display->ShowNotification("电机堵转，动作已锁定");
                    }
                });
            }
        });
        EnableStallDetection(true);
        
        InitializeBootButton();
        
        // IMU 初始化
        imu_init();
        
        // XGO 控制任务
        xTaskCreatePinnedToCore([](void* arg) {
            auto* board = static_cast<PuppyBoard*>(arg);
            while (true) {
                board->RunMotorControlStep();
                vTaskDelay(pdMS_TO_TICKS(XGO_TASK_INTERVAL_MS));
            }
            vTaskDelete(NULL);
        }, "xgo_task", 4096, this, 5, &xgo_task_handle_, 0);

        xTaskCreatePinnedToCore([](void* arg) {
            (void)arg;
            while (true) {
                xgo_rx();
                imu_read_once();
                vTaskDelay(pdMS_TO_TICKS(XGO_RX_TASK_INTERVAL_MS));
            }
            vTaskDelete(NULL);
        }, "xgo_rx_task", 4096, this, 5, &xgo_rx_task_handle_, 1);
        ESP_LOGI(TAG, "XGO control tasks created");
    }

    void StartCompanionControl() override {
        if (companion_task_handle_ != nullptr) return;
        std::string url = boot_gateway_.url;
        companion_control_token_ = boot_gateway_.token;
        const std::string suffix = "/v1/audio";
        if (!puppy::ValidGatewayFields(url, companion_control_token_)) {
            ESP_LOGW(TAG, "Companion control disabled: configure gateway token and URL");
            return;
        }
        url.replace(url.size() - suffix.size(), suffix.size(), "/v1/control");
        companion_control_url_ = std::move(url);
        auto created = xTaskCreatePinnedToCoreWithCaps([](void* arg) {
            static_cast<PuppyBoard*>(arg)->CompanionControlLoop();
            vTaskDeleteWithCaps(nullptr);
        }, "puppy_control", 8192, this, 3, &companion_task_handle_, 1,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        ESP_LOGW(TAG, "Control task: result=%d, free_internal=%u, largest_internal=%u",
                 created, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    }

    virtual AudioCodec* GetAudioCodec() override {
#ifdef AUDIO_I2S_METHOD_SIMPLEX
        static NoAudioCodecSimplex audio_codec(AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE,
            AUDIO_I2S_SPK_GPIO_BCLK, AUDIO_I2S_SPK_GPIO_LRCK, AUDIO_I2S_SPK_GPIO_DOUT,
            AUDIO_I2S_MIC_GPIO_SCK, AUDIO_I2S_MIC_GPIO_WS, AUDIO_I2S_MIC_GPIO_DIN);
#else
        static NoAudioCodecDuplex audio_codec(AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE,
            AUDIO_I2S_GPIO_BCLK, AUDIO_I2S_GPIO_WS, AUDIO_I2S_GPIO_DOUT, AUDIO_I2S_GPIO_DIN);
#endif
        return &audio_codec;
    }

    virtual Display* GetDisplay() override {
        return display_;
    }

    virtual Backlight* GetBacklight() override {
        if (DISPLAY_BACKLIGHT_PIN != GPIO_NUM_NC) {
            static PwmBacklight backlight(DISPLAY_BACKLIGHT_PIN, DISPLAY_BACKLIGHT_OUTPUT_INVERT);
            return &backlight;
        }
        return nullptr;
    }

    virtual Camera* GetCamera() override {
        return camera_;
    }

    virtual std::string GetDeviceStatusJson() override {
        std::string base_json = WifiBoard::GetDeviceStatusJson();
        cJSON* root = cJSON_Parse(base_json.c_str());
        if (!root) {
            return base_json;
        }

        imu_read_once();

        cJSON* imu = cJSON_CreateObject();
        cJSON_AddBoolToObject(imu, "initialized", imu_is_initialized());
        cJSON_AddNumberToObject(imu, "roll", roll);
        cJSON_AddNumberToObject(imu, "pitch", pitch);
        cJSON_AddNumberToObject(imu, "yaw", yaw);
        cJSON_AddItemToObject(root, "imu", imu);
        uint8_t stalled = stalled_motors_.load();
        cJSON_AddNumberToObject(root, "stalled_motors", stalled);
        cJSON_AddBoolToObject(root, "motion_locked", stalled != 0);
        cJSON_AddNumberToObject(root, "motion_protocol_version", 2);
        cJSON_AddBoolToObject(root, "auto_sleep_enabled", auto_sleep_enabled_.load());
        cJSON_AddNumberToObject(root, "idle_sleep_minutes", 30);
        cJSON_AddBoolToObject(root, "clock_trusted", puppy::HasTrustedClock(time(nullptr)));
        cJSON_AddBoolToObject(root, "back_lamp_on", GetLaser());
        const float reported_servo_voltage = servo_voltage;
        if (cJSON* battery = cJSON_GetObjectItemCaseSensitive(root, "battery");
            cJSON_IsObject(battery) && std::isfinite(reported_servo_voltage) && reported_servo_voltage >= 0.1f) {
            // This is the servo supply reported by XGO, not a measured charge
            // current or a calibrated battery state of charge.
            cJSON_DeleteItemFromObjectCaseSensitive(battery, "charging");
            cJSON_AddNumberToObject(battery, "servo_voltage_v", reported_servo_voltage);
            cJSON_AddBoolToObject(battery, "charging_known", false);
        }
        const char* gateway_error = nullptr;
        const auto gateway = puppy::LoadGatewayConfig(gateway_error);
        cJSON_AddNumberToObject(root, "gateway_config_version", gateway.version);
        cJSON_AddBoolToObject(root, "gateway_config_override", gateway.overridden);
        cJSON_AddStringToObject(root, "gateway_config_url", gateway.url.c_str());
        cJSON_AddBoolToObject(root, "gateway_config_restart_required", gateway.version != boot_gateway_.version ||
            gateway.url != boot_gateway_.url || gateway.token != boot_gateway_.token);
        {
            std::lock_guard<std::mutex> lock(motion_mutex_);
            cJSON_AddNumberToObject(root, "offline_library_version", offline_library_.version);
            cJSON_AddNumberToObject(root, "offline_library_count", offline_library_.count);
            cJSON_AddBoolToObject(root, "offline_action_running", offline_sequence_.active);
        }

        char* json_str = cJSON_PrintUnformatted(root);
        std::string result(json_str);
        cJSON_free(json_str);
        cJSON_Delete(root);

        return result;
    }

    virtual void OnStartup() override {
        // Keep startup quiet. Repeated reset attempts must never bark or move.
        const char* gateway_error = nullptr;
        boot_gateway_ = puppy::LoadGatewayConfig(gateway_error);
        if (gateway_error) ESP_LOGW(TAG, "%s", gateway_error);
        display_->SetEmotion("neutral");
        auto* codec = GetAudioCodec();
        Settings companion("companion", false);
        pet_name_ = companion.GetString("pet_name", "小狗");
        auto_sleep_enabled_.store(companion.GetBool("sleep_auto", false));
        MarkUserActivity();
        display_->SetPetIdentity(pet_name_, false);
        puppy::OfflineLibrary library{};
        const char* error = nullptr;
        if (puppy::LoadOfflineLibrary(library, error)) {
            InstallOfflineLibrary(library);
        } else ESP_LOGW(TAG, "Offline library unavailable: %s", error ? error : "read error");
        if (!companion.GetBool("volume_user_set", false) && codec->output_volume() > 20) {
            codec->SetOutputVolume(20);
        }
        ESP_LOGI(TAG, "Quiet startup, output volume=%d", codec->output_volume());
    }

    virtual void OnInitializationComplete() override {
        gpio_set_level(LASER_GPIO, 0);  // 初始化完成，关闭激光
        ESP_LOGI(TAG, "Initialization complete, laser off");
    }

    void OnDeviceStateChanged(DeviceState state) override {
        if (state == kDeviceStateConnecting || state == kDeviceStateListening || state == kDeviceStateSpeaking) {
            MarkUserActivity();
            user_interaction_since_boot_.store(true);
        }
        // This is a simple 3 V flowing lamp, not a pixel-addressable strip.
        // It indicates that the child can speak and stays dark in other states.
        SetLaser(state == kDeviceStateListening && stalled_motors_.load() == 0 &&
                 !entering_deep_sleep_.load());
    }

    void OnClockTick() override {
        if (!auto_sleep_enabled_.load() || entering_deep_sleep_.load()) return;
        auto& app = Application::GetInstance();
        if (app.GetDeviceState() != kDeviceStateIdle) {
            MarkUserActivity();
            return;
        }
        const int64_t now_us = esp_timer_get_time();
        const int64_t armed_us = sleep_enabled_at_us_.load();
        if (armed_us > 0 && now_us - armed_us < 5LL * 60 * 1000000) return;
        const time_t epoch = time(nullptr);
        if (!puppy::ShouldDeepSleep(epoch, now_us, last_user_activity_us_.load(),
                                    woke_by_button_ || user_interaction_since_boot_.load(), true)) return;
        if (!app.GetAudioService().IsIdle() || gpio_get_level(BOOT_BUTTON_GPIO) == 0 || calibrate_mode != 0 ||
            (lamp_test_timer_ && esp_timer_is_active(lamp_test_timer_))) return;
        {
            std::scoped_lock lock(motion_mutex_, photo_mutex_);
            if (Action_ID || offline_sequence_.active || active_motion_ || motion_deadline_us_) return;
            if (photo_job_) return;
            if (motion_receipts_) {
                for (size_t i = 0; i < kMotionReceiptCount; ++i) {
                    if (motion_receipts_.get()[i].pending) return;
                }
            }
        }
        const uint64_t until_morning_us = puppy::MicrosecondsUntilNextMorning(epoch);
        if (esp_sleep_enable_ext0_wakeup(BOOT_BUTTON_GPIO, 0) != ESP_OK ||
            rtc_gpio_pullup_en(BOOT_BUTTON_GPIO) != ESP_OK ||
            esp_sleep_enable_timer_wakeup(until_morning_us) != ESP_OK) {
            ESP_LOGE(TAG, "Could not configure deep-sleep wakeup");
            return;
        }
        {
            std::scoped_lock lock(motion_mutex_, photo_mutex_);
            if (Action_ID || offline_sequence_.active || active_motion_ || motion_deadline_us_ || photo_job_) return;
            entering_deep_sleep_.store(true);
        }
        if (!app.PrepareForDeepSleep()) {
            entering_deep_sleep_.store(false);
            MarkUserActivity();
            return;
        }
        {
            std::lock_guard<std::mutex> lock(motion_mutex_);
            StopMotionLocked();
            for (uint8_t id = 1; id <= MOTOR_NUM; ++id) EnableMotor(id, 0);
        }
        SetLaser(false);
        // The state-change event can still be queued when this clock tick
        // starts sleep. Replace any stale "listening" label before LCD sleep.
        display_->SetStatus("休眠中");
        display_->SetEmotion("sleepy");
        vTaskDelay(pdMS_TO_TICKS(150));
        display_->SetPowerSaveMode(true);
        app.GetAudioService().Stop();
        ESP_LOGW(TAG, "Deep sleep until 07:00 China time or top button; %u seconds; button_wake=%d interaction=%d idle_seconds=%d",
                 (unsigned)(until_morning_us / 1000000), (int)woke_by_button_,
                 (int)user_interaction_since_boot_.load(),
                 (int)((esp_timer_get_time() - last_user_activity_us_.load()) / 1000000));
        vTaskDelay(pdMS_TO_TICKS(150));
        esp_deep_sleep_start();
    }

    void SetLaser(bool on) override {
        gpio_set_level(LASER_GPIO, on ? 1 : 0);
    }

    bool GetLaser() override {
        return gpio_get_level(LASER_GPIO) == 1;
    }

    virtual void OnWifiConfigStart() override {
        // A Wi-Fi timeout can enter provisioning without a parent action.
        // Provisioning must never start or replace a motor sequence.
        ESP_LOGI(TAG, "WiFi config started; motion unchanged");
    }

    virtual void OnWifiConfigEnd() override {
        ESP_LOGI(TAG, "WiFi config ended; motion unchanged");
    }

    virtual void CheckCalibration(Display* display, AudioService& audio) override {
        if (kStationaryPlay) return;
        // 检查是否需要标定
        if (calibrate_mode != 1) {
            ESP_LOGI(TAG, "Device already calibrated, skipping calibration");
            return;
        }
        
        ESP_LOGW(TAG, "Device needs calibration, entering calibration mode");
        
        // 显示标定表情
        if (display) {
            display->SetEmotion("calibration");
        }
        
        // 播放进入标定语音
        audio.PlaySound(Lang::Sounds::OGG_CALIBRATION_ENTER());
        
        // 阻塞等待标定完成（用户三击按键退出标定模式）
        ESP_LOGI(TAG, "Waiting for calibration... (triple click to exit)");
        while (calibrate_mode == 1) {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
        
        ESP_LOGI(TAG, "Calibration completed!");
        
        // 标定完成，启用舵机
        for (int i = 0; i < MOTOR_NUM; i++) {
            motor[i].Load = 1;
        }
        
        // 播放退出标定语音
        audio.PlaySound(Lang::Sounds::OGG_CALIBRATION_EXIT());
        
        // 恢复正常表情
        if (display) {
            display->SetEmotion("neutral");
        }
        
        vTaskDelay(pdMS_TO_TICKS(1000));  // 等待语音播放
    }

    // 从舵机 ID=1 的 PRESENT_VOLTAGE 寄存器读取电池电压
    virtual bool GetBatteryLevel(int &level, bool& charging, bool& discharging) override {
        float voltage = servo_voltage;
        if (voltage < 0.1f) return false;  // 尚未读取到有效电压
        
        // 2S 锂电池: 6.6V(~0%) ~ 8.4V(100%)
        if (voltage >= 8.4f)
            level = 100;
        else if (voltage <= 6.6f)
            level = 0;
        else
            level = (int)((voltage - 6.6f) / (8.4f - 6.6f) * 100.0f);
        
        charging = false;
        discharging = true;
        return true;
    }

    virtual std::string GetBoardDescription() override {
        return "一个五自由度四足机器狗，搭载圆形 240x240 LCD 屏幕、ESP32-S3 MCU、8MB PSRAM、"
               "5路总线舵机、IMU 姿态传感器、GC0308 摄像头，支持光剑控制。"
               "2S 锂电池供电(6.6V-8.4V)";
    }
};

void* create_board() {
    const auto cause = esp_sleep_get_wakeup_cause();
    if (cause == ESP_SLEEP_WAKEUP_EXT0 || cause == ESP_SLEEP_WAKEUP_TIMER) {
        rtc_gpio_deinit(BOOT_BUTTON_GPIO);
    }
    return new PuppyBoard();
}
