// Modified by the RUG Puppy project. See NOTICE.
        #include "emote_display.h"

// Standard C++ headers
#include <cstring>
#include <memory>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <cinttypes>

// Standard C headers
#include <sys/time.h>
#include <time.h>

// ESP-IDF headers
#include <esp_log.h>
#include <esp_lcd_panel_io.h>
#include <esp_lcd_panel_commands.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <lvgl.h>

// FreeRTOS headers
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// Project headers
#include "assets/lang_config.h"
#include "assets.h"
#include "board.h"
#include "gfx.h"
#include "expression_emote.h"


namespace emote {

// ============================================================================
// Constants and Type Definitions
// ============================================================================

static const char* TAG = "EmoteDisplay";

// ============================================================================
// Forward Declarations
// ============================================================================

class EmoteDisplay;

// ============================================================================
// Helper Functions
// ============================================================================

static bool OnFlushIoReady(const esp_lcd_panel_io_handle_t panel_io,
    esp_lcd_panel_io_event_data_t* const edata, void* user_ctx)
{
    emote_handle_t handle = static_cast<emote_handle_t>(user_ctx);
    if (handle) {
        emote_notify_flush_finished(handle);
    }
    return true;
}

// Flush callback for emote
static void OnFlushCallback(int x_start, int y_start, int x_end, int y_end, const void* data, emote_handle_t handle)
{
    esp_lcd_panel_handle_t panel = (esp_lcd_panel_handle_t)emote_get_user_data(handle);
    if (panel != nullptr) {
        esp_lcd_panel_draw_bitmap(panel, x_start, y_start, x_end, y_end, data);
    }
}

// ============================================================================
// Graphics Initialization Functions
// ============================================================================

static emote_handle_t InitializeEmote(const esp_lcd_panel_handle_t panel, const int width, const int height)
{
    if (!panel) {
        ESP_LOGE(TAG, "Invalid panel");
        return nullptr;
    }

    emote_config_t emote_cfg = {
        .flags = {
            .swap = true,
            .double_buffer = false,
            .buff_dma = true,
            .buff_spiram = false,
        },
        .gfx_emote = {
            .h_res = width,
            .v_res = height,
            .fps = 30,
        },
        .buffers = {
            .buf_pixels = static_cast<size_t>(width * 16),
        },
        .task = {
            .task_priority = 5,
            .task_stack = 6 * 1024,
            .task_affinity = 0,
            .task_stack_in_ext = false,
        },
        .flush_cb = OnFlushCallback,
        .user_data = (void*)panel,
    };

    emote_handle_t emote_handle = emote_init(&emote_cfg);
    if (!emote_handle) {
        ESP_LOGE(TAG, "Failed to initialize emote");
        return nullptr;
    }

    return emote_handle;
}

// ============================================================================
// EmoteDisplay Class Implementation
// ============================================================================

EmoteDisplay::EmoteDisplay(const esp_lcd_panel_handle_t panel, const esp_lcd_panel_io_handle_t panel_io,
                           const int width, const int height)
{
    display_width_ = width;
    panel_io_ = panel_io;
    emote_handle_ = InitializeEmote(panel, width, height);

    esp_timer_create_args_t notification_timer_args = {
        .callback = [](void* arg) { static_cast<EmoteDisplay*>(arg)->RefreshPetIdentity(); },
        .arg = this,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "pet_notice",
    };
    if (esp_timer_create(&notification_timer_args, &notification_timer_) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create notification timer");
    }

    const esp_lcd_panel_io_callbacks_t cbs = {
        .on_color_trans_done = OnFlushIoReady,
    };
    esp_lcd_panel_io_register_event_callbacks(panel_io, &cbs, emote_handle_);
}

EmoteDisplay::~EmoteDisplay()
{
    if (notification_timer_) {
        esp_timer_stop(notification_timer_);
        esp_timer_delete(notification_timer_);
        notification_timer_ = nullptr;
    }
    // 释放预览定时器
    if (preview_timer_) {
        esp_timer_stop(preview_timer_);
        esp_timer_delete(preview_timer_);
        preview_timer_ = nullptr;
    }
    // 释放预览数据缓冲区
    if (preview_data_) {
        heap_caps_free(preview_data_);
        preview_data_ = nullptr;
        preview_data_size_ = 0;
    }
    // 释放 emote 句柄
    if (emote_handle_) {
        emote_deinit(emote_handle_);
        emote_handle_ = nullptr;
    }
    if (pet_status_font_) {
        gfx_font_lv_delete(pet_status_font_);
        pet_status_font_ = nullptr;
    }
}

void EmoteDisplay::SetEmotion(const char* const emotion)
{
    ESP_LOGI(TAG, "SetEmotion: %s", emotion);
    if (emote_handle_ && emotion && strlen(emotion) > 0) {
        emote_set_anim_emoji(emote_handle_, emotion);
    }
}

void EmoteDisplay::SetChatMessage(const char* const role, const char* const content)
{
    // 不显示对话内容，直接返回
    (void)role;
    (void)content;
}

void EmoteDisplay::SetPetIdentity(const std::string& name, bool online)
{
    {
        std::lock_guard<std::mutex> lock(label_mutex_);
        pet_name_text_ = name;
        pet_identity_text_ = name + (online ? " · 在线" : " · 离线");
    }
    RefreshPetIdentity();
}

void EmoteDisplay::RefreshPetIdentity()
{
    std::lock_guard<std::mutex> lock(label_mutex_);
    if (!emote_handle_) return;
    // Notifications temporarily replace the identity in the same label.
    // Status updates must not erase an active stop confirmation.
    if (notification_expires_at_us_ && esp_timer_get_time() >= notification_expires_at_us_) {
        notification_text_.clear();
        notification_expires_at_us_ = 0;
    }
    std::string text = notification_text_.empty()
        ? (activity_text_.empty() ? pet_identity_text_ :
           (pet_name_text_.empty() ? std::string("小狗") : pet_name_text_) + " · " + activity_text_)
        : notification_text_;
    if (text.empty()) return;
    if (!pet_status_label_) {
        // The stock toast label can be hidden by Emote event changes. Create
        // our own label after assets load so it stays above the eye animation.
        pet_status_label_ = emote_create_obj_by_type(emote_handle_, EMOTE_OBJ_TYPE_LABEL,
                                                      "puppy_status_label");
        if (!pet_status_label_) {
            ESP_LOGW(TAG, "Failed to create Puppy status label");
            return;
        }
        const uint8_t* font_data = nullptr;
        size_t font_size = 0;
        if (emote_get_asset_data_by_name(emote_handle_, "font_puhui_common_20_4.bin",
                                         &font_data, &font_size) == ESP_OK && font_data && font_size) {
            pet_status_font_ = gfx_font_lv_load_from_binary(const_cast<uint8_t*>(font_data));
        }
        emote_lock(emote_handle_);
        gfx_obj_align(pet_status_label_, GFX_ALIGN_TOP_MID, 0, 26);
        gfx_obj_set_size(pet_status_label_, display_width_ - 40, 32);
        gfx_label_set_text_align(pet_status_label_, GFX_TEXT_ALIGN_CENTER);
        if (pet_status_font_) gfx_label_set_font(pet_status_label_, pet_status_font_);
        emote_unlock(emote_handle_);
        ESP_LOGW(TAG, "Puppy status label ready; common_font=%d", pet_status_font_ != nullptr);
    }
    const char* visible_text = pet_status_font_ ? text.c_str() :
        (!notification_text_.empty() ? "STOP" : activity_text_.empty() ? "Puppy" : "LISTEN");
    emote_lock(emote_handle_);
    if (rendered_pet_identity_ != visible_text &&
        gfx_label_set_text(pet_status_label_, visible_text) == ESP_OK) {
        rendered_pet_identity_ = visible_text;
    }
    gfx_obj_set_visible(pet_status_label_, true);
    emote_unlock(emote_handle_);
}

void EmoteDisplay::SetStatus(const char* const status)
{
    ESP_LOGI(TAG, "SetStatus: %s", status);
    if (emote_handle_ && status && strlen(status) > 0) {
        // 统一使用 EMOTE_MGR_EVT_IDLE，避免设置 icon (没有 icon 资源)
        // LISTENING 原本用 EMOTE_MGR_EVT_LISTEN (设置 icon_mic)
        // SPEAKING 原本用 EMOTE_MGR_EVT_SPEAK (设置 icon_speaker)
        emote_set_event_msg(emote_handle_, EMOTE_MGR_EVT_IDLE, NULL);
        std::lock_guard<std::mutex> lock(label_mutex_);
        activity_text_ = strcmp(status, Lang::Strings::STANDBY()) == 0 ? "" : status;
        rendered_pet_identity_.clear();
    }
    RefreshPetIdentity();
}

void EmoteDisplay::ShowNotification(const char* notification, int duration_ms)
{
    ESP_LOGI(TAG, "ShowNotification: %s", notification);
    if (emote_handle_ && notification && strlen(notification) > 0) {
        std::lock_guard<std::mutex> lock(label_mutex_);
        notification_text_ = notification;
        const int visible_ms = duration_ms > 0 ? duration_ms : 3000;
        notification_expires_at_us_ = esp_timer_get_time() + int64_t(visible_ms) * 1000;
        if (notification_timer_) {
            esp_timer_stop(notification_timer_);
            esp_timer_start_once(notification_timer_, int64_t(visible_ms) * 1000);
        }
    }
    RefreshPetIdentity();
}

void EmoteDisplay::UpdateStatusBar(bool update_all)
{
    ESP_LOGD(TAG, "UpdateStatusBar: %s", update_all ? "true" : "false");
    if (!emote_handle_) {
        return;
    }
    RefreshPetIdentity();
}

void EmoteDisplay::SetPowerSaveMode(bool on)
{
    ESP_LOGI(TAG, "SetPowerSaveMode: %s", on ? "ON" : "OFF");
    if (!emote_handle_ || !panel_io_) {
        return;
    }
    auto panel = static_cast<esp_lcd_panel_handle_t>(emote_get_user_data(emote_handle_));
    if (!panel) return;

    if (on) {
        // DISP OFF only hides pixels. Sleep In also stops the LCD controller's
        // internal oscillator and scan circuits before ESP32 deep sleep.
        esp_err_t error = esp_lcd_panel_disp_on_off(panel, false);
        if (error != ESP_OK) {
            ESP_LOGW(TAG, "Panel display-off command failed: %s", esp_err_to_name(error));
        }
        error = esp_lcd_panel_io_tx_param(panel_io_, LCD_CMD_SLPIN, nullptr, 0);
        if (error != ESP_OK) {
            ESP_LOGW(TAG, "Panel sleep-in command failed: %s", esp_err_to_name(error));
        }
    } else {
        esp_err_t error = esp_lcd_panel_io_tx_param(panel_io_, LCD_CMD_SLPOUT, nullptr, 0);
        if (error != ESP_OK) {
            ESP_LOGW(TAG, "Panel sleep-out command failed: %s", esp_err_to_name(error));
            return;
        }
        vTaskDelay(pdMS_TO_TICKS(120));
        error = esp_lcd_panel_disp_on_off(panel, true);
        if (error != ESP_OK) {
            ESP_LOGW(TAG, "Panel display-on command failed: %s", esp_err_to_name(error));
        }
    }
}

void EmoteDisplay::SetPreviewImage(const void* image)
{
    if (image) {
        ESP_LOGI(TAG, "SetPreviewImage: Preview image not supported, using default icon");
    }
}

void EmoteDisplay::SetPreviewRgb565(const void* data, int width, int height, int stride)
{
    if (!emote_handle_) {
        return;
    }

    // 如果 data 为空，隐藏预览图像
    if (data == nullptr) {
        if (preview_obj_) {
            emote_lock(emote_handle_);
            gfx_obj_set_visible(preview_obj_, false);
            emote_unlock(emote_handle_);
        }
        if (preview_timer_) {
            esp_timer_stop(preview_timer_);
        }
        ESP_LOGI(TAG, "SetPreviewRgb565: Hide preview");
        return;
    }

    // 先隐藏旧预览，再更新数据后显示，强制 GFX 引擎重新渲染以避免显示旧图像
    emote_lock(emote_handle_);
    if (preview_obj_) {
        gfx_obj_set_visible(preview_obj_, false);
    }
    emote_unlock(emote_handle_);

    // 创建预览图像对象（如果不存在）
    if (!preview_obj_) {
        preview_obj_ = emote_create_obj_by_type(emote_handle_, EMOTE_OBJ_TYPE_IMAGE, "camera_preview");
        if (!preview_obj_) {
            ESP_LOGE(TAG, "Failed to create preview image object");
            return;
        }

        // 创建预览隐藏定时器
        esp_timer_create_args_t timer_args = {
            .callback = [](void* arg) {
                auto display = static_cast<EmoteDisplay*>(arg);
                if (display && display->preview_obj_ && display->emote_handle_) {
                    emote_lock(display->emote_handle_);
                    gfx_obj_set_visible(display->preview_obj_, false);
                    emote_unlock(display->emote_handle_);
                }
            },
            .arg = this,
            .dispatch_method = ESP_TIMER_TASK,
            .name = "preview_timer",
            .skip_unhandled_events = true,
        };
        esp_timer_create(&timer_args, &preview_timer_);
    }

    // 计算数据大小并复制数据（需要字节交换 RGB565 LE -> BE，LCD 需要大端序）
    size_t data_size = height * stride;
    if (preview_data_size_ < data_size) {
        // 重新分配缓冲区
        if (preview_data_) {
            heap_caps_free(preview_data_);
        }
        preview_data_ = (uint8_t*)heap_caps_malloc(data_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!preview_data_) {
            ESP_LOGE(TAG, "Failed to allocate preview data buffer");
            preview_data_size_ = 0;
            return;
        }
        preview_data_size_ = data_size;
    }
    
    // RGB565 字节交换: 高低字节交换 (LCD 需要大端序)
    const uint16_t* src = reinterpret_cast<const uint16_t*>(data);
    uint16_t* dst = reinterpret_cast<uint16_t*>(preview_data_);
    size_t pixel_count = width * height;
    for (size_t i = 0; i < pixel_count; i++) {
        dst[i] = __builtin_bswap16(src[i]);
    }

    // 设置图像描述符
    preview_img_dsc_.header.magic = 0x19;  // C_ARRAY_HEADER_MAGIC
    preview_img_dsc_.header.cf = GFX_COLOR_FORMAT_RGB565;
    preview_img_dsc_.header.w = width;
    preview_img_dsc_.header.h = height;
    preview_img_dsc_.header.stride = stride;
    preview_img_dsc_.data = preview_data_;
    preview_img_dsc_.data_size = data_size;

    // 设置图像并显示
    emote_lock(emote_handle_);
    gfx_img_set_src(preview_obj_, &preview_img_dsc_);
    gfx_obj_align(preview_obj_, GFX_ALIGN_CENTER, 0, 0);
    gfx_obj_set_visible(preview_obj_, true);
    emote_unlock(emote_handle_);

    // 启动定时器，5秒后自动隐藏
    esp_timer_stop(preview_timer_);
    esp_timer_start_once(preview_timer_, 5000000);  // 5秒

    ESP_LOGI(TAG, "SetPreviewRgb565: %dx%d, stride=%d", width, height, stride);
}

void EmoteDisplay::SetTheme(Theme* const theme)
{
    ESP_LOGI(TAG, "SetTheme: %p", theme);
}

bool EmoteDisplay::Lock(const int timeout_ms)
{
    (void)timeout_ms;
    return true;
}

void EmoteDisplay::Unlock()
{
}

bool EmoteDisplay::StopAnimDialog()
{
    ESP_LOGI(TAG, "StopAnimDialog");
    if (emote_handle_) {
        return emote_stop_anim_dialog(emote_handle_);
    }
    return false;
}

bool EmoteDisplay::InsertAnimDialog(const char* emoji_name, uint32_t duration_ms)
{
    ESP_LOGI(TAG, "InsertAnimDialog: %s, %" PRIu32, emoji_name, duration_ms);
    if (emote_handle_ && emoji_name) {
        return emote_insert_anim_dialog(emote_handle_, emoji_name, duration_ms);
    }
    return false;
}

void EmoteDisplay::RefreshAll()
{
    if (emote_handle_) {
        emote_notify_all_refresh(emote_handle_);
        return;
    }
}

} // namespace emote
