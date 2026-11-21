// Modified by the RUG Puppy project. See NOTICE.
#pragma once

#include "display.h"
#include <memory>
#include <mutex>
#include <string>
#include <esp_lcd_panel_io.h>
#include <esp_lcd_panel_ops.h>
#include <esp_timer.h>
#include "expression_emote.h"
#include "gfx.h"

namespace emote {

class EmoteDisplay : public Display {
public:
    EmoteDisplay(esp_lcd_panel_handle_t panel, esp_lcd_panel_io_handle_t panel_io, int width, int height);
    virtual ~EmoteDisplay();

    virtual void SetEmotion(const char* emotion) override;
    virtual void SetStatus(const char* status) override;
    virtual void SetChatMessage(const char* role, const char* content) override;
    virtual void SetTheme(Theme* theme) override;
    virtual void ShowNotification(const char* notification, int duration_ms = 3000) override;
    virtual void UpdateStatusBar(bool update_all = false) override;
    virtual void SetPowerSaveMode(bool on) override;
    virtual void SetPreviewRgb565(const void* data, int width, int height, int stride) override;
    virtual void SetPreviewImage(const void* image);
    // Enabled only by the Puppy board; show its name and current activity.
    void SetPetIdentity(const std::string& name, bool online);

    bool StopAnimDialog();
    bool InsertAnimDialog(const char* emoji_name, uint32_t duration_ms);

    void RefreshAll();

    // Get emote handle for internal use
    emote_handle_t GetEmoteHandle() const { return emote_handle_; }

private:
    virtual bool Lock(int timeout_ms = 0) override;
    virtual void Unlock() override;
    void RefreshPetIdentity();
    std::mutex label_mutex_;
    std::string pet_name_text_;
    std::string pet_identity_text_;
    std::string activity_text_;
    std::string rendered_pet_identity_;
    std::string notification_text_;
    int64_t notification_expires_at_us_ = 0;
    esp_timer_handle_t notification_timer_ = nullptr;
    gfx_obj_t* pet_status_label_ = nullptr;
    lv_font_t* pet_status_font_ = nullptr;
    int display_width_ = 0;

    emote_handle_t emote_handle_ = nullptr;
    esp_lcd_panel_io_handle_t panel_io_ = nullptr;
    gfx_obj_t* preview_obj_ = nullptr;  // 预览图像对象
    gfx_image_dsc_t preview_img_dsc_ = {};  // 预览图像描述符
    uint8_t* preview_data_ = nullptr;  // 预览数据副本
    size_t preview_data_size_ = 0;  // 预览数据大小
    esp_timer_handle_t preview_timer_ = nullptr;  // 预览隐藏定时器

};

} // namespace emote
