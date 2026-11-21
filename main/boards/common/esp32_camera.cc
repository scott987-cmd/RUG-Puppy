// Modified by the RUG Puppy project. See NOTICE.
#include "sdkconfig.h"

#include <esp_heap_caps.h>
#include <esp_http_client.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <esp_log.h>
#include <img_converters.h>

#include "esp32_camera.h"
#include "board.h"
#include "display.h"
#include "lvgl_display.h"
#include "mcp_server.h"
#include "system_info.h"
#include "jpg/image_to_jpeg.h"
#include "esp_timer.h"

#define TAG "Esp32Camera"

Esp32Camera::Esp32Camera(const camera_config_t &config) {
    fresh_capture_count_ = config.fb_count > 1 ? config.fb_count + 1 : 1;
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed with error 0x%x", err);
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        if (s->id.PID == GC0308_PID) {
            s->set_hmirror(s, 0); // Control camera mirror: 1 for mirror, 0 for normal
        }
        ESP_LOGI(TAG, "Camera initialized: format=%d", config.pixel_format);
    }

    streaming_on_ = true;
}

Esp32Camera::~Esp32Camera() {
    // Explain() can stop the driver before destruction. The separately owned
    // encode buffer still needs releasing in that state.
    if (encode_buf_) {
        heap_caps_free(encode_buf_);
        encode_buf_ = nullptr;
        encode_buf_size_ = 0;
    }
    if (streaming_on_) {
        if (current_fb_) {
            esp_camera_fb_return(current_fb_);
            current_fb_ = nullptr;
        }
        esp_camera_deinit();
        streaming_on_ = false;
    }
}

void Esp32Camera::SetExplainUrl(const std::string &url, const std::string &token) {
    explain_url_ = url;
    explain_token_ = token;
}

bool Esp32Camera::Capture() {
    if (!streaming_on_) {
        return false;
    }

    // One buffer starts DMA on demand; multiple buffers need stale frames drained.
    for (int i = 0; i < fresh_capture_count_; i++) {
        if (current_fb_) {
            esp_camera_fb_return(current_fb_);
        }
        current_fb_ = esp_camera_fb_get();
        if (!current_fb_) {
            vTaskDelay(pdMS_TO_TICKS(100));
            current_fb_ = esp_camera_fb_get();
        }
        if (!current_fb_) {
            ESP_LOGE(TAG, "Camera capture failed");
            return false;
        }
    }

    // Prepare encode buffer for RGB565 format (with optional byte swapping)
    if (current_fb_->format == PIXFORMAT_RGB565) {
        size_t pixel_count = current_fb_->width * current_fb_->height;
        size_t data_size = pixel_count * 2;

        // Allocate or reallocate encode buffer if needed
        if (encode_buf_size_ < data_size) {
            if (encode_buf_) {
                heap_caps_free(encode_buf_);
            }
            encode_buf_ = (uint8_t *)heap_caps_malloc(data_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (encode_buf_ == nullptr) {
                ESP_LOGE(TAG, "Failed to allocate memory for encode buffer");
                encode_buf_size_ = 0;
                return false;
            }
            encode_buf_size_ = data_size;
        }

        // Copy data to encode buffer with optional byte swapping
        uint16_t *src = (uint16_t *)current_fb_->buf;
        uint16_t *dst = (uint16_t *)encode_buf_;
        if (swap_bytes_enabled_) {
            for (size_t i = 0; i < pixel_count; i++) {
                dst[i] = __builtin_bswap16(src[i]);
            }
        } else {
            memcpy(encode_buf_, current_fb_->buf, data_size);
        }

        // Remote Puppy snapshots do not need an on-device preview. Keeping a
        // second 115 KB image and its graphics objects exhausts scarce SRAM.
        if (!Board::GetInstance().UsesLocalCompanion()) {
        // Allocate separate buffer for preview display
        uint8_t *preview_data = (uint8_t *)heap_caps_malloc(data_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (preview_data != nullptr) {
            memcpy(preview_data, encode_buf_, data_size);
            auto display = Board::GetInstance().GetDisplay();
            auto lvgl_display = dynamic_cast<LvglDisplay *>(display);
            if (lvgl_display != nullptr) {
                // LvglDisplay 使用专用接口
                lvgl_display->SetPreviewImage(std::make_unique<LvglAllocatedImage>(preview_data, data_size, current_fb_->width, current_fb_->height, current_fb_->width * 2, LV_COLOR_FORMAT_RGB565));
            } else if (display != nullptr) {
                // 其他 Display 使用通用 RGB565 接口
                display->SetPreviewRgb565(preview_data, current_fb_->width, current_fb_->height, current_fb_->width * 2);
                heap_caps_free(preview_data);  // 通用接口会复制数据，释放临时缓冲区
            } else {
                heap_caps_free(preview_data);
            }
        }
        }
    } else if (current_fb_->format == PIXFORMAT_JPEG) {
        // JPEG format preview usually requires decoding, skip preview display for now, just log
        ESP_LOGW(TAG, "JPEG capture success, len=%u, but not supported for preview", (unsigned)current_fb_->len);
    }

    ESP_LOGI(TAG, "Captured frame: %dx%d, len=%u, format=%d",
             current_fb_->width, current_fb_->height, (unsigned)current_fb_->len, current_fb_->format);

    return true;
}

bool Esp32Camera::SetHMirror(bool enabled) {
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        return false;
    }
    s->set_hmirror(s, enabled ? 1 : 0);
    return true;
}

bool Esp32Camera::SetVFlip(bool enabled) {
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        return false;
    }
    s->set_vflip(s, enabled ? 1 : 0);
    return true;
}

bool Esp32Camera::SetSwapBytes(bool enabled) {
    swap_bytes_enabled_ = enabled;
    return true;
}

uint8_t* Esp32Camera::EncodeJpeg(size_t& jpeg_len) {
    if (current_fb_ == nullptr) {
        throw std::runtime_error("No camera frame captured");
    }
    v4l2_pix_fmt_t enc_fmt;
    switch (current_fb_->format) {
        case PIXFORMAT_RGB565: enc_fmt = V4L2_PIX_FMT_RGB565; break;
        case PIXFORMAT_YUV422: enc_fmt = V4L2_PIX_FMT_YUYV; break;
        case PIXFORMAT_YUV420: enc_fmt = V4L2_PIX_FMT_YUV420; break;
        case PIXFORMAT_GRAYSCALE: enc_fmt = V4L2_PIX_FMT_GREY; break;
        case PIXFORMAT_JPEG: enc_fmt = V4L2_PIX_FMT_JPEG; break;
        case PIXFORMAT_RGB888: enc_fmt = V4L2_PIX_FMT_RGB24; break;
        default: throw std::runtime_error("Unsupported camera pixel format");
    }
    uint8_t* src = current_fb_->buf;
    size_t src_len = current_fb_->len;
    if (current_fb_->format == PIXFORMAT_RGB565 && encode_buf_ != nullptr) {
        src = encode_buf_;
        src_len = encode_buf_size_;
    }
    uint8_t* jpeg = nullptr;
    jpeg_len = 0;
    // Keep local snapshots within the ESP32 TCP send window for reliable
    // delivery over the control channel. Other boards retain full quality.
    uint8_t quality = Board::GetInstance().UsesLocalCompanion() ? 30 : 80;
    bool encoded = image_to_jpeg(src, src_len, current_fb_->width, current_fb_->height,
                                 enc_fmt, quality, &jpeg, &jpeg_len);
    if (!encoded || jpeg == nullptr || jpeg_len == 0 || jpeg_len > 512000) {
        heap_caps_free(jpeg);
        throw std::runtime_error("Failed to encode a small JPEG");
    }
    ReleaseFrame();
    return jpeg;
}

std::string Esp32Camera::Explain(const std::string &question) {
    if (explain_url_.empty()) {
        throw std::runtime_error("Image explain URL or token is not set");
    }
    if (current_fb_ == nullptr) {
        throw std::runtime_error("No camera frame captured");
    }
    const int width = current_fb_->width;
    const int height = current_fb_->height;
    size_t jpeg_len = 0;
    std::unique_ptr<uint8_t, void(*)(void*)> jpeg(EncodeJpeg(jpeg_len), heap_caps_free);
    if (Board::GetInstance().UsesLocalCompanion()) {
        // The Puppy opens the camera only for this photo. Return its DMA
        // descriptors before opening another TCP socket for the upload.
        esp_camera_deinit();
        streaming_on_ = false;
    }

    std::string boundary = "----ESP32_CAMERA_BOUNDARY";
    std::string question_field = "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"question\"\r\n\r\n" + question + "\r\n";
    std::string file_header = "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"camera.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n";
    std::string footer = "\r\n--" + boundary + "--\r\n";
    esp_http_client_config_t config = {};
    config.url = explain_url_.c_str();
    config.timeout_ms = 15000;
    config.buffer_size = 1024;
    config.buffer_size_tx = 1024;
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) throw std::runtime_error("Failed to initialize photo HTTP client");
    esp_http_client_set_method(client, HTTP_METHOD_POST);
    std::string result;
    try {
        auto send_all = [client](const char* data, size_t size) {
            size_t sent = 0;
            while (sent < size) {
                int n = esp_http_client_write(client, data + sent,
                                              std::min<size_t>(4096, size - sent));
                if (n <= 0) throw std::runtime_error("Photo upload interrupted");
                sent += n;
            }
        };
        std::string auth = "Bearer " + explain_token_;
        std::string content_type = "multipart/form-data; boundary=" + boundary;
        esp_http_client_set_header(client, "Device-Id", SystemInfo::GetMacAddress().c_str());
        esp_http_client_set_header(client, "Authorization", auth.c_str());
        esp_http_client_set_header(client, "Content-Type", content_type.c_str());
        size_t body_size = question_field.size() + file_header.size() + jpeg_len + footer.size();
        if (body_size > INT_MAX || esp_http_client_open(client, (int)body_size) != ESP_OK) {
            throw std::runtime_error("Failed to open photo upload");
        }
        send_all(question_field.data(), question_field.size());
        send_all(file_header.data(), file_header.size());
        send_all(reinterpret_cast<const char*>(jpeg.get()), jpeg_len);
        send_all(footer.data(), footer.size());
        if (esp_http_client_fetch_headers(client) < 0 ||
            esp_http_client_get_status_code(client) != 200) {
            throw std::runtime_error("Photo gateway rejected upload");
        }
        char response[256];
        for (;;) {
            int n = esp_http_client_read(client, response, sizeof(response));
            if (n < 0) throw std::runtime_error("Failed to read photo receipt");
            if (n == 0) break;
            if (result.size() + n > 4096) throw std::runtime_error("Photo receipt too large");
            result.append(response, n);
        }
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
    } catch (...) {
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        throw;
    }
    ESP_LOGI(TAG, "Photo uploaded: %dx%d, JPEG bytes=%u", width, height,
             (unsigned)jpeg_len);
    return result;
}

void Esp32Camera::ReleaseFrame() {
    if (current_fb_) {
        esp_camera_fb_return(current_fb_);
        current_fb_ = nullptr;
    }
}
