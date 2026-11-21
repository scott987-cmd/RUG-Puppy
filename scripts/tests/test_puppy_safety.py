#!/usr/bin/env python3
"""Run Puppy safety regressions on the host without connecting to hardware.

The harness compiles the current production action engine and board function
bodies with motor, codec, and socket stubs. It checks action completion/replay,
cancellation, device deadlines, disconnect/provisioning safety, the motor-stall
latch, and MCP mute protection; it does not
establish physical timing, measured servo arrival, or network reliability.

Requirements: Python 3, host clang/clang++, and a local ESP-IDF cJSON checkout.
Set IDF_PATH to ESP-IDF, or PUPPY_CJSON_DIR directly to the directory containing
cJSON.c and cJSON.h. CC and CXX can override the host compilers.

    IDF_PATH=/path/to/esp-idf python3 scripts/tests/test_puppy_safety.py

All generated files are removed from a unique temporary directory after each
run. Firmware sources, build configuration, and devices are left untouched.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_TOKENS = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',
    re.DOTALL,
)


def extract_function(source: str, signature: str) -> str:
    """Extract one target function, ignoring braces in comments and strings."""
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for token in CPP_TOKENS.finditer(source, opening):
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                return source[start:token.end()]
    raise ValueError(f"Unterminated production function: {signature}")


HOST_STUBS = r'''
#include <atomic>
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <memory>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "cJSON.h"

constexpr int MALLOC_CAP_SPIRAM = 1, MALLOC_CAP_8BIT = 2, MOTOR_NUM = 5;
int motor_disable_calls = 0;
int motor_enable_calls = 0, control_calls = 0, calibrate_mode = 0;
int voltage_reads = 0;
void ReadServoVoltage(uint8_t) { ++voltage_reads; }
std::atomic<int64_t> fake_now_us{1000};
int64_t esp_timer_get_time() { return fake_now_us.load(); }
int control_mode = 0;
float vx = 0, vyaw = 0;
uint16_t motor_speed = 0;
uint8_t actionLoop_FLAG = 0;
struct Motor { short DesPos = 0, ZeroPos = 0; };
Motor motor[MOTOR_NUM];
constexpr float PI = 3.14159;
#define ESP_LOGI(...) ((void)0)
#define ESP_LOGD(...) ((void)0)
#define ESP_LOGW(...) ((void)0)
constexpr int LASER_GPIO = 0;
void gpio_set_level(int, int) {}
uint8_t Action_ID = 0;

void EnableMotor(uint8_t, int mode) {
    mode ? ++motor_enable_calls : ++motor_disable_calls;
}
void xgo_control();
void heap_caps_free(void* p) { free(p); }
void* heap_caps_calloc(size_t n, size_t size, int) { return calloc(n, size); }

using nvs_handle_t = unsigned;
using esp_err_t = int;
constexpr int ESP_OK = 0, ESP_FAIL = 1, ESP_ERR_NVS_NOT_FOUND = 2;
constexpr int NVS_READONLY = 0, NVS_READWRITE = 1;
struct nvs_stats_t { size_t available_entries = 0; };
std::string nvs_blob, nvs_pending;
size_t nvs_available = 400;
bool nvs_fail_set = false, nvs_fail_commit = false;
unsigned nvs_writes = 0, nvs_handles = 0;
int nvs_open(const char*, int, nvs_handle_t* handle) { *handle = 1; ++nvs_handles; return ESP_OK; }
void nvs_close(nvs_handle_t) { --nvs_handles; }
int nvs_get_stats(const char*, nvs_stats_t* stats) { stats->available_entries = nvs_available; return ESP_OK; }
int nvs_get_blob(nvs_handle_t, const char*, void* data, size_t* size) {
    if (nvs_blob.empty()) return ESP_ERR_NVS_NOT_FOUND;
    if (!data) { *size = nvs_blob.size(); return ESP_OK; }
    if (*size < nvs_blob.size()) return ESP_FAIL;
    memcpy(data, nvs_blob.data(), nvs_blob.size()); *size = nvs_blob.size(); return ESP_OK;
}
int nvs_set_blob(nvs_handle_t, const char*, const void* data, size_t size) {
    ++nvs_writes;
    if (nvs_fail_set) return ESP_FAIL;
    nvs_pending.assign(static_cast<const char*>(data), size); return ESP_OK;
}
int nvs_commit(nvs_handle_t) {
    if (nvs_fail_commit) return ESP_FAIL;
    nvs_blob = nvs_pending; return ESP_OK;
}

struct WebSocket {
    std::vector<std::string> messages;
    std::atomic<bool> connected{true};
    bool IsConnected() const { return connected.load(); }
    bool fail = false;
    std::function<void()> before_send;
    bool Send(const char* message) {
        if (before_send) { auto callback = std::move(before_send); before_send = {}; callback(); }
        if (fail) return false;
        messages.emplace_back(message);
        return true;
    }
};
struct Codec {
    int volume = 0;
    int output_volume() { return volume; }
    void SetOutputVolume(int value) { volume = value; }
};
struct Board {
    Codec codec;
    bool local = true;
    Codec* GetAudioCodec() { return &codec; }
    bool UsesLocalCompanion() { return local; }
};
struct gfx_obj_t { std::string text; bool visible = false; unsigned writes = 0; };
struct Screen { gfx_obj_t label; bool available = true; int locks = 0; unsigned animations = 0; };
using lv_font_t = int;
using esp_timer_handle_t = void*;
constexpr int EMOTE_OBJ_TYPE_LABEL = 1, GFX_ALIGN_TOP_MID = 1, GFX_TEXT_ALIGN_CENTER = 1;
gfx_obj_t* emote_create_obj_by_type(Screen* screen, int, const char*) { return screen->available ? &screen->label : nullptr; }
int emote_get_asset_data_by_name(Screen*, const char*, const uint8_t** data, size_t* size) {
    static const uint8_t font[] = {1}; *data = font; *size = sizeof(font); return ESP_OK;
}
lv_font_t* gfx_font_lv_load_from_binary(uint8_t*) { static lv_font_t font = 1; return &font; }
void gfx_obj_align(gfx_obj_t*, int, int, int) {}
void gfx_obj_set_size(gfx_obj_t*, int, int) {}
void gfx_label_set_text_align(gfx_obj_t*, int) {}
void gfx_label_set_font(gfx_obj_t*, lv_font_t*) {}
void esp_timer_stop(esp_timer_handle_t) {}
void esp_timer_start_once(esp_timer_handle_t, int64_t) {}
namespace Lang { namespace Strings { const char* STANDBY() { return "idle"; } } }
constexpr const char* EMT_DEF_ELEM_TOAST_LABEL = "toast_label";
constexpr const char* EMOTE_MGR_EVT_IDLE = "idle";
void emote_lock(Screen* screen) { ++screen->locks; }
void emote_unlock(Screen* screen) { --screen->locks; }
gfx_obj_t* emote_get_obj_by_name(Screen* screen, const char*) { return screen->available ? &screen->label : nullptr; }
int gfx_label_set_text(gfx_obj_t* label, const char* text) { label->text = text; ++label->writes; return ESP_OK; }
void gfx_obj_set_visible(gfx_obj_t* label, bool visible) { label->visible = visible; }
void emote_set_event_msg(Screen* screen, const char*, const char*) { screen->label.visible = false; }
void emote_set_anim_emoji(Screen* screen, const char*) { ++screen->animations; }
'''


REGRESSIONS = r'''
int main() {
    Pet pet;
    pet.entering_deep_sleep_ = true;
    assert(!pet.StartSafeAction("wave"));
    int calls_before_sleep = control_calls;
    pet.RunMotorControlStep();
    assert(control_calls == calls_before_sleep);
    pet.entering_deep_sleep_ = false;
    WebSocket socket;
    pet.SetCompanionSocket(&socket);
    auto receipt = [&](const char* id, const char* phase) {
        for (auto it = socket.messages.rbegin(); it != socket.messages.rend(); ++it) {
            auto* parsed = cJSON_Parse(it->c_str());
            assert(parsed);
            auto* item_id = cJSON_GetObjectItem(parsed, "id");
            auto* item_phase = cJSON_GetObjectItem(parsed, "phase");
            bool same = cJSON_IsString(item_id) && strcmp(item_id->valuestring, id) == 0;
            if (same) {
                assert(cJSON_IsString(item_phase));
                assert(strcmp(item_phase->valuestring, phase) == 0);
                bool ok = cJSON_IsTrue(cJSON_GetObjectItem(parsed, "ok"));
                assert(ok == (strcmp(phase, "accepted") == 0 || strcmp(phase, "completed") == 0));
            }
            cJSON_Delete(parsed);
            if (same) return;
        }
        assert(false && "missing receipt");
    };
    auto run = [&](const char* json, bool drain = true) {
        auto* request = cJSON_Parse(json);
        assert(request);
        const auto* id = cJSON_GetObjectItem(request, "id");
        const auto* type = cJSON_GetObjectItem(request, "type");
        pet.HandleMotionCommand(socket, request, id->valuestring, type->valuestring);
        cJSON_Delete(request);
        if (drain) pet.FlushMotionReceipts(socket);
    };
    auto finish = [&]() {
        for (int step = 0; step < 5000 && Action_ID != 0; ++step) {
            fake_now_us += 4000;
            pet.RunMotorControlStep();
        }
        assert(Action_ID == 0 && ACTION_DONE == 1);
        pet.FlushMotionReceipts(socket);
    };

    if (kStationaryPlay) {
        for (const char* action : {"wave", "sit", "stretch", "hug"}) {
            assert(!pet.StartSafeAction(action));
        }
        puppy::OfflineLibrary library{};
        library.count = 1;
        pet.InstallOfflineLibrary(library);
        assert(!pet.StartOfflineActionLocked(0));
        assert(!pet.StartConfirmedLocalAction(0, library.version));
        run(R"({"id":"blocked","type":"action","name":"wave"})");
        receipt("blocked", "failed");
        run(R"({"id":"stop","type":"stop"})");
        receipt("stop", "completed");
        fake_now_us = 61000000;
        pet.RunMotorControlStep();
        pet.RunMotorControlStep();
        assert(voltage_reads == 1 && Action_ID == 0 && control_calls == 0 && motor_enable_calls == 0);
        puts("PASS: stationary play rejects local, offline and gateway motion, with stop and voltage reads intact");
        return 0;
    }

    // Uses the real Wave/Updated_Counter/Clear_State producer, without hardware
    // or sleeps. Accepted cannot become completed merely from polling receipts.
    run(R"({"id":"a","type":"action","name":"wave"})");
    receipt("a", "accepted");
    assert(Action_ID == Wave_ID && Action_Counter[Wave_ID] == 0);
    for (int i = 0; i < 20; ++i) pet.RunMotorControlStep();
    unsigned counter = Action_Counter[Wave_ID];
    run(R"({"name":"wave","type":"action","id":"a"})");
    assert(Action_Counter[Wave_ID] == counter);
    receipt("a", "accepted");
    pet.FlushMotionReceipts(socket);
    receipt("a", "accepted");
    run(R"({"id":"a","type":"action","name":"sit"})");
    receipt("a", "failed");
    assert(Action_ID == Wave_ID && Action_Counter[Wave_ID] == counter);
    run(R"({"id":"a","type":"stop"})");
    receipt("a", "failed");
    assert(Action_ID == Wave_ID);
    finish();
    receipt("a", "completed");
    run(R"({"id":"a","type":"action","name":"wave"})");
    receipt("a", "completed");
    assert(Action_ID == 0);
    puts("PASS: real ACTION_DONE completion, in-progress/final replay, conflicting IDs");

    // Replacing with the same action restarts its actual counter and cancels
    // the old request. An explicit stop has its own completed receipt.
    run(R"({"id":"b","type":"action","name":"wave"})");
    for (int i = 0; i < 20; ++i) pet.RunMotorControlStep();
    run(R"({"id":"c","type":"action","name":"wave"})");
    assert(Action_Counter[Wave_ID] == 0);
    pet.FlushMotionReceipts(socket);
    receipt("b", "cancelled");
    receipt("c", "accepted");
    run(R"({"id":"stop","type":"stop"})");
    pet.FlushMotionReceipts(socket);
    receipt("c", "cancelled");
    receipt("stop", "completed");
    assert(Action_ID == 0 && ACTION_DONE == 0);
    run(R"({"id":"local","type":"action","name":"wave"})");
    assert(pet.StartSafeAction("sit"));
    pet.FlushMotionReceipts(socket);
    receipt("local", "cancelled");
    assert(Action_ID == Sit_ID);
    // Retrying the previous stop must not cancel the newer local action.
    run(R"({"type":"stop","id":"stop"})");
    receipt("stop", "completed");
    assert(Action_ID == Sit_ID);
    run(R"({"id":"stop","type":"stop","name":"wave"})");
    receipt("stop", "failed");
    assert(Action_ID == Sit_ID);
    run(R"({"id":"bad","type":"action","name":"wave","raw_angle":180})");
    receipt("bad", "failed");
    run(R"({"id":"laser","type":"action","name":"laser"})");
    receipt("laser", "failed");
    run(R"({"id":"laser","type":"action","name":"hug"})");
    receipt("laser", "failed");
    run(R"({"id":"duplicate-field","type":"action","name":"wave","name":"sit"})");
    receipt("duplicate-field", "failed");
    assert(Action_ID == Sit_ID);
    puts("PASS: replacement/stop cancellation, fresh counters, whitelist/schema rejection");

    // A blocked first send does not own the motor mutex. Finish the actual
    // action on another thread during Send; acceptance still precedes terminal.
    socket.messages.clear();
    socket.before_send = [&]() {
        std::thread motor_task([&]() {
            for (int i = 0; i < 5000 && Action_ID; ++i) pet.RunMotorControlStep();
            assert(ACTION_DONE == 1);
        });
        motor_task.join();
    };
    run(R"({"id":"send-race","type":"action","name":"hug"})");
    receipt("send-race", "completed");
    assert(socket.messages.size() == 2);
    auto* first_phase = cJSON_Parse(socket.messages.front().c_str());
    assert(strcmp(cJSON_GetObjectItem(first_phase, "phase")->valuestring, "accepted") == 0);
    cJSON_Delete(first_phase);
    // An independent receive callback can apply stop while the sender waits.
    socket.before_send = [&]() {
        std::thread receiver([&]() {
            auto* stop = cJSON_Parse(R"({"id":"send-stop","type":"stop"})");
            pet.HandleMotionCommand(socket, stop, "send-stop", "stop");
            cJSON_Delete(stop);
            assert(Action_ID == 0);
        });
        receiver.join();
    };
    run(R"({"id":"send-cancel","type":"action","name":"wave"})");
    receipt("send-cancel", "cancelled");
    receipt("send-stop", "completed");
    // Failed output is retained across reconnect; no second action execution.
    socket.messages.clear();
    socket.fail = true;
    run(R"({"id":"reconnect","type":"action","name":"stretch"})");
    finish();
    assert(socket.messages.empty());
    socket.fail = false;
    pet.FlushMotionReceipts(socket);
    assert(socket.messages.size() == 2);
    auto* accepted = cJSON_Parse(socket.messages.front().c_str());
    assert(strcmp(cJSON_GetObjectItem(accepted, "phase")->valuestring, "accepted") == 0);
    cJSON_Delete(accepted);
    receipt("reconnect", "completed");
    puts("PASS: completion during socket backpressure and reconnect acceptance ordering");

    // Saturating the bounded outbox cannot evict the active request or prevent
    // a physical stop. All pending terminals remain replayable after recovery.
    run(R"({"id":"cache-active","type":"action","name":"wave"})");
    pet.RunMotorControlStep();
    counter = Action_Counter[Wave_ID];
    socket.fail = true;
    for (int i = 0; i < 80; ++i) {
        std::string request = "{\"id\":\"full-" + std::to_string(i) +
            "\",\"type\":\"action\",\"name\":\"laser\"}";
        run(request.c_str());
    }
    socket.fail = false;
    run(R"({"id":"cache-active","type":"action","name":"wave"})", false);
    receipt("cache-active", "accepted");
    assert(Action_Counter[Wave_ID] == counter);
    run(R"({"id":"cache-stop","type":"stop"})", false);
    receipt("cache-stop", "failed");
    assert(Action_ID == 0);
    pet.FlushMotionReceipts(socket);
    receipt("cache-active", "cancelled");
    puts("PASS: bounded outbox preserves active IDs and permits stop when full");

    // Every permitted sequence completes with the production ACTION_DONE;
    // its cutoff is not used to manufacture a completed acknowledgement.
    for (const char* action : {"wave", "sit", "stretch", "hug"}) {
        std::string id = std::string("normal-") + action;
        std::string request = "{\"id\":\"" + id + "\",\"type\":\"action\",\"name\":\"" + action + "\"}";
        run(request.c_str());
        finish();
        receipt(id.c_str(), "completed");
        assert(pet.motion_deadline_us_ == 0);
    }
    assert(!pet.StartActionLocked(Keep_Sit_ID));
    assert(!pet.StartActionLocked(Sit_Reset_ID));
    puts("PASS: only allowed bounded sequences start and finish before the safety cutoff");

    run(R"({"id":"deadline","type":"action","name":"wave"})");
    const int64_t deadline = pet.motion_deadline_us_;
    fake_now_us = deadline - 1;
    run(R"({"id":"deadline","type":"action","name":"wave"})");
    assert(pet.motion_deadline_us_ == deadline);
    pet.RunMotorControlStep();
    assert(Action_ID == Wave_ID);
    fake_now_us = deadline;
    pet.RunMotorControlStep();
    assert(Action_ID == 0 && ACTION_DONE == 0 && pet.motion_deadline_us_ == 0);
    pet.FlushMotionReceipts(socket);
    receipt("deadline", "failed");
    run(R"({"id":"deadline","type":"action","name":"wave"})");
    receipt("deadline", "failed");
    assert(Action_ID == 0);
    assert(pet.StartSafeAction("hug"));
    assert(pet.motion_deadline_us_ == fake_now_us + Pet::kMotionTimeoutUs);
    fake_now_us = pet.motion_deadline_us_;
    pet.RunMotorControlStep();
    assert(Action_ID == 0 && pet.motion_deadline_us_ == 0);
    puts("PASS: exact deadline failure, no extension on retry, local action cutoff");

    run(R"({"id":"disconnect","type":"action","name":"wave"})");
    // The motor task observes the socket flag even if the sender is blocked.
    socket.connected = false;
    pet.RunMotorControlStep();
    assert(Action_ID == 0);
    pet.SetCompanionSocket(nullptr);
    pet.SetCompanionSocket(nullptr);
    pet.FlushMotionReceipts(socket);
    receipt("disconnect", "cancelled");
    run(R"({"id":"offline-remote","type":"action","name":"wave"})");
    receipt("offline-remote", "failed");
    assert(!pet.StartSafeAction("wave", Pet::MotionSource::Gateway));
    assert(pet.StartSafeAction("sit"));
    const int64_t local_deadline = pet.motion_deadline_us_;
    pet.SetCompanionSocket(nullptr);
    pet.SetCompanionSocket(nullptr);
    assert(Action_ID == Sit_ID && pet.motion_deadline_us_ == local_deadline);
    socket.connected = true;
    pet.SetCompanionSocket(&socket);
    run(R"({"id":"disconnect","type":"action","name":"wave"})");
    receipt("disconnect", "cancelled");
    assert(Action_ID == Sit_ID);
    finish();
    // Legacy safe MCP tools are also gateway-owned and bounded, despite not
    // producing control-channel receipts.
    assert(pet.StartSafeAction("hug", Pet::MotionSource::Gateway));
    pet.SetCompanionSocket(nullptr);
    assert(Action_ID == 0 && pet.motion_deadline_us_ == 0);
    puts("PASS: disconnect cancels only gateway motion, offline buttons and replay remain safe");

    for (int iteration = 0; iteration < 100; ++iteration) {
        pet.SetCompanionSocket(&socket);
        assert(pet.StartSafeAction("wave", Pet::MotionSource::Gateway));
        std::thread disconnect([&]() { pet.SetCompanionSocket(nullptr); });
        std::thread button([&]() { assert(pet.StartSafeAction("sit")); });
        disconnect.join();
        button.join();
        // Both serializations preserve the freshly requested local action.
        assert(Action_ID == Sit_ID && pet.motion_source_ == Pet::MotionSource::LocalButton);
    }
    pet.OnWifiConfigStart();
    pet.OnWifiConfigEnd();
    assert(Action_ID == Sit_ID);
    pet.StopMotionLocked();
    pet.OnWifiConfigStart();
    pet.OnWifiConfigEnd();
    assert(Action_ID == 0);
    pet.SetCompanionSocket(&socket);
    puts("PASS: concurrent disconnect/local action and silent provisioning do not start or replace motion");

    const char* library_json = R"({"id":"lib","type":"set_offline_actions","version":1,"actions":[{"name":"开心舞","steps":[{"action":"wave","wait_ms":500},{"action":"sit"},{"wait_ms":200}]}]})";
    auto parse_library = [&](const std::string& json, puppy::OfflineLibrary& result, std::string& encoded) {
        const char* error = nullptr;
        auto* root = cJSON_Parse(json.c_str());
        bool ok = !puppy::HasJsonNulEscape(json.data(), json.size()) &&
            puppy::ParseOfflineLibrary(root, true, result, encoded, error);
        cJSON_Delete(root); return ok;
    };
    puppy::OfflineLibrary library{};
    std::string encoded;
    assert(parse_library(library_json, library, encoded));
    assert(library.count == 1 && library.actions[0].count == 3 && encoded.size() <= 4096);
    for (const char* invalid : {
        R"({"id":"x","type":"set_offline_actions","version":0,"actions":[]})",
        R"({"id":"x","type":"set_offline_actions","version":1.1,"actions":[]})",
        R"({"id":"x","type":"set_offline_actions","version":2147483648,"actions":[]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"version":2,"actions":[]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[],"execute":true})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{"action":"stop"}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{"action":"wave","angle":90}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{"wait_ms":3001}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{"wait_ms":1.5}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{"wait_ms":1,"wait_ms":2}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a\u0000b","steps":[{"wait_ms":1}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{"wait_ms":1}]},{"name":"a","steps":[{"wait_ms":2}]}]})",
        R"({"id":"x","type":"set_offline_actions","version":1,"actions":[{"name":"a","steps":[{"wait_ms":1},{"wait_ms":1},{"wait_ms":1},{"wait_ms":1},{"wait_ms":1},{"wait_ms":1},{"wait_ms":1}]}]})",
    }) {
        puppy::OfflineLibrary rejected{}; std::string unused;
        assert(!parse_library(invalid, rejected, unused));
    }
    for (const std::string name : {std::string(65, 'a'), std::string("bad\xc0\xaf"), std::string("bad\xf4\x90\x80\x80")}) {
        puppy::OfflineLibrary rejected{}; std::string unused;
        assert(!parse_library("{\"id\":\"x\",\"type\":\"set_offline_actions\",\"version\":1,\"actions\":[{\"name\":\"" + name + "\",\"steps\":[{\"wait_ms\":1}]}]}", rejected, unused));
    }
    std::string maximum = R"({"id":"max","type":"set_offline_actions","version":1,"actions":[)";
    for (int count = 0; count < 4; ++count) {
        if (count) maximum += ',';
        maximum += "{\"name\":\"" + std::string(63, 'a') + std::to_string(count) + "\",\"steps\":[";
        for (int step = 0; step < 6; ++step) {
            if (step) maximum += ',';
            maximum += R"({"action":"wave","wait_ms":3000})";
        }
        maximum += "]}";
        puppy::OfflineLibrary checked{}; std::string unused;
        assert(parse_library(maximum + "]}", checked, unused) == (count < 3));
    }
    puts("PASS: strict offline schema, UTF-8 names, NUL rejection and bounded safe steps");

    const char* storage_error = nullptr;
    assert(puppy::SaveOfflineLibrary(library, encoded, storage_error));
    assert(nvs_writes == 1 && nvs_handles == 0);
    puppy::OfflineLibrary restored{};
    assert(puppy::LoadOfflineLibrary(restored, storage_error));
    assert(puppy::SameOfflineLibrary(library, restored));
    puppy::OfflineLibrary equivalent{}; std::string normalized;
    assert(parse_library(R"({"actions":[{"steps":[{"wait_ms":500,"action":"wave"},{"wait_ms":0,"action":"sit"},{"wait_ms":200}],"name":"开心舞"}],"version":1,"type":"set_offline_actions","id":"another-id"})", equivalent, normalized));
    assert(encoded == normalized);
    assert(puppy::SaveOfflineLibrary(equivalent, normalized, storage_error) && nvs_writes == 1);
    puppy::OfflineLibrary changed{}; std::string changed_json;
    assert(parse_library(R"({"id":"x","type":"set_offline_actions","version":1,"actions":[]})", changed, changed_json));
    assert(!puppy::SaveOfflineLibrary(changed, changed_json, storage_error));
    assert(parse_library(R"({"id":"x","type":"set_offline_actions","version":2,"actions":[]})", changed, changed_json));
    nvs_available = 0;
    assert(!puppy::SaveOfflineLibrary(changed, changed_json, storage_error) && nvs_writes == 1);
    nvs_available = 400; nvs_fail_set = true;
    assert(!puppy::SaveOfflineLibrary(changed, changed_json, storage_error));
    nvs_fail_set = false; nvs_fail_commit = true;
    assert(!puppy::SaveOfflineLibrary(changed, changed_json, storage_error));
    nvs_fail_commit = false;
    assert(puppy::SaveOfflineLibrary(changed, changed_json, storage_error));
    assert(!puppy::SaveOfflineLibrary(library, encoded, storage_error));
    assert(puppy::LoadOfflineLibrary(restored, storage_error) && restored.version == 2 && restored.count == 0);
    assert(nvs_handles == 0);
    nvs_blob = "invalid";
    assert(!puppy::LoadOfflineLibrary(restored, storage_error));
    assert(nvs_handles == 0);
    puts("PASS: NVS reopen, canonical idempotence, stale/conflicting versions and capacity/write failure");

    pet.InstallOfflineLibrary(library);
    assert(Action_ID == 0 && !pet.offline_sequence_.active);
    assert(pet.StartOfflineActionLocked(0));
    socket.connected = false; pet.SetCompanionSocket(nullptr);
    pet.RunMotorControlStep();
    assert(Action_ID == Wave_ID && pet.offline_sequence_.step_running);
    while (Action_ID) { fake_now_us += 4000; pet.RunMotorControlStep(); }
    assert(ACTION_DONE == 1 && pet.offline_sequence_.index == 1);
    auto until = pet.offline_sequence_.wait_until_us;
    assert(until == fake_now_us + 500000);
    // Replacing/clearing the saved library must not mutate or restart the
    // running snapshot, including its post-completion wait.
    pet.InstallOfflineLibrary(changed);
    assert(pet.offline_sequence_.active && pet.offline_sequence_.action.count == 3);
    fake_now_us = until - 1; pet.RunMotorControlStep(); assert(Action_ID == 0);
    fake_now_us = until; pet.RunMotorControlStep(); assert(Action_ID == Sit_ID);
    for (int ticks = 0; pet.offline_sequence_.active && ticks < 5000; ++ticks) {
        fake_now_us += 4000; pet.RunMotorControlStep(); assert(Action_ID != Hug_ID);
    }
    assert(!pet.offline_sequence_.active && Action_ID == 0);
    puts("PASS: offline snapshot survives saving and disconnect, waits only after real completion");

    pet.InstallOfflineLibrary(library);
    pet.local_action_index_ = 0;
    assert(!pet.StartConfirmedLocalAction(1, library.version));
    assert(!pet.StartConfirmedLocalAction(0, library.version + 1));
    assert(pet.StartConfirmedLocalAction(0, library.version));
    assert(Action_ID == Wave_ID);
    assert(!pet.StartConfirmedLocalAction(1, library.version));
    pet.StopMotionLocked();
    pet.local_action_index_ = 0;
    for (uint8_t preset : {Wave_ID, Sit_ID, Stretch_ID, Hug_ID}) {
        pet.StartNextLocalAction(); assert(Action_ID == preset && !pet.offline_sequence_.active);
    }
    pet.StartNextLocalAction(); assert(pet.offline_sequence_.active);
    pet.StartNextLocalAction(); assert(Action_ID == Wave_ID && !pet.offline_sequence_.active);
    assert(pet.StartOfflineActionLocked(0)); pet.RunMotorControlStep();
    pet.StopMotionLocked(); pet.RunMotorControlStep();
    assert(Action_ID == 0 && !pet.offline_sequence_.active);
    assert(pet.StartOfflineActionLocked(0));
    fake_now_us = pet.offline_sequence_.deadline_us;
    pet.RunMotorControlStep(); assert(Action_ID == 0 && !pet.offline_sequence_.active);
    assert(pet.StartOfflineActionLocked(0)); pet.RunMotorControlStep();
    fake_now_us = pet.motion_deadline_us_;
    pet.RunMotorControlStep(); assert(Action_ID == 0 && !pet.offline_sequence_.active);
    assert(pet.StartOfflineActionLocked(0)); pet.RunMotorControlStep();
    assert(pet.StartSafeAction("hug")); assert(!pet.offline_sequence_.active && Action_ID == Hug_ID);
    pet.StopMotionLocked();
    assert(pet.StartOfflineActionLocked(0)); pet.RunMotorControlStep();
    Pet::stalled_motors_ = 2;
    pet.RunMotorControlStep();
    assert(!pet.offline_sequence_.active && Action_ID == 0 && !pet.StartOfflineActionLocked(0));
    // Reset only the simulated fixture, equivalent to a new boot, for the
    // independent remote-action stall test below. Production never does this.
    Pet::stalled_motors_ = 0; motor_disable_calls = 0;
    socket.connected = true; pet.SetCompanionSocket(&socket);
    puts("PASS: button cycle, stop/manual interruption, per-step and whole-library deadlines");

    run(R"({"id":"stall","type":"action","name":"wave"})");
    Pet::stalled_motors_ = 2;
    int previous_control = control_calls;
    pet.RunMotorControlStep();
    pet.RunMotorControlStep();
    assert(control_calls == previous_control && motor_disable_calls == 10 && motor_enable_calls == 0);
    pet.FlushMotionReceipts(socket);
    receipt("stall", "failed");
    assert(!pet.StartSafeAction("wave"));
    run(R"({"id":"locked","type":"action","name":"wave"})");
    receipt("locked", "failed");
    run(R"({"id":"locked-stop","type":"stop"})");
    receipt("locked-stop", "completed");
    assert(Action_ID == 0);
    assert(!pet.StartOfflineActionLocked(0));
    puts("PASS: stall final failure, sustained torque-off, rejected motion and available stop");

    Board board;
    auto rejected = [&](int volume) {
        try {
            SetSpeakerVolumeFromMcp(board, volume);
            return false;
        } catch (const std::exception&) {
            return true;
        }
    };
    assert(rejected(20));
    assert(board.codec.volume == 0);
    assert(rejected(61));
    assert(rejected(-1));
    SetSpeakerVolumeFromMcp(board, 0);
    assert(board.codec.volume == 0);
    board.codec.volume = 20;
    SetSpeakerVolumeFromMcp(board, 60);
    assert(board.codec.volume == 60);
    SetSpeakerVolumeFromMcp(board, 0);
    assert(rejected(60));
    assert(board.codec.volume == 0);
    board.local = false;
    SetSpeakerVolumeFromMcp(board, 100);
    assert(board.codec.volume == 100);
    puts("PASS: MCP volume bounds and mute protection");

    Screen screen;
    PetDisplay display; display.emote_handle_ = &screen;
    display.SetStatus("idle"); assert(!screen.label.visible);
    display.SetPetIdentity("豆豆", false);
    assert(screen.label.text == "豆豆 · 离线" && screen.label.visible && screen.animations == 0);
    display.SetEmotion("happy");
    display.SetStatus("listening");
    unsigned writes = screen.label.writes;
    display.UpdateStatusBar(true);
    assert(screen.label.visible && screen.label.writes == writes && screen.animations == 1);
    display.SetPetIdentity("团团", true);
    assert(screen.label.text == "团团 · listening");
    display.ShowNotification("stopped", 3000);
    assert(screen.label.text == "stopped" && screen.label.visible && screen.animations == 1 && screen.locks == 0);
    fake_now_us += 3000001;
    display.UpdateStatusBar(true);
    assert(screen.label.text == "团团 · listening");
    display.SetStatus("idle");
    screen.available = false; display.SetPetIdentity("小狗", false);
    screen.available = true; display.UpdateStatusBar(true);
    assert(screen.label.text == "小狗 · 离线" && screen.locks == 0);
    puts("PASS: nickname/connection label survives status updates without replacing emotion");
}
'''


def make_harness() -> str:
    board = (REPO_ROOT / "main/boards/puppy/puppy_board.cc").read_text(encoding="utf-8")
    mcp = (REPO_ROOT / "main/mcp_server.cc").read_text(encoding="utf-8")
    actions = (REPO_ROOT / "main/boards/puppy/xgo_action.h").read_text(encoding="utf-8")
    definitions = "\n".join(line for line in actions.splitlines() if not line.startswith("#include"))
    engine = (REPO_ROOT / "main/boards/puppy/xgo_action.cc").read_text(encoding="utf-8")
    engine = "\n".join(line for line in engine.splitlines() if not line.startswith("#include"))
    offline_header = (REPO_ROOT / "main/boards/puppy/offline_actions.h").read_text(encoding="utf-8")
    offline_source = (REPO_ROOT / "main/boards/puppy/offline_actions.cc").read_text(encoding="utf-8")
    offline_code = "\n".join(line for line in (offline_header + "\n" + offline_source).splitlines()
                             if not line.startswith(("#include", "#pragma")))
    fields = board[board.index("    struct MotionReceipt {"):board.index("    struct PhotoJob {")]
    mode_start = board.index("#ifdef CONFIG_PUPPY_STATIONARY_PLAY")
    mode = board[mode_start:board.index("#endif", mode_start) + len("#endif")] + "\n"
    methods = "\n".join(extract_function(board, signature) for signature in (
        "    uint8_t SafeActionId(",
        "    void FinishActiveMotionLocked(",
        "    bool StartActionLocked(",
        "    bool StartSafeAction(",
        "    void StopMotionLocked(",
        "    bool StartOfflineActionLocked(",
        "    void InstallOfflineLibrary(",
        "    void StartNextLocalAction(",
        "    bool StartConfirmedLocalAction(",
        "    void AdvanceOfflineSequenceLocked(",
        "    void SetCompanionSocket(",
        "    void RunMotorControlStep(",
        "    bool SendControlReceipt(",
        "    MotionReceipt* FindMotionReceiptLocked(",
        "    bool MotionIdUsed(",
        "    bool SendSavedMotionReceipt(",
        "    void FlushMotionReceipts(",
        "    void HandleMotionCommand(",
        "    virtual void OnWifiConfigStart(",
        "    virtual void OnWifiConfigEnd(",
    ))
    methods = methods.replace(" override", "")
    volume = extract_function(mcp, "static void SetSpeakerVolumeFromMcp(")
    display_source = (REPO_ROOT / "main/display/emote_display.cc").read_text(encoding="utf-8")
    display_methods = "\n".join(extract_function(display_source, "void EmoteDisplay::" + name + "(")
                                 for name in ("SetPetIdentity", "RefreshPetIdentity", "SetStatus", "ShowNotification", "UpdateStatusBar", "SetEmotion"))
    display_code = ("\nclass PetDisplay { public: Screen* emote_handle_ = nullptr; std::mutex label_mutex_; "
                    "std::string pet_name_text_, pet_identity_text_, activity_text_, rendered_pet_identity_, notification_text_; "
                    "int64_t notification_expires_at_us_ = 0; esp_timer_handle_t notification_timer_ = nullptr; "
                    "gfx_obj_t* pet_status_label_ = nullptr; lv_font_t* pet_status_font_ = nullptr; int display_width_ = 240;\n"
                    + display_methods.replace("EmoteDisplay::", "") + "\n};\n")
    return (
        HOST_STUBS + mode + definitions + engine + offline_code
        + "\nvoid xgo_control() { ++control_calls; if (Action_ID) xgo_action(); }\n"
        + "\nclass Pet { public:\n uint8_t local_action_index_ = 0; std::atomic<bool> entering_deep_sleep_{false};\n"
        + fields + methods + "\n};\n" + display_code + volume + REGRESSIONS
    )


def cjson_directory() -> Path:
    if os.environ.get("PUPPY_CJSON_DIR"):
        directory = Path(os.environ["PUPPY_CJSON_DIR"]).expanduser()
    elif os.environ.get("IDF_PATH"):
        directory = Path(os.environ["IDF_PATH"]).expanduser() / "components/json/cJSON"
    else:
        raise ValueError("Set IDF_PATH to ESP-IDF, or PUPPY_CJSON_DIR to its cJSON source directory.")
    directory = directory.resolve()
    if not all((directory / name).is_file() for name in ("cJSON.c", "cJSON.h")):
        raise ValueError(f"cJSON.c and cJSON.h were not found in {directory}")
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    try:
        cjson = cjson_directory()
        harness = make_harness()
        cc = shlex.split(os.environ.get("CC", "clang"))
        cxx = shlex.split(os.environ.get("CXX", "clang++"))
        if not cc or not cxx:
            raise ValueError("CC and CXX must name host compilers.")
        flags = ["-fsanitize=address,undefined", "-g", "-UNDEBUG"]
        with tempfile.TemporaryDirectory(prefix="puppy-safety-host-") as temporary:
            folder = Path(temporary)
            source = folder / "safety.cc"
            object_file = folder / "cJSON.o"
            executable = folder / "safety-test"
            source.write_text(harness, encoding="utf-8")
            subprocess.run(cc + flags + ["-c", str(cjson / "cJSON.c"), "-o", str(object_file)], check=True)
            for options in ([], ["-DCONFIG_PUPPY_STATIONARY_PLAY=1"]):
                subprocess.run(cxx + ["-std=c++17"] + flags + options + [
                    "-I" + str(cjson), str(source), str(object_file), "-o", str(executable),
                ], check=True)
                subprocess.run([str(executable)], check=True, timeout=30)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Puppy safety regression failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
