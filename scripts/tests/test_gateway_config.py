#!/usr/bin/env python3
"""Compile gateway configuration + control save path with in-memory NVS only.

Requires host clang/clang++ and IDF_PATH or PUPPY_CJSON_DIR, as documented by
test_puppy_safety.py. Never opens devices, a network, or an audio interface.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from test_puppy_safety import REPO_ROOT, cjson_directory, extract_function

NVS_HEADER = r'''
#pragma once
#include <cstddef>
using nvs_handle_t = unsigned;
using esp_err_t = int;
constexpr int ESP_OK = 0, ESP_FAIL = 1, ESP_ERR_NVS_NOT_FOUND = 2;
constexpr int NVS_READONLY = 0, NVS_READWRITE = 1;
struct nvs_stats_t { size_t available_entries; };
int nvs_open(const char*, int, nvs_handle_t*);
void nvs_close(nvs_handle_t);
int nvs_get_blob(nvs_handle_t, const char*, void*, size_t*);
int nvs_get_str(nvs_handle_t, const char*, char*, size_t*);
int nvs_set_blob(nvs_handle_t, const char*, const void*, size_t);
int nvs_get_stats(const char*, nvs_stats_t*);
int nvs_commit(nvs_handle_t);
int nvs_erase_key(nvs_handle_t, const char*);
'''

STUBS = r'''
#include "gateway_config.h"
#include "nvs.h"
#include "cJSON.h"
#include <atomic>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

std::map<std::string, std::string> blobs, legacy;
std::string pending_key, pending_blob;
bool fail_read = false, fail_open = false, fail_set = false, fail_commit = false, fail_stats = false;
bool main_stack = true;
bool pending_erase = false, fail_erase = false;
size_t available = 1000;
unsigned handles = 0, writes = 0;
int nvs_open(const char* ns, int, nvs_handle_t* handle) {
    assert(!strcmp(ns, "companion"));
    if (fail_open) return ESP_FAIL;
    *handle = 1; ++handles; return ESP_OK;
}
void nvs_close(nvs_handle_t) { assert(handles); --handles; }
int read_value(const std::map<std::string, std::string>& values, const char* key, void* data, size_t* size, bool nul) {
    if (fail_read) return ESP_FAIL;
    auto item = values.find(key);
    if (item == values.end()) return ESP_ERR_NVS_NOT_FOUND;
    size_t needed = item->second.size() + (nul ? 1 : 0);
    if (!data) { *size = needed; return ESP_OK; }
    if (*size < needed) return ESP_FAIL;
    memcpy(data, item->second.c_str(), needed); *size = needed; return ESP_OK;
}
int nvs_get_blob(nvs_handle_t, const char* key, void* data, size_t* size) {
    return read_value(blobs, key, data, size, false);
}
int nvs_get_str(nvs_handle_t, const char* key, char* data, size_t* size) {
    return read_value(legacy, key, data, size, true);
}
int nvs_set_blob(nvs_handle_t, const char* key, const void* data, size_t size) {
    assert(main_stack); assert(!strcmp(key, "gateway_config"));
    ++writes;
    if (fail_set) return ESP_FAIL;
    pending_erase = false; pending_key = key; pending_blob.assign(static_cast<const char*>(data), size); return ESP_OK;
}
int nvs_get_stats(const char*, nvs_stats_t* stats) {
    stats->available_entries = available; return fail_stats ? ESP_FAIL : ESP_OK;
}
int nvs_commit(nvs_handle_t) {
    assert(main_stack);
    if (fail_commit) return ESP_FAIL;
    if (pending_erase) blobs.erase(pending_key); else blobs[pending_key] = pending_blob;
    return ESP_OK;
}
int nvs_erase_key(nvs_handle_t, const char* key) {
    assert(main_stack && !strcmp(key, "gateway_config"));
    if (fail_erase) return ESP_FAIL;
    if (!blobs.count(key)) return ESP_ERR_NVS_NOT_FOUND;
    pending_key = key; pending_erase = true; return ESP_OK;
}
struct Application {
    std::vector<std::function<void()>> tasks;
    static Application& GetInstance() { static Application app; return app; }
    void Schedule(std::function<void()> task) { tasks.push_back(std::move(task)); }
    void Drain() { main_stack = true; auto pending = std::move(tasks); tasks.clear(); for (auto& f : pending) f(); }
};
struct WebSocket {
    bool fail = false;
    std::vector<std::string> messages;
    bool Send(const char* text) { if (fail) return false; messages.emplace_back(text); return true; }
};
int calibrate_mode = 0, Action_ID = 0;
struct Display {
    std::string text;
    void SetPetIdentity(const std::string& value, bool) { text = value; }
};
#define ESP_LOGW(...) ((void)0)
bool SendControlReceipt(WebSocket& socket, const char* id, bool ok, const char*) {
    auto* root = cJSON_CreateObject(); cJSON_AddStringToObject(root, "id", id); cJSON_AddBoolToObject(root, "ok", ok);
    char* text = cJSON_PrintUnformatted(root); bool sent = socket.Send(text); cJSON_free(text); cJSON_Delete(root); return sent;
}
'''

REGRESSIONS = r'''
const std::string TOKEN(32, 'T');
std::string document(int version, const std::string& url, const std::string& token = TOKEN, bool envelope = true) {
    auto* root = cJSON_CreateObject();
    if (envelope) { cJSON_AddStringToObject(root, "id", "config-test"); cJSON_AddStringToObject(root, "type", "set_gateway_config"); }
    cJSON_AddNumberToObject(root, "version", version); cJSON_AddStringToObject(root, "url", url.c_str());
    cJSON_AddStringToObject(root, "token", token.c_str());
    char* text = cJSON_PrintUnformatted(root); std::string result(text); cJSON_free(text); cJSON_Delete(root); return result;
}
bool parse(const std::string& text, puppy::GatewayConfig& config, bool envelope = true) {
    const char* error = nullptr; return puppy::ParseGatewayConfig(text.data(), text.size(), envelope, config, error);
}
int main() {
    const std::string url = "ws://192.0.2.78:8000/v1/audio";
    const char* error = nullptr;
    puppy::GatewayConfig config;
    assert(parse(document(1, url), config));
    assert(config.version == 1 && config.url == url && config.token == TOKEN && config.overridden);
    for (const char* valid : {"wss://puppy.example:443/v1/audio", "ws://puppy-gateway.local:8000/v1/audio", "ws://localhost:1/v1/audio"})
        assert(puppy::ValidGatewayFields(valid, TOKEN));
    for (const char* invalid : {"http://a:1/v1/audio", "WS://a:1/v1/audio", "ws://a/v1/audio", "ws://a:0/v1/audio",
         "ws://a:65536/v1/audio", "ws://a:-1/v1/audio", "ws://a:80x/v1/audio", "ws://a:80/v1/audio?token=x",
         "ws://a:80/v1/audio#x", "ws://u@a:80/v1/audio", "ws://u:p@a:80/v1/audio", "ws://a:80/v1/audio/",
         "ws://a:80/v1/control", "ws://[::1]:80/v1/audio", "ws://a..b:80/v1/audio", "ws://-a:80/v1/audio",
         "ws://a-:80/v1/audio", "ws://a.:80/v1/audio", "ws://a_b:80/v1/audio", "ws://127.0.0.999:80/v1/audio",
         "ws://127.0.0.01:80/v1/audio", "ws://127.1:80/v1/audio", "ws://a%40b:80/v1/audio", "ws://a:80/v1/../audio"})
        assert(!puppy::ValidGatewayFields(invalid, TOKEN));
    assert(!puppy::ValidGatewayFields("ws://" + std::string(64, 'a') + ":80/v1/audio", TOKEN));
    for (auto token : {std::string(23, 'x'), std::string(257, 'x'), TOKEN + " ", TOKEN + "\n", TOKEN + "\177", TOKEN + "名字"})
        assert(!puppy::ValidGatewayFields(url, token));
    assert(puppy::ValidGatewayFields(url, std::string(24, '!')));
    assert(puppy::ValidGatewayFields(url, std::string(256, '~')));
    puts("PASS: strict transport URL, host/port, path and token boundaries");

    for (const std::string& bad : {"{}", "null", "[]", "true", "{} {}"}) assert(!parse(bad, config));
    auto valid = document(1, url);
    for (const char* field : {",\"version\":2", ",\"url\":\"ws://a:1/v1/audio\"", ",\"token\":\"x\"", ",\"type\":\"set_gateway_config\"", ",\"extra\":true"}) {
        auto bad = valid; bad.insert(bad.size() - 1, field); assert(!parse(bad, config));
    }
    for (int version : {0, -1}) assert(!parse(document(version, url), config));
    // Reject cJSON's permissive number forms and fractional values that strtod
    // rounds to an integer; validation must inspect the original JSON bytes.
    for (const char* number : {"1.5", "true", "\"1\"", "2147483648", "1e999",
                              "1.", "01", "1.0000000000000001"}) {
        auto bad = valid; auto pos = bad.find("\"version\":1"); bad.replace(pos, 11, std::string("\"version\":") + number);
        assert(!parse(bad, config));
    }
    auto nul = valid; nul.insert(nul.find("config-test") + 6, "\\u0000"); assert(!parse(nul, config));
    nul = valid; nul.insert(nul.find("config-test"), 1, '\0'); assert(!parse(nul, config));
    assert(!parse(valid + std::string(4096, ' '), config));
    assert(parse(" \n" + valid + "\t\r\n", config));
    assert(parse(document(2147483647, url, TOKEN, false), config, false));
    assert(!parse(valid, config, false));
    const std::string deep = std::string(1000, '[') + "0" + std::string(1000, ']');
    assert(!puppy::BoundedControlJson(deep.data(), deep.size()));
    assert(!parse(deep, config));
    assert(puppy::BoundedControlJson(valid.data(), valid.size()));
    puts("PASS: strict JSON envelope, duplicate/unknown fields, versions, NUL and wire limits");

    auto fallback = puppy::LoadGatewayConfig(error);
    assert(!fallback.overridden && !fallback.version && fallback.url == "ws://fallback.local:8000/v1/audio");
    legacy["gateway_url"] = url; legacy["gateway_token"] = TOKEN;
    assert(puppy::LoadGatewayConfig(error).url == url);
    legacy["gateway_token"] = "short";
    assert(puppy::LoadGatewayConfig(error).url == fallback.url);
    legacy["gateway_token"] = TOKEN;
    blobs["gateway_config"] = "broken";
    assert(puppy::LoadGatewayConfig(error).url == url && error);
    parse(document(1, url), config);
    assert(!puppy::SaveGatewayConfig(config, error) && !writes);
    blobs.clear();
    fail_read = true; assert(puppy::LoadGatewayConfig(error).url == fallback.url); fail_read = false;
    fail_open = true; assert(puppy::LoadGatewayConfig(error).url == fallback.url); fail_open = false;
    assert(!handles);
    puts("PASS: bounded legacy/build fallback, corrupt blob refusal and all NVS handles closed");

    assert(puppy::SaveGatewayConfig(config, error));
    const auto old_blob = blobs.at("gateway_config");
    auto installed = puppy::LoadGatewayConfig(error);
    assert(installed.overridden && installed.version == 1 && installed.url == url && installed.token == TOKEN);
    assert(blobs.size() == 1 && legacy.at("gateway_token") == TOKEN);
    const auto old_writes = writes;
    assert(puppy::SaveGatewayConfig(config, error) && writes == old_writes);
    config.token = std::string(32, 'U');
    assert(!puppy::SaveGatewayConfig(config, error) && writes == old_writes);
    config.version = 2;
    assert(puppy::SaveGatewayConfig(config, error));
    config.version = 1;
    assert(!puppy::SaveGatewayConfig(config, error));
    config.version = 3;
    const auto retained = blobs.at("gateway_config");
    available = 0; assert(!puppy::SaveGatewayConfig(config, error)); available = 1000;
    fail_stats = true; assert(!puppy::SaveGatewayConfig(config, error)); fail_stats = false;
    fail_set = true; assert(!puppy::SaveGatewayConfig(config, error)); fail_set = false;
    fail_commit = true; assert(!puppy::SaveGatewayConfig(config, error)); fail_commit = false;
    assert(blobs.at("gateway_config") == retained && !handles);
    puts("PASS: single-blob persistence, version replay/conflict, capacity/write/commit failures without abort");

    Pet pet; WebSocket socket;
    const auto before = writes;
    main_stack = false;
    pet.Handle(socket, document(3, url));
    assert(writes == before && Application::GetInstance().tasks.size() == 1 && socket.messages.empty());
    pet.FlushGatewayReceipt(socket); assert(socket.messages.empty());
    pet.Handle(socket, document(4, url));
    assert(Application::GetInstance().tasks.size() == 1 && socket.messages.size() == 1);
    socket.messages.clear();
    Application::GetInstance().Drain();
    assert(writes == before + 1 && !handles);
    socket.fail = true; pet.FlushGatewayReceipt(socket); assert(pet.gateway_save_job_);
    socket.fail = false; pet.FlushGatewayReceipt(socket); assert(!pet.gateway_save_job_);
    assert(socket.messages.size() == 1 && socket.messages[0].find(TOKEN) == std::string::npos);
    auto* receipt = cJSON_Parse(socket.messages[0].c_str());
    assert(cJSON_IsTrue(cJSON_GetObjectItem(receipt, "ok")));
    assert(cJSON_IsFalse(cJSON_GetObjectItem(receipt, "executed")));
    assert(cJSON_IsTrue(cJSON_GetObjectItem(receipt, "restart_required")));
    assert(!strcmp(cJSON_GetObjectItem(receipt, "phase")->valuestring, "saved"));
    assert(cJSON_GetObjectItem(receipt, "version")->valueint == 3);
    cJSON_Delete(receipt);
    assert(pet.boot_gateway_.url.empty() && pet.boot_gateway_.token.empty());
    puts("PASS: RX queues one main-stack write, bounded outbox, saved receipt without token or live activation");

    blobs["offline_actions"] = "keep-action-library";
    legacy["pet_name"] = "keep-name";
    legacy["wifi"] = "keep-network";
    main_stack = false;
    Action_ID = 11; pet.RequestGatewayRecovery(); assert(Application::GetInstance().tasks.empty()); Action_ID = 0;
    pet.offline_sequence_.active = true; pet.RequestGatewayRecovery(); assert(Application::GetInstance().tasks.empty()); pet.offline_sequence_.active = false;
    calibrate_mode = 1; pet.RequestGatewayRecovery(); assert(Application::GetInstance().tasks.empty()); calibrate_mode = 0;
    pet.Handle(socket, document(4, url));
    pet.RequestGatewayRecovery(); pet.RequestGatewayRecovery();
    assert(Application::GetInstance().tasks.size() == 2 && pet.gateway_recovery_pending_);
    pet.Handle(socket, document(5, url)); assert(Application::GetInstance().tasks.size() == 2);
    Application::GetInstance().Drain();
    assert(!pet.gateway_recovery_pending_ && !pet.gateway_save_job_ && !blobs.count("gateway_config"));
    assert(blobs.at("offline_actions") == "keep-action-library" && legacy.at("pet_name") == "keep-name" && legacy.at("wifi") == "keep-network");
    assert(puppy::LoadGatewayConfig(error).url == url && !puppy::LoadGatewayConfig(error).overridden);
    assert(pet.display_->text.find("请重启") != std::string::npos && Action_ID == 0);
    assert(puppy::ClearGatewayOverride(error));
    blobs["gateway_config"] = "broken";
    assert(puppy::ClearGatewayOverride(error) && !blobs.count("gateway_config"));
    blobs["gateway_config"] = "broken"; fail_erase = true;
    assert(!puppy::ClearGatewayOverride(error)); fail_erase = false;
    fail_commit = true; assert(!puppy::ClearGatewayOverride(error)); fail_commit = false;
    assert(blobs.at("gateway_config") == "broken" && !handles);
    puts("PASS: physical idle recovery follows earlier save, erases only override, keeps legacy/name/library and never activates");
}
'''


def harness() -> str:
    board = (REPO_ROOT / "main/boards/puppy/puppy_board.cc").read_text()
    fields = board[board.index("    struct GatewaySaveJob {"):board.index("    Button boot_button_;")]
    branch = extract_function(board, 'if (strcmp(type->valuestring, "set_gateway_config") == 0)')
    flush = extract_function(board, "    void FlushGatewayReceipt(")
    recovery = extract_function(board, "    void RequestGatewayRecovery(")
    return STUBS + "\nclass Pet { public:\n" + fields + r'''
    std::mutex motion_mutex_;
    struct { bool active = false; } offline_sequence_;
    Display display; Display* display_ = &display;
    std::string pet_name_ = "Puppy";
    std::atomic<bool> companion_online_{false};
    void Handle(WebSocket& socket, const std::string& payload) {
        const char* data = payload.data(); size_t len = payload.size();
        auto* message = cJSON_ParseWithLength(data, len);
        auto* id = cJSON_GetObjectItem(message, "id");
        auto* type = cJSON_GetObjectItem(message, "type");
        const char* error = nullptr;
''' + branch + "\n cJSON_Delete(message); }\n" + flush + recovery + "\n};\n" + REGRESSIONS


def main() -> None:
    cjson = cjson_directory()
    flags = ["-fsanitize=address,undefined", "-g", "-UNDEBUG"]
    with tempfile.TemporaryDirectory(prefix="puppy-gateway-host-") as directory:
        folder = Path(directory)
        (folder / "nvs.h").write_text(NVS_HEADER)
        (folder / "sdkconfig.h").write_text(
            '#define CONFIG_PUPPY_LOCAL_GATEWAY_URL "ws://fallback.local:8000/v1/audio"\n'
            '#define CONFIG_PUPPY_LOCAL_GATEWAY_TOKEN "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"\n')
        (folder / "gateway-test.cc").write_text(harness())
        subprocess.run(shlex.split(os.environ.get("CC", "clang")) + flags + ["-c", str(cjson / "cJSON.c"),
                       "-o", str(folder / "cJSON.o")], check=True)
        subprocess.run(shlex.split(os.environ.get("CXX", "clang++")) + ["-std=c++17"] + flags + [
            "-I" + str(folder), "-I" + str(cjson), "-I" + str(REPO_ROOT / "main"),
            str(REPO_ROOT / "main/gateway_config.cc"), str(folder / "gateway-test.cc"),
            str(folder / "cJSON.o"), "-o", str(folder / "gateway-test")], check=True)
        subprocess.run([str(folder / "gateway-test")], check=True, timeout=30)


if __name__ == "__main__":
    main()
