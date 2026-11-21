#include "offline_actions.h"
#include "xgo_action.h"
#include <cmath>
#include <cstring>
#include <nvs.h>

namespace puppy {
namespace {
bool Fields(const cJSON* object, const char* const* allowed, size_t count) {
    if (!cJSON_IsObject(object)) return false;
    unsigned seen = 0;
    const cJSON* field = nullptr;
    cJSON_ArrayForEach(field, object) {
        size_t index = 0;
        for (; index < count; ++index) if (field->string && strcmp(field->string, allowed[index]) == 0) break;
        if (index == count || (seen & (1u << index))) return false;
        seen |= 1u << index;
    }
    return true;
}

bool Utf8Name(const char* value) {
    if (!value) return false;
    size_t size = strlen(value);
    if (!size || size > 64) return false;
    for (size_t i = 0; i < size;) {
        auto first = static_cast<unsigned char>(value[i++]);
        if (first < 0x20 || first == 0x7f) return false;
        if (first < 0x80) continue;
        unsigned count = first >= 0xc2 && first <= 0xdf ? 1 : first >= 0xe0 && first <= 0xef ? 2 : first >= 0xf0 && first <= 0xf4 ? 3 : 0;
        if (!count || i + count > size) return false;
        unsigned code = first & (0x7f >> (count + 1));
        for (unsigned j = 0; j < count; ++j) {
            auto next = static_cast<unsigned char>(value[i++]);
            if ((next & 0xc0) != 0x80) return false;
            code = (code << 6) | (next & 0x3f);
        }
        if ((count == 1 && code < 0x80) || (count == 2 && code < 0x800) ||
            (count == 3 && code < 0x10000) || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
    }
    return true;
}

const char* ActionName(uint8_t id) {
    if (id == Wave_ID) return "wave";
    if (id == Sit_ID) return "sit";
    if (id == Stretch_ID) return "stretch";
    if (id == Hug_ID) return "hug";
    return nullptr;
}

bool ReadLibrary(nvs_handle_t handle, OfflineLibrary& library, const char*& error) {
    size_t length = 0;
    esp_err_t result = nvs_get_blob(handle, "offline_actions", nullptr, &length);
    if (result == ESP_ERR_NVS_NOT_FOUND) { library = {}; return true; }
    if (result != ESP_OK || !length || length > kOfflineEncodedLimit) { error = "offline library read failed"; return false; }
    std::string data(length, '\0');
    if (nvs_get_blob(handle, "offline_actions", data.data(), &length) != ESP_OK || HasJsonNulEscape(data.data(), length)) {
        error = "offline library read failed"; return false;
    }
    const char* end = nullptr;
    cJSON* root = cJSON_ParseWithLengthOpts(data.data(), length, &end, false);
    std::string canonical;
    bool ok = root && end == data.data() + length && ParseOfflineLibrary(root, false, library, canonical, error);
    cJSON_Delete(root);
    if (!ok) error = "stored offline library is invalid";
    return ok;
}
}

bool HasJsonNulEscape(const char* data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (data[i] == '\0') return true;
        if (data[i] == '\\' && ++i < length && data[i] == 'u' && i + 4 < length &&
            memcmp(data + i + 1, "0000", 4) == 0) return true;
    }
    return false;
}

bool ParseOfflineLibrary(const cJSON* document, bool envelope, OfflineLibrary& library,
                         std::string& encoded, const char*& error) {
    error = "invalid offline action library";
    static const char* root_fields[] = {"version", "actions", "id", "type"};
    static const char* action_fields[] = {"name", "steps"};
    static const char* step_fields[] = {"action", "wait_ms"};
    if (!Fields(document, root_fields, envelope ? 4 : 2)) return false;
    if (envelope) {
        auto* id = cJSON_GetObjectItemCaseSensitive(document, "id");
        auto* type = cJSON_GetObjectItemCaseSensitive(document, "type");
        if (!cJSON_IsString(id) || !id->valuestring[0] || strlen(id->valuestring) > 64 ||
            !cJSON_IsString(type) || strcmp(type->valuestring, "set_offline_actions")) return false;
    }
    auto* version = cJSON_GetObjectItemCaseSensitive(document, "version");
    auto* actions = cJSON_GetObjectItemCaseSensitive(document, "actions");
    if (!cJSON_IsNumber(version) || !std::isfinite(version->valuedouble) || version->valuedouble < 1 ||
        version->valuedouble > 2147483647 || version->valuedouble != std::floor(version->valuedouble) ||
        !cJSON_IsArray(actions) || cJSON_GetArraySize(actions) > static_cast<int>(kOfflineActionLimit)) return false;
    OfflineLibrary parsed{};
    parsed.version = static_cast<uint32_t>(version->valuedouble);
    parsed.count = cJSON_GetArraySize(actions);
    for (size_t i = 0; i < parsed.count; ++i) {
        auto* action = cJSON_GetArrayItem(actions, i);
        auto* name = cJSON_GetObjectItemCaseSensitive(action, "name");
        auto* steps = cJSON_GetObjectItemCaseSensitive(action, "steps");
        if (!Fields(action, action_fields, 2) || !cJSON_IsString(name) || !Utf8Name(name->valuestring) ||
            !cJSON_IsArray(steps) || cJSON_GetArraySize(steps) < 1 ||
            cJSON_GetArraySize(steps) > static_cast<int>(kOfflineStepLimit)) return false;
        auto& target = parsed.actions[i];
        strcpy(target.name, name->valuestring);
        for (size_t j = 0; j < i; ++j) if (!strcmp(parsed.actions[j].name, target.name)) return false;
        target.count = cJSON_GetArraySize(steps);
        unsigned total_wait = 0;
        for (size_t j = 0; j < target.count; ++j) {
            auto* step = cJSON_GetArrayItem(steps, j);
            if (!Fields(step, step_fields, 2)) return false;
            auto* action_name = cJSON_GetObjectItemCaseSensitive(step, "action");
            auto* wait = cJSON_GetObjectItemCaseSensitive(step, "wait_ms");
            auto& value = target.steps[j];
            if (action_name) {
                if (!cJSON_IsString(action_name)) return false;
                for (auto id : {Wave_ID, Sit_ID, Stretch_ID, Hug_ID})
                    if (!strcmp(ActionName(id), action_name->valuestring)) value.action = id;
                if (!value.action) return false;
            }
            if (wait) {
                if (!cJSON_IsNumber(wait) || !std::isfinite(wait->valuedouble) || wait->valuedouble < 0 ||
                    wait->valuedouble > 3000 || wait->valuedouble != std::floor(wait->valuedouble)) return false;
                value.wait_ms = wait->valueint;
            }
            if (!value.action && !value.wait_ms) return false;
            total_wait += value.wait_ms;
        }
        if (total_wait > 20000) return false;
    }
    // Canonicalize validated fields only. Request IDs and input key ordering do
    // not change the library's identity; absent wait_ms means zero.
    cJSON* canonical = cJSON_CreateObject();
    if (!canonical) { error = "offline library memory unavailable"; return false; }
    bool allocated = cJSON_AddNumberToObject(canonical, "version", parsed.version) != nullptr;
    cJSON* list = cJSON_AddArrayToObject(canonical, "actions");
    allocated = allocated && list;
    for (size_t i = 0; allocated && i < parsed.count; ++i) {
        cJSON* item = cJSON_CreateObject();
        if (!item) { allocated = false; break; }
        cJSON_AddItemToArray(list, item);
        allocated = cJSON_AddStringToObject(item, "name", parsed.actions[i].name) != nullptr;
        cJSON* steps = cJSON_AddArrayToObject(item, "steps");
        allocated = allocated && steps;
        for (size_t j = 0; allocated && j < parsed.actions[i].count; ++j) {
            cJSON* step = cJSON_CreateObject();
            if (!step) { allocated = false; break; }
            cJSON_AddItemToArray(steps, step);
            const auto& value = parsed.actions[i].steps[j];
            if (value.action) allocated = cJSON_AddStringToObject(step, "action", ActionName(value.action)) != nullptr;
            allocated = allocated && cJSON_AddNumberToObject(step, "wait_ms", value.wait_ms);
        }
    }
    char* text = allocated ? cJSON_PrintUnformatted(canonical) : nullptr;
    bool ok = text && strlen(text) <= kOfflineEncodedLimit;
    if (ok) { encoded = text; library = parsed; }
    else error = "offline library exceeds storage or memory limit";
    cJSON_free(text);
    cJSON_Delete(canonical);
    return ok;
}

bool SameOfflineLibrary(const OfflineLibrary& left, const OfflineLibrary& right) {
    if (left.version != right.version || left.count != right.count) return false;
    for (size_t i = 0; i < left.count; ++i) {
        const auto& a = left.actions[i]; const auto& b = right.actions[i];
        if (strcmp(a.name, b.name) || a.count != b.count) return false;
        for (size_t j = 0; j < a.count; ++j)
            if (a.steps[j].action != b.steps[j].action || a.steps[j].wait_ms != b.steps[j].wait_ms) return false;
    }
    return true;
}

bool LoadOfflineLibrary(OfflineLibrary& library, const char*& error) {
    nvs_handle_t handle = 0;
    auto result = nvs_open("companion", NVS_READONLY, &handle);
    if (result == ESP_ERR_NVS_NOT_FOUND) { library = {}; return true; }
    if (result != ESP_OK) { error = "offline library NVS unavailable"; return false; }
    bool ok = ReadLibrary(handle, library, error);
    nvs_close(handle);
    return ok;
}

bool SaveOfflineLibrary(const OfflineLibrary& library, const std::string& encoded, const char*& error) {
    OfflineLibrary previous{};
    if (!LoadOfflineLibrary(previous, error)) return false;
    if (library.version < previous.version) { error = "offline library version is older"; return false; }
    if (library.version == previous.version) {
        bool same = SameOfflineLibrary(library, previous);
        if (!same) error = "offline library version has different content";
        return same;
    }
    if (encoded.empty() || encoded.size() > kOfflineEncodedLimit) { error = "offline library too large"; return false; }
    nvs_stats_t stats{};
    // Require room for the new blob without credit for reclaiming the old one,
    // plus chunk/namespace overhead and 16 spare entries for other settings.
    const size_t entries = (encoded.size() + 31) / 32 + 8 + 16;
    if (nvs_get_stats(nullptr, &stats) != ESP_OK || stats.available_entries < entries) {
        error = "insufficient NVS capacity for offline library"; return false;
    }
    nvs_handle_t handle = 0;
    if (nvs_open("companion", NVS_READWRITE, &handle) != ESP_OK) { error = "offline library NVS unavailable"; return false; }
    auto result = nvs_set_blob(handle, "offline_actions", encoded.data(), encoded.size());
    if (result == ESP_OK) result = nvs_commit(handle);
    nvs_close(handle);
    if (result != ESP_OK) { error = "offline library save failed; persistence unconfirmed"; return false; }
    return true;
}
}
