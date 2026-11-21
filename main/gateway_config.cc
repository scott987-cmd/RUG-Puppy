#include "gateway_config.h"

#include <cJSON.h>
#include <nvs.h>
#include <sdkconfig.h>
#include <cmath>
#include <cstring>

#ifndef CONFIG_PUPPY_LOCAL_GATEWAY_URL
#define CONFIG_PUPPY_LOCAL_GATEWAY_URL ""
#endif
#ifndef CONFIG_PUPPY_LOCAL_GATEWAY_TOKEN
#define CONFIG_PUPPY_LOCAL_GATEWAY_TOKEN ""
#endif

namespace puppy {
namespace {
constexpr const char* kNamespace = "companion";
constexpr const char* kKey = "gateway_config";
enum class Stored { Missing, Valid, Invalid };

bool VisibleAscii(const std::string& value, size_t minimum, size_t maximum) {
    if (value.size() < minimum || value.size() > maximum) return false;
    for (unsigned char ch : value) if (ch < 0x21 || ch > 0x7e) return false;
    return true;
}

bool ValidHost(const std::string& host) {
    if (host.empty() || host.size() > 253) return false;
    bool numeric = true;
    for (char ch : host) if ((ch < '0' || ch > '9') && ch != '.') numeric = false;
    unsigned labels = 0;
    size_t start = 0;
    while (start < host.size()) {
        size_t end = host.find('.', start);
        if (end == std::string::npos) end = host.size();
        size_t length = end - start;
        if (!length || length > 63 || host[start] == '-' || host[end - 1] == '-') return false;
        unsigned octet = 0;
        for (size_t i = start; i < end; ++i) {
            char ch = host[i];
            if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                  (ch >= '0' && ch <= '9') || ch == '-')) return false;
            if (numeric) {
                octet = octet * 10 + (ch - '0');
                if (octet > 255) return false;
            }
        }
        if (numeric && (length > 3 || (length > 1 && host[start] == '0'))) return false;
        ++labels;
        if (end == host.size()) break;
        start = end + 1;
        if (start == host.size()) return false;
    }
    return !numeric || labels == 4;
}

bool HasNul(const char* data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (!data[i]) return true;
        if (data[i] == '\\' && ++i < length && data[i] == 'u' && i + 4 < length &&
            memcmp(data + i + 1, "0000", 4) == 0) return true;
    }
    return false;
}

bool Fields(const cJSON* root, bool envelope) {
    if (!cJSON_IsObject(root)) return false;
    const char* names[] = {"version", "url", "token", "id", "type"};
    unsigned seen = 0;
    for (auto* item = root->child; item; item = item->next) {
        unsigned i = 0, count = envelope ? 5 : 3;
        while (i < count && strcmp(item->string, names[i])) ++i;
        if (i == count || (seen & (1U << i))) return false;
        seen |= 1U << i;
    }
    return seen == (envelope ? 31U : 7U);
}

bool StrictVersionLexeme(const char* data, size_t length, uint32_t& version) {
    // This flat schema has exactly one non-string value: version. cJSON's
    // strtod parser accepts 01/1. and rounds fractional values near integers,
    // so validate the original numeric bytes before converting to double.
    bool quoted = false, found = false;
    for (size_t i = 0; i < length; ++i) {
        const char ch = data[i];
        if (quoted && ch == '\\') { ++i; continue; }
        if (ch == '"') { quoted = !quoted; continue; }
        if (quoted || ch == '{' || ch == '}' || ch == ':' || ch == ',' ||
            ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
        if (found || ch < '1' || ch > '9') return false;
        uint32_t value = 0;
        while (i < length && data[i] >= '0' && data[i] <= '9') {
            const uint32_t digit = data[i] - '0';
            if (value > (2147483647U - digit) / 10) return false;
            value = value * 10 + digit;
            ++i;
        }
        --i;
        version = value;
        found = true;
    }
    return found && !quoted;
}

Stored ReadStored(nvs_handle_t handle, GatewayConfig& config) {
    size_t length = 0;
    auto result = nvs_get_blob(handle, kKey, nullptr, &length);
    if (result == ESP_ERR_NVS_NOT_FOUND) return Stored::Missing;
    if (result != ESP_OK || !length || length > kGatewayBlobLimit) return Stored::Invalid;
    std::string data(length, '\0');
    if (nvs_get_blob(handle, kKey, data.data(), &length) != ESP_OK || length != data.size()) return Stored::Invalid;
    const char* error = nullptr;
    if (!ParseGatewayConfig(data.data(), data.size(), false, config, error)) return Stored::Invalid;
    config.overridden = true;
    return Stored::Valid;
}

bool LegacyString(nvs_handle_t handle, const char* key, size_t maximum,
                  const std::string& fallback, std::string& value) {
    size_t length = 0;
    auto result = nvs_get_str(handle, key, nullptr, &length);
    if (result == ESP_ERR_NVS_NOT_FOUND) { value = fallback; return true; }
    if (result != ESP_OK || !length || length > maximum + 1) return false;
    std::string data(length, '\0');
    if (nvs_get_str(handle, key, data.data(), &length) != ESP_OK || length != data.size() ||
        data.back() != '\0' || strlen(data.c_str()) != length - 1) return false;
    data.pop_back(); value = std::move(data); return true;
}

std::string Encode(const GatewayConfig& config) {
    auto* root = cJSON_CreateObject();
    if (!root) return {};
    bool ok = cJSON_AddNumberToObject(root, "version", config.version) &&
              cJSON_AddStringToObject(root, "url", config.url.c_str()) &&
              cJSON_AddStringToObject(root, "token", config.token.c_str());
    char* text = ok ? cJSON_PrintUnformatted(root) : nullptr;
    std::string encoded = text ? text : "";
    cJSON_free(text); cJSON_Delete(root);
    return encoded;
}
}

bool ValidGatewayFields(const std::string& url, const std::string& token) {
    if (!VisibleAscii(token, 24, 256) || !VisibleAscii(url, 1, 320)) return false;
    size_t begin = url.compare(0, 5, "ws://") == 0 ? 5 : url.compare(0, 6, "wss://") == 0 ? 6 : 0;
    if (!begin) return false;
    const size_t colon = url.find(':', begin), slash = url.find('/', begin);
    if (colon == std::string::npos || slash == std::string::npos || colon >= slash ||
        url.substr(slash) != "/v1/audio" || !ValidHost(url.substr(begin, colon - begin))) return false;
    // The current transport parser supports DNS/IPv4, not bracketed IPv6.
    const size_t digits = slash - colon - 1;
    if (!digits || digits > 5) return false;
    unsigned port = 0;
    for (size_t i = colon + 1; i < slash; ++i) {
        if (url[i] < '0' || url[i] > '9') return false;
        port = port * 10 + (url[i] - '0');
    }
    return port >= 1 && port <= 65535;
}

bool BoundedControlJson(const char* data, size_t length, unsigned max_depth) {
    if (!data || !length || HasNul(data, length)) return false;
    unsigned depth = 0;
    bool quoted = false;
    for (size_t i = 0; i < length; ++i) {
        const char ch = data[i];
        if (quoted && ch == '\\') { ++i; continue; }
        if (ch == '"') quoted = !quoted;
        if (quoted) continue;
        if (ch == '{' || ch == '[') { if (++depth > max_depth) return false; }
        if (ch == '}' || ch == ']') { if (!depth) return false; --depth; }
    }
    return !quoted && depth == 0;
}

bool ParseGatewayConfig(const char* data, size_t length, bool envelope,
                        GatewayConfig& config, const char*& error) {
    error = "invalid gateway configuration";
    if (!data || !length || length > (envelope ? kGatewayWireLimit : kGatewayBlobLimit) ||
        !BoundedControlJson(data, length, 1)) return false;
    uint32_t exact_version = 0;
    if (!StrictVersionLexeme(data, length, exact_version)) return false;
    const char* end = nullptr;
    auto* root = cJSON_ParseWithLengthOpts(data, length, &end, false);
    while (end && end < data + length && (*end == ' ' || *end == '\n' || *end == '\r' || *end == '\t')) ++end;
    bool ok = end == data + length && Fields(root, envelope);
    auto* version = cJSON_GetObjectItemCaseSensitive(root, "version");
    auto* url = cJSON_GetObjectItemCaseSensitive(root, "url");
    auto* token = cJSON_GetObjectItemCaseSensitive(root, "token");
    ok = ok && cJSON_IsNumber(version) && std::isfinite(version->valuedouble) &&
         version->valuedouble >= 1 && version->valuedouble <= 2147483647 &&
         version->valuedouble == std::floor(version->valuedouble) && version->valuedouble == exact_version &&
         cJSON_IsString(url) && cJSON_IsString(token);
    if (envelope && ok) {
        auto* id = cJSON_GetObjectItemCaseSensitive(root, "id");
        auto* type = cJSON_GetObjectItemCaseSensitive(root, "type");
        ok = cJSON_IsString(id) && VisibleAscii(id->valuestring, 1, 64) &&
             cJSON_IsString(type) && !strcmp(type->valuestring, "set_gateway_config");
    }
    if (ok) ok = ValidGatewayFields(url->valuestring, token->valuestring);
    if (ok) config = {static_cast<uint32_t>(version->valuedouble), url->valuestring, token->valuestring, true};
    cJSON_Delete(root);
    return ok;
}

GatewayConfig LoadGatewayConfig(const char*& error) {
    error = nullptr;
    GatewayConfig fallback{0, CONFIG_PUPPY_LOCAL_GATEWAY_URL, CONFIG_PUPPY_LOCAL_GATEWAY_TOKEN, false};
    nvs_handle_t handle = 0;
    auto result = nvs_open(kNamespace, NVS_READONLY, &handle);
    if (result == ESP_OK) {
        GatewayConfig stored;
        const auto state = ReadStored(handle, stored);
        if (state == Stored::Valid) { nvs_close(handle); return stored; }
        if (state == Stored::Invalid) error = "stored gateway configuration invalid; using fallback";
        GatewayConfig legacy;
        const bool legacy_ok = LegacyString(handle, "gateway_url", 320, fallback.url, legacy.url) &&
                               LegacyString(handle, "gateway_token", 256, fallback.token, legacy.token);
        nvs_close(handle);
        if (legacy_ok && ValidGatewayFields(legacy.url, legacy.token)) return legacy;
    } else if (result != ESP_ERR_NVS_NOT_FOUND) error = "gateway storage unavailable; using fallback";
    if (ValidGatewayFields(fallback.url, fallback.token)) return fallback;
    error = "no valid local gateway configuration; local controls remain available";
    return {};
}

bool SaveGatewayConfig(const GatewayConfig& config, const char*& error) {
    error = nullptr;
    if (!config.version || config.version > 2147483647 || !ValidGatewayFields(config.url, config.token)) {
        error = "invalid gateway configuration"; return false;
    }
    nvs_handle_t handle = 0;
    auto result = nvs_open(kNamespace, NVS_READONLY, &handle);
    GatewayConfig previous;
    if (result == ESP_OK) {
        auto state = ReadStored(handle, previous);
        nvs_close(handle);
        if (state == Stored::Invalid) { error = "stored gateway configuration unreadable; fallback retained"; return false; }
    } else if (result != ESP_ERR_NVS_NOT_FOUND) { error = "gateway storage unavailable"; return false; }
    if (config.version < previous.version) { error = "gateway configuration version is older"; return false; }
    if (config.version == previous.version) {
        if (config.url == previous.url && config.token == previous.token) return true;
        error = "gateway configuration version has different content"; return false;
    }
    std::string encoded = Encode(config);
    if (encoded.empty() || encoded.size() > kGatewayBlobLimit) { error = "gateway encoding unavailable"; return false; }
    nvs_stats_t stats{};
    const size_t needed = (encoded.size() + 31) / 32 + 8 + 16;
    if (nvs_get_stats(nullptr, &stats) != ESP_OK || stats.available_entries < needed) {
        error = "insufficient NVS capacity for gateway configuration"; return false;
    }
    if (nvs_open(kNamespace, NVS_READWRITE, &handle) != ESP_OK) { error = "gateway storage unavailable"; return false; }
    result = nvs_set_blob(handle, kKey, encoded.data(), encoded.size());
    if (result == ESP_OK) result = nvs_commit(handle);
    nvs_close(handle);
    if (result != ESP_OK) { error = "gateway save failed; persistence unconfirmed"; return false; }
    return true;
}

bool ClearGatewayOverride(const char*& error) {
    error = nullptr;
    nvs_handle_t handle = 0;
    auto result = nvs_open(kNamespace, NVS_READWRITE, &handle);
    if (result != ESP_OK) { error = "gateway recovery storage unavailable"; return false; }
    result = nvs_erase_key(handle, kKey);
    if (result == ESP_ERR_NVS_NOT_FOUND) result = ESP_OK;
    else if (result == ESP_OK) result = nvs_commit(handle);
    nvs_close(handle);
    if (result != ESP_OK) { error = "gateway recovery failed; persistence unconfirmed"; return false; }
    return true;
}
}
