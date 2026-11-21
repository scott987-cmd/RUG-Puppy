#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace puppy {
constexpr size_t kGatewayWireLimit = 4096;
constexpr size_t kGatewayBlobLimit = 2048;
struct GatewayConfig {
    uint32_t version = 0;  // Zero means legacy keys / build-time fallback.
    std::string url;
    std::string token;
    bool overridden = false;
};

bool ValidGatewayFields(const std::string& url, const std::string& token);
bool BoundedControlJson(const char* data, size_t length, unsigned max_depth = 8);
bool ParseGatewayConfig(const char* data, size_t length, bool envelope,
                        GatewayConfig& config, const char*& error);
GatewayConfig LoadGatewayConfig(const char*& error);
bool SaveGatewayConfig(const GatewayConfig& config, const char*& error);
bool ClearGatewayOverride(const char*& error);
}
