#pragma once

#include <cstdint>
#include <string>
#include <cJSON.h>

namespace puppy {
constexpr size_t kOfflineActionLimit = 3;
constexpr size_t kOfflineStepLimit = 6;
constexpr size_t kOfflineEncodedLimit = 4096;
struct OfflineStep { uint8_t action = 0; uint16_t wait_ms = 0; };
struct OfflineAction {
    char name[65]{};
    uint8_t count = 0;
    OfflineStep steps[kOfflineStepLimit]{};
};
struct OfflineLibrary {
    uint32_t version = 0;
    uint8_t count = 0;
    OfflineAction actions[kOfflineActionLimit]{};
};

bool ParseOfflineLibrary(const cJSON* document, bool envelope, OfflineLibrary& library,
                         std::string& encoded, const char*& error);
bool SameOfflineLibrary(const OfflineLibrary& left, const OfflineLibrary& right);
bool HasJsonNulEscape(const char* data, size_t length);
// These functions must run on an internal-stack task, never the TCP callback.
bool LoadOfflineLibrary(OfflineLibrary& library, const char*& error);
bool SaveOfflineLibrary(const OfflineLibrary& library, const std::string& encoded, const char*& error);
}
