#pragma once

#include <cstdint>
#include <ctime>

namespace puppy {

// The family schedule is in China Standard Time. Keep the epoch in UTC so
// deep-sleep timer wakeup does not depend on the C library's process timezone.
constexpr int64_t kChinaUtcOffsetSeconds = 8 * 60 * 60;
constexpr int64_t kIdleSleepUs = 30LL * 60 * 1000000;
constexpr int64_t kEarliestTrustedEpoch = 1735689600; // 2025-01-01 UTC
constexpr int64_t kLatestTrustedEpoch = 4102444800;   // 2100-01-01 UTC

inline bool HasTrustedClock(time_t epoch) {
    return int64_t(epoch) >= kEarliestTrustedEpoch && int64_t(epoch) < kLatestTrustedEpoch;
}

inline int SecondsOfChinaDay(time_t epoch) {
    return int((int64_t(epoch) + kChinaUtcOffsetSeconds) % 86400);
}

inline bool IsNight(time_t epoch) {
    const int seconds = SecondsOfChinaDay(epoch);
    return seconds >= 21 * 3600 || seconds < 7 * 3600;
}

inline uint64_t MicrosecondsUntilNextMorning(time_t epoch) {
    const int seconds = SecondsOfChinaDay(epoch);
    int until = (7 * 3600 - seconds + 86400) % 86400;
    if (until == 0) until = 86400;
    return uint64_t(until) * 1000000;
}

inline bool ShouldDeepSleep(time_t epoch, int64_t now_us, int64_t last_activity_us,
                            bool has_interaction_window, bool idle) {
    if (!idle || !HasTrustedClock(epoch) || last_activity_us <= 0 || now_us < last_activity_us) return false;
    const bool inactive = now_us - last_activity_us >= kIdleSleepUs;
    // At 21:00 an idle pet sleeps promptly. A button press or conversation
    // grants a 30-minute window, including after a normal reboot at night.
    return (IsNight(epoch) && !has_interaction_window) || inactive;
}

} // namespace puppy
