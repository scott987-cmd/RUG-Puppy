#!/usr/bin/env python3
"""Exercise the production sleep decision at clock and activity boundaries."""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include <cassert>
#include <cstdint>
#include <ctime>
#include "main/boards/puppy/sleep_policy.h"

int main() {
    using namespace puppy;
    // 2026-10-11 13:00 UTC = 21:00 China time.
    constexpr time_t night = 1791723600;
    constexpr int64_t now = 3600LL * 1000000;
    static_assert(kIdleSleepUs == 1800000000LL);
    assert(IsNight(night));
    assert(!IsNight(night - 1));
    assert(IsNight(night + 9 * 3600 + 59 * 60));
    assert(!IsNight(night + 10 * 3600));
    assert(ShouldDeepSleep(night, now, now - 1, false, true));
    assert(!ShouldDeepSleep(night, now, now - 1, true, true));
    assert(ShouldDeepSleep(night, now, now - kIdleSleepUs, true, true));
    assert(!ShouldDeepSleep(night, now, now - kIdleSleepUs, false, false));
    assert(!ShouldDeepSleep(night - 3600, now, now - kIdleSleepUs + 1, false, true));
    assert(ShouldDeepSleep(night - 3600, now, now - kIdleSleepUs, false, true));
    assert(!ShouldDeepSleep(0, now, now - kIdleSleepUs, false, true));
    assert(MicrosecondsUntilNextMorning(night) == 10ULL * 3600 * 1000000);
    assert(MicrosecondsUntilNextMorning(night + 10 * 3600) == 24ULL * 3600 * 1000000);
}
'''


with tempfile.TemporaryDirectory(prefix="puppy-sleep-") as directory:
    source = Path(directory) / "sleep_policy_test.cc"
    binary = Path(directory) / "sleep_policy_test"
    source.write_text(SOURCE)
    subprocess.run(["clang++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print("Puppy sleep policy boundaries: PASS")
