// Real tests for phase 4: include/exec_core/core_affinity.hpp +
// src/core_affinity.cpp (cross-platform thread pinning) and
// include/exec_core/cache_line.hpp (kCacheLineSize).
//
// Plain assert()-based tests, consistent with tests/test_order_book.cpp /
// tests/test_ring_buffer.cpp / tests/test_slab_allocator.cpp.
//
// This project's local dev machine is macOS/Apple Silicon (M2), which per
// core_affinity.hpp's file header has NO real kernel-enforced pinning
// guarantee and NO way to read back "which core is this thread on right
// now" -- so the macOS-side assertions below only check what is honestly
// verifiable there: the call doesn't crash, and it reports the outcome the
// real OS call actually produced (verified empirically below, not assumed
// to match any prior report -- see the printed os_status this test emits).
//
// The #ifdef __linux__ branch below asserts the STRONGER, kernel-enforced
// guarantee (pin to core 0, then confirm sched_getcpu() == 0). It has not
// been run by this test invocation on this machine (there is no Linux here
// to run it on) -- it is written to be exercised for real by CI
// (ubuntu-latest; see .github/workflows/ci.yml), not asserted as already
// verified.

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <thread>

#include "exec_core/cache_line.hpp"
#include "exec_core/core_affinity.hpp"

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

using namespace exec_core;

namespace {

// ---------------------------------------------------------------------
// kCacheLineSize: trivial sanity check that the fallback/derived constant
// is a plausible cache-line size, not e.g. 0, negative-cast-to-huge, or
// something absurd like 3. 64 is the specific value documented in
// cache_line.hpp for this project's two real targets (Apple Silicon local,
// x86-64 ubuntu-latest CI); assert that value directly (not just
// "some power of 2") since this project only claims to support those two
// targets today.
// ---------------------------------------------------------------------
void test_cache_line_size_is_sane() {
    assert(kCacheLineSize > 0);
    // Power-of-two check: exactly one bit set.
    assert((kCacheLineSize & (kCacheLineSize - 1)) == 0);
    assert(kCacheLineSize == 64);
}

// ---------------------------------------------------------------------
// pin_current_thread_to_core() must never crash and must always report one
// of the three documented PinOutcome values, whatever the real OS call
// underneath it actually returned. This test does not hardcode an expected
// outcome -- it prints what actually happened (this is the only place this
// test suite is allowed to "report a benchmark-adjacent fact," and it must
// be the real one, not a copy-pasted claim from the header comment).
// ---------------------------------------------------------------------
void test_pin_does_not_crash_and_reports_honest_outcome() {
    const PinResult r = pin_current_thread_to_core(0);

    const bool is_known_outcome = (r.outcome == PinOutcome::Pinned) ||
                                   (r.outcome == PinOutcome::BestEffortHint) ||
                                   (r.outcome == PinOutcome::Unsupported);
    assert(is_known_outcome);

    const char* name = "?";
    switch (r.outcome) {
        case PinOutcome::Pinned:         name = "Pinned"; break;
        case PinOutcome::BestEffortHint: name = "BestEffortHint"; break;
        case PinOutcome::Unsupported:    name = "Unsupported"; break;
    }
    std::printf("  [observed] pin_current_thread_to_core(0) -> outcome=%s os_status=%d\n",
                name, r.os_status);

#if defined(__linux__)
    // Linux: this file's header documents pthread_setaffinity_np() success
    // as a real, kernel-enforced guarantee. Assert that guarantee for real,
    // via the strongest available cross-check: after a successful pin to
    // core 0, sched_getcpu() (a separate syscall, not just re-reading our
    // own PinResult) must independently agree the thread is on core 0.
    // (If core 0 happens to be offline/unavailable in a given CI container,
    // pthread_setaffinity_np itself would fail and report Unsupported --
    // this assert only fires on the success path.)
    if (r.outcome == PinOutcome::Pinned) {
        assert(r.os_status == 0);
        const std::optional<int> cpu = current_core_id();
        assert(cpu.has_value());
        assert(*cpu == 0);
    } else {
        // Documented fallback path if pinning to core 0 specifically failed
        // in this environment (e.g. restrictive container cpuset): still
        // must be a real, non-fabricated Unsupported with the errno set.
        assert(r.outcome == PinOutcome::Unsupported);
    }

    // Re-pin to whatever the current core is (or core 0 if unknown) and
    // confirm sched_getcpu() tracks a *different* explicit pin too, not
    // just core 0 by coincidence.
    {
        const long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        if (nproc > 1) {
            const unsigned other_core = 1;
            const PinResult r2 = pin_current_thread_to_core(other_core);
            if (r2.outcome == PinOutcome::Pinned) {
                const std::optional<int> cpu2 = current_core_id();
                assert(cpu2.has_value());
                assert(*cpu2 == static_cast<int>(other_core));
            }
        }
    }
#elif defined(__APPLE__)
    // macOS: per core_affinity.hpp's file header, thread_policy_set can
    // only ever honestly report BestEffortHint (kr == KERN_SUCCESS, but
    // documented-unhonored on Apple Silicon) or Unsupported (any other
    // kern_return_t) -- never Pinned, since this platform has no real
    // pinning API this file implements. Assert that Pinned is categorically
    // impossible here, and that os_status is a real kern_return_t (0 for
    // the BestEffortHint/KERN_SUCCESS case).
    assert(r.outcome != PinOutcome::Pinned);
    if (r.outcome == PinOutcome::BestEffortHint) {
        assert(r.os_status == 0); // KERN_SUCCESS
    }
    // Do NOT assert a specific kern_return_t value (e.g. KERN_NOT_SUPPORTED
    // == 46) as if it were a portable guarantee: this is what was actually
    // observed on this project's specific M2 dev machine (see this file's
    // printed output and the tester's report), not a documented cross-Mac
    // contract. A future macOS version or different Apple Silicon SKU could
    // legitimately return a different kern_return_t while still correctly
    // falling into the same honest BestEffortHint/Unsupported dichotomy.
#endif
}

// ---------------------------------------------------------------------
// current_core_id() must not crash, and on macOS must be exactly the
// documented std::nullopt (there is no "sometimes it works" middle ground
// claimed by this header -- see core_affinity.hpp).
// ---------------------------------------------------------------------
void test_current_core_id_matches_documented_platform_behavior() {
    const std::optional<int> cpu = current_core_id();
#if defined(__linux__)
    // Linux: sched_getcpu() should succeed on any normal running thread.
    assert(cpu.has_value());
    assert(*cpu >= 0);
#elif defined(__APPLE__)
    // macOS: documented to always be nullopt -- no fallback/private API.
    assert(!cpu.has_value());
#endif
}

// ---------------------------------------------------------------------
// pin_current_thread_to_core() must behave the same way (not crash, still
// return one of the three documented outcomes) when called from a thread
// other than main -- this matters because the real intended caller is a
// dedicated matching-engine worker thread, not main(), and a naive
// implementation could plausibly rely on some main-thread-only assumption
// (e.g. pthread_self() vs a cached thread handle) that this catches.
// ---------------------------------------------------------------------
void test_pin_from_worker_thread_does_not_crash() {
    PinResult worker_result{};
    bool worker_ran = false;

    std::thread worker([&]() {
        worker_result = pin_current_thread_to_core(0);
        worker_ran = true;
    });
    worker.join();

    assert(worker_ran);
    const bool is_known_outcome = (worker_result.outcome == PinOutcome::Pinned) ||
                                   (worker_result.outcome == PinOutcome::BestEffortHint) ||
                                   (worker_result.outcome == PinOutcome::Unsupported);
    assert(is_known_outcome);
#if defined(__APPLE__)
    assert(worker_result.outcome != PinOutcome::Pinned);
#endif
}

// ---------------------------------------------------------------------
// Calling pin_current_thread_to_core() with an out-of-range core_id (e.g.
// far beyond the machine's actual CPU count) must still return an honest
// result -- Unsupported (errno/kern_return_t reflecting the invalid
// request), never a crash and never a fabricated Pinned/BestEffortHint.
// ---------------------------------------------------------------------
void test_pin_to_absurd_core_id_is_reported_honestly_not_crashed() {
    const PinResult r = pin_current_thread_to_core(4096);
    std::printf("  [observed] pin_current_thread_to_core(4096) -> outcome=%d os_status=%d\n",
                static_cast<int>(r.outcome), r.os_status);
    assert(r.outcome == PinOutcome::Unsupported);
}

} // namespace

int main() {
    struct NamedTest {
        const char* name;
        void (*fn)();
    };
    const NamedTest tests[] = {
        {"cache_line_size_is_sane", test_cache_line_size_is_sane},
        {"pin_does_not_crash_and_reports_honest_outcome", test_pin_does_not_crash_and_reports_honest_outcome},
        {"current_core_id_matches_documented_platform_behavior", test_current_core_id_matches_documented_platform_behavior},
        {"pin_from_worker_thread_does_not_crash", test_pin_from_worker_thread_does_not_crash},
        {"pin_to_absurd_core_id_is_reported_honestly_not_crashed", test_pin_to_absurd_core_id_is_reported_honestly_not_crashed},
    };

    for (const auto& t : tests) {
        t.fn();
        std::printf("[PASS] %s\n", t.name);
    }
    std::printf("All %zu core_affinity tests passed.\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
