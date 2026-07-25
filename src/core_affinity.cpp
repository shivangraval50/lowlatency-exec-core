#include "exec_core/core_affinity.hpp"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <cerrno>
#include <cstring>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#endif

namespace exec_core {

#if defined(__linux__)

// Real, kernel-enforced pinning. sched_setaffinity/pthread_setaffinity_np
// restricts the calling thread to the given core; a successful return here
// is a binding guarantee, not a hint (see core_affinity.hpp file header).
PinResult pin_current_thread_to_core(unsigned core_id) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(static_cast<int>(core_id), &cpu_set);

    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);
    if (rc == 0) {
        return PinResult{PinOutcome::Pinned, 0};
    }
    // pthread_setaffinity_np returns the errno value directly (it does not
    // set the global errno itself), so rc here IS the status to report.
    return PinResult{PinOutcome::Unsupported, rc};
}

std::optional<int> current_core_id() {
    const int cpu = sched_getcpu();
    if (cpu < 0) {
        return std::nullopt;
    }
    return cpu;
}

#elif defined(__APPLE__)

// Best-effort ONLY -- see core_affinity.hpp's file header for why this can
// never honestly return PinOutcome::Pinned. thread_affinity_policy_data_t's
// single field is an opaque "affinity set tag", not a literal logical-core
// index: threads that share a tag are hinted to the scheduler as wanting
// L2-cache locality with each other. We reuse `core_id` as that tag (a
// reasonable choice -- it still lets a caller group threads deterministically
// -- but it is NOT "run on logical core `core_id`"). Verified locally on
// this project's M2: `kr` comes back KERN_NOT_SUPPORTED (46), so this
// function reports PinOutcome::Unsupported here, not BestEffortHint --
// the outcome below is driven by the real `kr`, never hardcoded.
PinResult pin_current_thread_to_core(unsigned core_id) {
    thread_affinity_policy_data_t policy{static_cast<integer_t>(core_id)};
    const kern_return_t kr = thread_policy_set(
        pthread_mach_thread_np(pthread_self()),
        THREAD_AFFINITY_POLICY,
        reinterpret_cast<thread_policy_t>(&policy),
        THREAD_AFFINITY_POLICY_COUNT);

    if (kr == KERN_SUCCESS) {
        // Accepted by the kernel -- but see file header: on Apple Silicon
        // this is a documented no-op. Never report Pinned here.
        return PinResult{PinOutcome::BestEffortHint, static_cast<int>(kr)};
    }
    return PinResult{PinOutcome::Unsupported, static_cast<int>(kr)};
}

std::optional<int> current_core_id() {
    // Deliberately unimplemented: see core_affinity.hpp file header. No
    // public, documented API this project relies on can answer "which
    // logical core is this thread on" on Apple Silicon.
    return std::nullopt;
}

#else

// Any other platform: no affinity implementation exists here at all. Report
// this honestly rather than silently doing nothing and claiming success.
PinResult pin_current_thread_to_core(unsigned /*core_id*/) {
    return PinResult{PinOutcome::Unsupported, 0};
}

std::optional<int> current_core_id() {
    return std::nullopt;
}

#endif

} // namespace exec_core
