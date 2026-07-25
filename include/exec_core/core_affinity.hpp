// Thread-to-CPU-core pinning, abstracted across Linux and macOS -- and
// deliberately honest about the fact that those two platforms do NOT offer
// the same guarantee.
//
// ---------------------------------------------------------------------------
// Why this file is careful to distinguish "pinned" from "hinted"
// ---------------------------------------------------------------------------
// On Linux, `pthread_setaffinity_np` (a thin wrapper over `sched_setaffinity`)
// is a real, binding instruction to the kernel scheduler: after a successful
// call, the calling thread will only ever be scheduled on the requested
// core(s), and this is independently verifiable from the same thread via
// `sched_getcpu()`. This is what actually runs and gets verified on this
// project's CI (ubuntu-latest; see .github/workflows/ci.yml).
//
// On macOS, the closest public API is `thread_policy_set` with
// `THREAD_AFFINITY_POLICY`. Apple's own documentation
// (Kernel Programming Guide / mach headers) describes this as a *hint* the
// scheduler MAY use to co-locate threads that share an affinity tag onto the
// same L2 cache domain -- it was never a "pin this thread to logical CPU N"
// API even on Intel Macs. On Apple Silicon it is documented and widely
// observed to be unhonored.
//
// Locally verified on this project's actual dev machine (M2, macOS,
// AppleClang 16): the call does not even silently no-op -- it returns
// `kern_return_t` 46 (`KERN_NOT_SUPPORTED`), i.e. the kernel itself rejects
// the request outright rather than pretending to accept a hint it then
// ignores. This is a real, measured result (see the phase-4 scratch check
// run against this header), not an assumption -- and it means this file's
// pin_current_thread_to_core() correctly reports PinOutcome::Unsupported
// (not a fabricated PinOutcome::BestEffortHint) on this hardware, because
// its result path is driven by the actual `kr` returned, not by a hardcoded
// per-platform assumption. On older Intel Macs where `thread_policy_set`
// might return KERN_SUCCESS, this same code path would report
// BestEffortHint instead -- untested here, since this project has no Intel
// Mac to verify that branch on, so that specific claim is left as
// documented-but-unverified rather than asserted as measured fact.
// There is no working equivalent of `sched_getcpu()` on Apple Silicon that
// this project is willing to rely on to verify anything past that either.
//
// Concretely, this means: do NOT write code (here or anywhere calling this
// header) that treats a macOS `BestEffortHint` result as if it were the same
// as Linux's `Pinned` result. They are not the same guarantee, and
// benchmarks/claims must say which one actually happened.
#pragma once

#include <optional>

namespace exec_core {

// What actually happened when pin_current_thread_to_core() was called.
enum class PinOutcome {
    // Linux only: sched_setaffinity/pthread_setaffinity_np reported success.
    // This is a real, kernel-enforced guarantee -- verifiable via
    // current_core_id() returning the requested core (modulo the OS
    // rescheduling the thread again later if pin_current_thread_to_core()
    // is called again with a different mask).
    Pinned,
    // macOS only: thread_policy_set(THREAD_AFFINITY_POLICY) was accepted by
    // the kernel, but per this file's header comment, Apple Silicon does not
    // honor this hint at all (and even on the Intel Macs where it did
    // something, it never meant "pinned to core N" -- it meant "grouped with
    // other same-tag threads for possible L2 locality"). Treat this as
    // "no meaningful effect confirmed," not as pinning.
    BestEffortHint,
    // The underlying OS call failed (see PinResult::os_status for the
    // errno/kern_return_t), or this platform has no affinity API this file
    // implements at all.
    Unsupported,
};

struct PinResult {
    PinOutcome outcome;
    // Linux: `errno` value from pthread_setaffinity_np() (0 on success).
    // macOS: the `kern_return_t` from thread_policy_set() (KERN_SUCCESS == 0
    // on success, but see PinOutcome::BestEffortHint -- success here is not
    // pinning).
    int os_status = 0;
};

// Requests that the *calling* thread be pinned to logical core `core_id`
// (0-based). See file header for what this does and does not guarantee on
// each platform. This function itself decides, based on
// `__linux__`/`__APPLE__`, which real OS call (if any) to make -- there is
// no "fake" fallback implementation that pretends to succeed.
PinResult pin_current_thread_to_core(unsigned core_id);

// Returns the OS-reported logical core the calling thread is running on
// right now, if this platform can honestly answer that question.
//   - Linux: sched_getcpu() (wrapped in std::optional; failure -> nullopt).
//   - macOS: std::nullopt, always. There is no public, documented API this
//     project is willing to rely on to answer this on Apple Silicon --
//     reaching for an undocumented/private syscall to manufacture an answer
//     would be exactly the kind of fabricated-result behavior this project
//     explicitly avoids (see CLAUDE.md). A caller on macOS therefore cannot
//     verify pin_current_thread_to_core()'s effect locally, and should not
//     pretend it can.
std::optional<int> current_core_id();

} // namespace exec_core
