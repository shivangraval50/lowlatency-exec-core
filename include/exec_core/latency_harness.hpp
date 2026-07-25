// Phase 6: a real latency-measurement harness -- warm up, run N timed
// iterations of a caller-supplied workload, and reduce the recorded
// per-iteration durations into a percentile summary (p50/p90/p99/p99.9/
// p99.99, min/max/mean).
//
// ---------------------------------------------------------------------------
// Honest scope
// ---------------------------------------------------------------------------
// This file is the MEASUREMENT MECHANISM, not the results. It does not claim
// or embed any numbers -- those come from actually running
// src/bench_main.cpp (see that file and this project's README.md "Results"
// section for the current, reviewed numbers, if any). Building this harness
// and running it once as a local sanity check is NOT the same thing as
// "the latency numbers are final" -- see bench_main.cpp's file header.
//
// ---------------------------------------------------------------------------
// Why a pre-allocated sample buffer, and why warmup iterations are discarded
// ---------------------------------------------------------------------------
// The whole point of a latency percentile is that it is sensitive to rare,
// large outliers (that's what p99.9/p99.99 exist to surface) -- so the
// measurement loop itself must not introduce its own outliers. Concretely:
//   - `measure_latency_ns` resizes its `samples_ns` buffer to
//     `measured_iters` ONE time, before entering the timed loop. If it
//     instead called `push_back` inside the loop, an occasional
//     std::vector reallocation would show up as a fake latency spike in the
//     very tail percentiles this harness exists to measure honestly.
//   - `warmup_iters` runs of the workload happen first and are NOT timed or
//     recorded at all: the intent is to let one-time costs that are not
//     representative of steady-state (e.g. first-touch page faults, cold
//     branch predictor/cache state, the SlabAllocator's very first chunk
//     allocation happening lazily on first use, OS-level lazy binding of
//     this binary's own symbols) happen before the clock starts, so the
//     recorded samples reflect steady-state operation cost, not one-time
//     process startup cost. This is a real methodological choice with a
//     real tradeoff: it can also hide a genuinely-recurring cost if the
//     workload itself is not representative of steady state (e.g. if it
//     causes an allocator growth event -- see SlabAllocator::growth_events()
//     -- AFTER warmup, on iteration 4097 of a 4096-capacity pool, that
//     WOULD show up in the timed samples, deliberately, because it's a real
//     recurring possibility, not a one-time startup artifact). Callers of
//     this harness (bench_main.cpp) are expected to reason about which
//     category their workload's costs fall into and choose warmup_iters
//     accordingly -- this header does not (and cannot) know that for them.
//
// ---------------------------------------------------------------------------
// Percentile method: nearest-rank, computed with EXACT INTEGER arithmetic
// ---------------------------------------------------------------------------
// `compute_latency_stats` sorts the (copied) samples ascending and applies
// the "nearest-rank" method (see
// https://en.wikipedia.org/wiki/Percentile#The_nearest_rank_method): for a
// percentile P in (0, 100] and N samples, the reported value is the sample
// at 1-based rank `ceil(P/100 * N)`, clamped to [1, N]. This is a real,
// standard, well-defined percentile definition (not an approximation and
// not linear interpolation between adjacent ranks -- some other tools, e.g.
// numpy's default, use linear interpolation instead and can report a
// slightly different number for the same data; this file uses nearest-rank
// because it is simple, has no interpolation edge cases, and always
// reports an actual observed sample value rather than a value that was
// never measured).
//
// `nearest_rank_percentile`'s rank arithmetic is done entirely in integer
// arithmetic, not `double`, specifically to avoid a real, previously-shipped
// bug: computing `std::ceil((p / 100.0) * N)` directly in IEEE-754 `double`
// is wrong for some inputs, because a percentage like 99.9 is not exactly
// representable as a binary double (same reason 0.1 isn't) -- for any N
// where the mathematically-exact rank is itself an integer (e.g. any exact
// multiple of 1000, for p=99.9), that tiny representation error can push
// the computed value a hair past the integer, and `ceil` then rounds up to
// the wrong rank. The fix: convert the `double` percentage to an exact
// integer scaled by 100 via round-to-nearest (not truncation, not
// ceiling -- see the implementation comment in latency_harness.cpp for why
// rounding is the correct conversion), then compute
// `rank = ceil(p_scaled_1e2 * N / 10000)` via pure integer ceiling-division
// (`(a + b - 1) / b`), with the multiply widened to `unsigned __int128` as
// a deliberate, free overflow safety margin. No `double` is touched after
// the initial percentage-to-integer conversion.
//
// This function takes a plain `std::vector<uint64_t>` of nanosecond
// durations and is deliberately independent of `std::chrono`/the timed loop
// below, specifically so it can be tested against a known synthetic
// distribution (e.g. 1..1000) without needing to run any real timed
// workload at all.
//
// Per this project's build methodology, the tester phase writes the
// authoritative correctness tests for `compute_latency_stats` against such
// synthetic distributions; this file only needs to be internally consistent
// and documented well enough for that to be checkable.
//
// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
// `std::chrono::steady_clock` -- monotonic (never goes backwards, unaffected
// by wall-clock adjustments/NTP steps), which is the correct clock family
// for measuring elapsed durations. This project does not reach for a
// platform-specific high-resolution counter (e.g. `mach_absolute_time` on
// macOS, `clock_gettime(CLOCK_MONOTONIC_RAW)` on Linux, or reading the raw
// TSC via `rdtsc`) -- `steady_clock` is required by the standard to be
// monotonic and is, on every platform/toolchain this project actually
// builds on (AppleClang on this M2; GCC/Clang on CI's ubuntu-latest), a thin
// wrapper over exactly those same underlying calls, so it already gives
// nanosecond-representable, low-overhead timestamps without hand-rolling a
// platform-specific timer. A known limitation this project is explicit
// about (see README.md): steady_clock's actual resolution/overhead is
// itself platform- and hardware-dependent and is NOT independently
// benchmarked here (e.g. "what does an empty measured workload read as" is
// not a case this harness separately reports) -- very fast operations
// (single-digit nanoseconds) may be dominated by the clock call's own
// overhead rather than the workload; this is disclosed, not hidden.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace exec_core {

// Percentile summary of a set of recorded latency samples, all in whole
// nanoseconds except `mean_ns` (which is naturally fractional).
struct LatencyStats {
    std::size_t count = 0;
    std::uint64_t min_ns = 0;
    std::uint64_t max_ns = 0;
    double mean_ns = 0.0;
    std::uint64_t p50_ns = 0;
    std::uint64_t p90_ns = 0;
    std::uint64_t p99_ns = 0;
    std::uint64_t p999_ns = 0;  // p99.9
    std::uint64_t p9999_ns = 0; // p99.99
};

// Nearest-rank percentile of an ALREADY-SORTED-ASCENDING sample set. `p` is
// a percentage in (0, 100]; out-of-range values are clamped. Returns 0 if
// `sorted_ascending` is empty (there is no meaningful percentile of zero
// samples; callers are expected not to do this, but this avoids UB/a crash
// over throwing for what is, in a benchmark driver, a programmer error).
// Exposed (not just used internally by compute_latency_stats) specifically
// so it can be unit-tested directly against a known sorted synthetic
// distribution without the surrounding sort/reduce machinery.
std::uint64_t nearest_rank_percentile(const std::vector<std::uint64_t>& sorted_ascending, double p);

// Reduces a set of recorded latency samples (nanoseconds, any order) into
// a full LatencyStats summary. Takes `samples_ns` BY VALUE deliberately: it
// sorts its own copy internally, so callers (including tests) can pass a
// literal/temporary vector, or pass their own timed-loop buffer and keep
// using the original afterward. This is the function this project's tester
// phase is expected to check against synthetic distributions (e.g. 1..N)
// with known, hand-computable nearest-rank percentiles.
LatencyStats compute_latency_stats(std::vector<std::uint64_t> samples_ns);

// Runs `workload` `warmup_iters` times (untimed, discarded -- see file
// header), then `measured_iters` times with each iteration's wall-clock
// duration recorded via std::chrono::steady_clock into a buffer allocated
// ONCE before the timed loop starts (see file header for why). Returns the
// reduced LatencyStats over the measured iterations.
//
// Templated (not declared in a .cpp) because `Fn` is a caller-supplied
// callable type, same reason ring_buffer.hpp/slab_allocator.hpp are
// header-only templates rather than living in a .cpp.
//
// `workload` must be callable with no arguments; any return value is
// discarded. It is the caller's responsibility (see bench_main.cpp) to
// make sure `workload`'s side effects are real (e.g. actually mutate an
// OrderBook/ring buffer) so the compiler has no basis to optimize the call
// away -- this harness does not attempt to defeat compiler optimizations
// itself (e.g. no inline asm "escape"/"clobber" tricks), since every
// workload this project measures already has real, externally-visible
// memory side effects through an opaque (separately-compiled) library call.
template <typename Fn>
LatencyStats measure_latency_ns(Fn&& workload, std::size_t warmup_iters, std::size_t measured_iters) {
    for (std::size_t i = 0; i < warmup_iters; ++i) {
        workload();
    }

    // Allocated once, before the timed loop -- see file header. resize()
    // (not reserve()+push_back() in the loop) so every element exists
    // up front and the loop body only ever does an in-place assignment.
    std::vector<std::uint64_t> samples_ns;
    samples_ns.resize(measured_iters);

    for (std::size_t i = 0; i < measured_iters; ++i) {
        const auto start = std::chrono::steady_clock::now();
        workload();
        const auto end = std::chrono::steady_clock::now();
        samples_ns[i] =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    return compute_latency_stats(std::move(samples_ns));
}

// Pretty-prints one row of a results table to stdout:
// name, iteration count, min/p50/p90/p99/p99.9/p99.99/max/mean, all in
// nanoseconds. Kept as a plain free function (not iostream operator
// overloading) so bench_main.cpp's table formatting is visible in one
// place. Not templated -- lives in latency_harness.cpp, same pattern as
// order_book.cpp/core_affinity.cpp/price_scan.cpp for non-template code.
void print_latency_row(const std::string& name, std::size_t warmup_iters, const LatencyStats& stats);

} // namespace exec_core
