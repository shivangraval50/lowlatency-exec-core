// Phase 6 benchmark driver: uses latency_harness.hpp to measure REAL
// operations from phases 1-5 (OrderBook, SpscRingBuffer, price_scan) and
// prints an honest results table.
//
// ---------------------------------------------------------------------------
// Read this before trusting any number this prints
// ---------------------------------------------------------------------------
// Running this binary once on a developer's laptop is a SANITY CHECK that
// the harness itself works end-to-end against real code -- it is NOT the
// authoritative benchmark run this project's README.md "Results" table
// should be filled in from. A rigorous reviewer should insist on, at
// minimum:
//   - A Release build (-O2/-O3, no assertions stripped from timing --
//     see CMakeLists.txt: exec_core_bench links the same `exec_core`
//     library as everything else, built under whatever CMAKE_BUILD_TYPE
//     the configure step chose; this project defaults to Release, but
//     that is a CMake default, not something this file enforces itself --
//     confirm it before trusting a number).
//   - Multiple independent runs (this is a laptop, not an isolated
//     benchmarking rig): background daemons, Spotlight indexing, thermal
//     throttling after sustained load, and other processes competing for
//     the same physical cores are all real, unaccounted-for noise sources
//     here. A single run's p99.9/p99.99 in particular can easily be an
//     artifact of one unlucky scheduler preemption, not a property of the
//     code being measured.
//   - Honest acknowledgment that core pinning (see the pin attempt below)
//     does NOT actually happen on this machine (macOS/Apple Silicon) --
//     so none of these numbers benefit from being isolated from OS
//     scheduler noise/migration the way a real pinned Linux CI run could
//     be. See core_affinity.hpp's file header for why.
//   - Large enough sample counts for the specific percentile claimed: the
//     nearest-rank method (see latency_harness.hpp) needs roughly
//     1/(1-p) samples for percentile p to have single-sample resolution
//     at the tail -- e.g. p99.99 needs on the order of 10,000+ samples
//     just to have ANY resolution, and this driver's default
//     measured_iters (see kMeasuredIters below) is deliberately modest
//     (fast enough to run interactively) rather than tuned to that bar.
//     This is disclosed, not hidden -- see the printed sample count in
//     each results row.
//
// ---------------------------------------------------------------------------
// What is and isn't measured here
// ---------------------------------------------------------------------------
// Every operation below calls into the SAME library code the rest of this
// project builds and tests (exec_core) -- nothing here is a synthetic
// microbenchmark of a toy standalone function. Specifically:
//   - OrderBook::add_limit_order, resting/non-crossing case: every order is
//     a Buy at the SAME fixed price (so no crossing ever happens, and no new
//     price level is created after the first one) -- this isolates the
//     SlabAllocator::allocate() + intrusive-list push_back() cost from the
//     separate question of std::map's O(log P) insertion cost when a brand
//     new price level appears. That second question is real and NOT
//     measured here; it would need its own workload (distinct prices per
//     order) and is left as a documented gap, not silently ignored.
//   - OrderBook::add_limit_order, crossing case: the ask side is
//     pre-seeded (untimed, before the timed loop starts) with exactly
//     (warmup + measured) resting asks of quantity 1 at a fixed price.
//     Each timed call submits a Buy of quantity 1 at that same price, which
//     fully consumes exactly one resting ask and rests nothing itself. This
//     measures "cross and fully consume exactly one resting order" -- a
//     deep multi-level crossing walk (consuming many resting orders across
//     several price levels in one call) is a different, harder case, NOT
//     measured here.
//   - OrderBook::cancel_order: a separate OrderBook is pre-seeded (untimed)
//     with exactly (warmup + measured) resting orders, all at one price, in
//     increasing id order; the timed loop cancels them one at a time in the
//     same order they were inserted (so each cancel is a real hash lookup
//     plus an intrusive-list erase of the current head -- not a synthetic
//     no-op).
//   - SpscRingBuffer<std::uint64_t, ...>: push then immediately pop, by ONE
//     thread, in a loop. This measures single-threaded push+pop call
//     overhead -- it is explicitly NOT a measurement of real cross-thread
//     producer/consumer latency (cache-line ping-pong between two actual
//     cores), which would need a genuinely different two-thread benchmark
//     this project does not yet have.
//   - scan_quantity_at_or_better (dispatches to the NEON path on this
//     machine -- see price_scan.hpp): a fixed flat array of PriceLevel is
//     built once (untimed) at each of two depths (16, 64), and the same
//     scan call is timed repeatedly against it.
//
// ---------------------------------------------------------------------------
// Core pinning attempt
// ---------------------------------------------------------------------------
// This driver calls pin_current_thread_to_core(0) once, before any
// measurement, and prints whatever PinOutcome actually came back (see
// core_affinity.hpp). On this project's local dev machine (macOS/Apple
// Silicon M2) that is expected to be PinOutcome::Unsupported (measured
// kern_return_t KERN_NOT_SUPPORTED -- see core_affinity.hpp's file header);
// this file does NOT claim the measurements below ran on a pinned core
// unless the printed outcome actually says Pinned.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "exec_core/core_affinity.hpp"
#include "exec_core/latency_harness.hpp"
#include "exec_core/order_book.hpp"
#include "exec_core/price_scan.hpp"
#include "exec_core/ring_buffer.hpp"

using namespace exec_core;

namespace {

constexpr std::size_t kWarmupIters = 2'000;
constexpr std::size_t kMeasuredIters = 50'000;

const char* pin_outcome_name(PinOutcome outcome) {
    switch (outcome) {
        case PinOutcome::Pinned:
            return "Pinned (kernel-enforced, Linux)";
        case PinOutcome::BestEffortHint:
            return "BestEffortHint (macOS thread_policy_set accepted, but Apple Silicon "
                   "does not honor it -- not real pinning)";
        case PinOutcome::Unsupported:
            return "Unsupported (OS call rejected the request outright -- no pinning)";
    }
    return "unknown";
}

void report_pin_attempt() {
    const PinResult result = pin_current_thread_to_core(0);
    std::printf("core pinning attempt (core 0): %s [os_status=%d]\n", pin_outcome_name(result.outcome),
                result.os_status);
    if (const std::optional<int> core = current_core_id(); core.has_value()) {
        std::printf("current_core_id() reports: %d\n", *core);
    } else {
        std::printf("current_core_id(): not available on this platform (see core_affinity.hpp)\n");
    }
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// Workload 1: OrderBook::add_limit_order, resting / non-crossing.
// ---------------------------------------------------------------------------
void bench_resting_insert() {
    OrderBook book;
    std::uint64_t next_id = 1;
    constexpr Price kFixedPrice = 100;

    const auto workload = [&]() {
        // Discard the (empty, since nothing crosses) trades vector -- the
        // real work already happened as a side effect on `book`.
        (void)book.add_limit_order(next_id++, Side::Buy, kFixedPrice, /*quantity=*/1);
    };

    const LatencyStats stats = measure_latency_ns(workload, kWarmupIters, kMeasuredIters);
    print_latency_row("OrderBook::add_limit_order (resting, non-crossing)", kWarmupIters, stats);
    std::printf("    -> final order_count=%zu, slab growth_events reflected internally (not exposed here)\n",
                book.order_count());
}

// ---------------------------------------------------------------------------
// Workload 2: OrderBook::add_limit_order, crossing (consumes exactly one
// resting order, rests nothing).
// ---------------------------------------------------------------------------
void bench_crossing_insert() {
    OrderBook book;
    constexpr Price kFixedPrice = 100;
    const std::size_t total_iters = kWarmupIters + kMeasuredIters;

    // Pre-seed the ask side, UNTIMED, with exactly enough resting asks (one
    // per iteration this workload will ever run) so every timed call has a
    // real resting order to cross against and never runs dry mid-measurement.
    std::uint64_t next_ask_id = 1;
    for (std::size_t i = 0; i < total_iters; ++i) {
        (void)book.add_limit_order(next_ask_id++, Side::Sell, kFixedPrice, /*quantity=*/1);
    }

    std::uint64_t next_buy_id = 1'000'000'000ULL;
    const auto workload = [&]() {
        (void)book.add_limit_order(next_buy_id++, Side::Buy, kFixedPrice, /*quantity=*/1);
    };

    const LatencyStats stats = measure_latency_ns(workload, kWarmupIters, kMeasuredIters);
    print_latency_row("OrderBook::add_limit_order (crossing, consumes 1 resting order)", kWarmupIters, stats);
    std::printf("    -> final order_count=%zu (expected 0: every seeded ask should be consumed)\n",
                book.order_count());
}

// ---------------------------------------------------------------------------
// Workload 3: OrderBook::cancel_order.
// ---------------------------------------------------------------------------
void bench_cancel() {
    OrderBook book;
    constexpr Price kFixedPrice = 100;
    const std::size_t total_iters = kWarmupIters + kMeasuredIters;

    // Pre-seed, UNTIMED, exactly one resting order per iteration this
    // workload will ever cancel.
    std::vector<OrderId> ids;
    ids.reserve(total_iters);
    for (std::size_t i = 0; i < total_iters; ++i) {
        const OrderId id = static_cast<OrderId>(i + 1);
        (void)book.add_limit_order(id, Side::Buy, kFixedPrice, /*quantity=*/1);
        ids.push_back(id);
    }

    std::size_t next_index = 0;
    const auto workload = [&]() {
        const bool cancelled = book.cancel_order(ids[next_index++]);
        (void)cancelled; // real work is the mutation of `book`; return value is a correctness check, not the point
    };

    const LatencyStats stats = measure_latency_ns(workload, kWarmupIters, kMeasuredIters);
    print_latency_row("OrderBook::cancel_order", kWarmupIters, stats);
    std::printf("    -> final order_count=%zu (expected 0: every seeded order should be cancelled)\n",
                book.order_count());
}

// ---------------------------------------------------------------------------
// Workload 4: SpscRingBuffer push+pop round trip, single-threaded.
// ---------------------------------------------------------------------------
void bench_ring_buffer_round_trip() {
    SpscRingBuffer<std::uint64_t, 1024> ring;
    std::uint64_t next_value = 0;
    // Volatile sink: the popped value has no other observable use in this
    // benchmark, so without this a sufficiently aggressive optimizer could
    // in principle treat `out` as dead and elide the pop's read entirely.
    // The push/pop calls themselves are real (atomics can't be reordered
    // away regardless), but this removes any doubt about the read.
    static volatile std::uint64_t sink = 0;

    const auto workload = [&]() {
        const bool pushed = ring.try_push(next_value++);
        std::uint64_t out = 0;
        const bool popped = ring.try_pop(out);
        (void)pushed;
        (void)popped;
        sink = out;
    };

    const LatencyStats stats = measure_latency_ns(workload, kWarmupIters, kMeasuredIters);
    print_latency_row("SpscRingBuffer push+pop round trip (single-threaded)", kWarmupIters, stats);
    // Read `sink` back out (not just written) so it is not merely a "set
    // but never read" variable from the compiler's point of view.
    std::printf("    -> last popped value observed: %llu\n", static_cast<unsigned long long>(sink));
}

// ---------------------------------------------------------------------------
// Workload 5: scan_quantity_at_or_better at a couple of realistic depths.
// ---------------------------------------------------------------------------
void bench_price_scan(std::size_t depth) {
    std::vector<PriceLevel> levels;
    levels.reserve(depth);
    for (std::size_t i = 0; i < depth; ++i) {
        levels.push_back(PriceLevel{static_cast<Price>(100 + static_cast<Price>(i)), /*quantity=*/10});
    }
    // Threshold roughly in the middle of the depth so about half the levels
    // are counted as "at or better" -- an arbitrary but fixed, realistic
    // choice, not tuned to make the scan artificially fast/slow.
    const Price limit_price = static_cast<Price>(100 + static_cast<Price>(depth / 2));

    static volatile Quantity sink = 0;
    const auto workload = [&]() {
        const ScanResult result = scan_quantity_at_or_better(levels.data(), levels.size(), limit_price, Side::Buy);
        sink = result.total_quantity;
    };

    const LatencyStats stats = measure_latency_ns(workload, kWarmupIters, kMeasuredIters);
    const std::string name = "scan_quantity_at_or_better (NEON dispatch, depth=" + std::to_string(depth) + ")";
    print_latency_row(name, kWarmupIters, stats);
    std::printf("    -> last scan total_quantity observed: %lld\n", static_cast<long long>(sink));
}

} // namespace

int main() {
#if defined(NDEBUG)
    std::printf("build: Release-like (NDEBUG defined)\n");
#else
    std::printf("build: assertions ENABLED (NDEBUG not defined) -- if this was meant to be an "
                "optimized benchmark run, re-configure with -DCMAKE_BUILD_TYPE=Release and rebuild; "
                "an assert-laden/Debug build's numbers are NOT representative of real latency.\n");
#endif
    std::printf("warmup_iters=%zu measured_iters=%zu per workload (see file header for why these were chosen "
                "and their tail-percentile-resolution limits)\n\n",
                kWarmupIters, kMeasuredIters);

    report_pin_attempt();

    bench_resting_insert();
    bench_crossing_insert();
    bench_cancel();
    bench_ring_buffer_round_trip();
    bench_price_scan(16);
    bench_price_scan(64);

    std::printf("\nThese are a local developer sanity-check run, not the reviewed/authoritative numbers -- "
                "see src/bench_main.cpp's file header and README.md's Results section.\n");
    return 0;
}
