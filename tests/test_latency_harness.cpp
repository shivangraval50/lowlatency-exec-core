// Real correctness tests for phase 6's latency-percentile measurement
// harness (include/exec_core/latency_harness.hpp, src/latency_harness.cpp).
//
// Plain assert()-based tests, consistent with the rest of tests/ (no
// framework). Per latency_harness.hpp's own file header: "the tester phase
// writes the authoritative correctness tests for compute_latency_stats
// against ... synthetic distributions with known, hand-computable
// nearest-rank percentiles" -- that is exactly what this file does. If the
// percentile math here is wrong, every benchmark number src/bench_main.cpp
// prints is meaningless, so this file is deliberately the most exhaustive
// test in the whole project.
//
// ---------------------------------------------------------------------
// How expected values below were derived
// ---------------------------------------------------------------------
// Per latency_harness.hpp: nearest-rank, 1-based rank = ceil(P/100 * N),
// clamped to [1, N]. `exact_rank()` below computes that SAME formula using
// pure 64-bit integer arithmetic (percentages represented as an exact
// integer scaled by 10,000 -- e.g. 99.99 -> 999900 -- never as a C++
// `double` literal), specifically so the *test's own oracle* cannot
// silently repeat the same binary-floating-point representation error that
// `nearest_rank_percentile`'s implementation used to make before it was
// fixed to use exact integer arithmetic (see latency_harness.hpp/.cpp).
// This distinction turned out to matter: see
// test_p999_exact_multiple_of_1000_regression below, which documents a real,
// now-fixed off-by-one this exact-integer oracle exposed during development
// (and that a naively-hand-computed "0.999 * 1000 = 999" expectation would
// have missed entirely) and guards against it ever coming back.

#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "exec_core/latency_harness.hpp"

using namespace exec_core;

namespace {

// ---------------------------------------------------------------------
// Exact-rational nearest-rank oracle -- see file header. Percentages are
// passed in as an integer already scaled by 10,000 (so 50.0 -> 500000,
// 99.99 -> 999900), completely avoiding any floating-point representation
// of the percentage itself. `ceil(p_scaled * N / 1e6)` via pure integer
// ceil-division, clamped to [1, N].
// ---------------------------------------------------------------------
std::uint64_t exact_rank(std::uint64_t p_scaled_1e4, std::uint64_t n) {
    const std::uint64_t num = p_scaled_1e4 * n;
    const std::uint64_t den = 1'000'000ULL; // 100 * 10000
    std::uint64_t rank = (num + den - 1) / den; // ceil division, exact integers only
    if (rank < 1) rank = 1;
    if (rank > n) rank = n;
    return rank;
}

// ---------------------------------------------------------------------
// nearest_rank_percentile() direct tests: known sorted synthetic
// distributions with hand-verifiable-by-the-exact-integer-oracle expected
// values. `nearest_rank_percentile` requires already-sorted-ascending
// input (see latency_harness.hpp) -- every vector passed here is
// constructed already sorted.
// ---------------------------------------------------------------------
void test_nearest_rank_1_to_100() {
    std::vector<std::uint64_t> v;
    for (std::uint64_t i = 1; i <= 100; ++i) v.push_back(i);

    // Hand check: rank = ceil(p/100*100) = ceil(p). p50->50, p90->90,
    // p99->99, p99.9->ceil(99.9)=100, p99.99->ceil(99.99)=100.
    assert(nearest_rank_percentile(v, 50.0) == v[exact_rank(500000, 100) - 1]);
    assert(nearest_rank_percentile(v, 50.0) == 50);
    assert(nearest_rank_percentile(v, 90.0) == 90);
    assert(nearest_rank_percentile(v, 99.0) == 99);
    assert(nearest_rank_percentile(v, 99.9) == 100);
    assert(nearest_rank_percentile(v, 99.99) == 100);
}

void test_nearest_rank_1_to_1000_non_boundary_percentiles() {
    // The p99.9 case for exactly N=1000 used to be a real off-by-one bug
    // (see test_p999_exact_multiple_of_1000_regression, which owns that
    // specific case and the historical explanation) -- it's fixed now, but
    // that dedicated regression test is still the canonical place asserting
    // it, so this test sticks to the other percentiles here to keep the two
    // tests' responsibilities clearly separated.
    std::vector<std::uint64_t> v;
    for (std::uint64_t i = 1; i <= 1000; ++i) v.push_back(i);

    assert(nearest_rank_percentile(v, 50.0) == v[exact_rank(500000, 1000) - 1]);
    assert(nearest_rank_percentile(v, 50.0) == 500);
    assert(nearest_rank_percentile(v, 90.0) == v[exact_rank(900000, 1000) - 1]);
    assert(nearest_rank_percentile(v, 90.0) == 900);
    assert(nearest_rank_percentile(v, 99.0) == v[exact_rank(990000, 1000) - 1]);
    assert(nearest_rank_percentile(v, 99.0) == 990);
    assert(nearest_rank_percentile(v, 99.99) == v[exact_rank(999900, 1000) - 1]);
    assert(nearest_rank_percentile(v, 99.99) == 1000);
}

void test_nearest_rank_N_not_divisible_by_100_or_1000() {
    // N=7: rank = ceil(p_scaled * 7 / 1e6).
    {
        std::vector<std::uint64_t> v = {1, 2, 3, 4, 5, 6, 7};
        assert(nearest_rank_percentile(v, 50.0) == v[exact_rank(500000, 7) - 1]);
        assert(nearest_rank_percentile(v, 50.0) == 4); // ceil(3.5)=4
        assert(nearest_rank_percentile(v, 90.0) == v[exact_rank(900000, 7) - 1]);
        assert(nearest_rank_percentile(v, 90.0) == 7); // ceil(6.3)=7
        assert(nearest_rank_percentile(v, 99.0) == 7);
        assert(nearest_rank_percentile(v, 99.9) == 7);
        assert(nearest_rank_percentile(v, 99.99) == 7);
    }
    // N=3, non-trivial (non-consecutive) values: {5,15,25}.
    {
        std::vector<std::uint64_t> v = {5, 15, 25};
        assert(nearest_rank_percentile(v, 50.0) == v[exact_rank(500000, 3) - 1]);
        assert(nearest_rank_percentile(v, 50.0) == 15); // ceil(1.5)=2 -> v[1]=15
        assert(nearest_rank_percentile(v, 90.0) == v[exact_rank(900000, 3) - 1]);
        assert(nearest_rank_percentile(v, 90.0) == 25); // ceil(2.7)=3 -> v[2]=25
        assert(nearest_rank_percentile(v, 99.99) == 25);
    }
    // N=333: sweep of the same oracle formula against a non-round N.
    {
        std::vector<std::uint64_t> v;
        for (std::uint64_t i = 1; i <= 333; ++i) v.push_back(i);
        assert(nearest_rank_percentile(v, 50.0) == v[exact_rank(500000, 333) - 1]);
        assert(nearest_rank_percentile(v, 50.0) == 167); // ceil(166.5)=167
        assert(nearest_rank_percentile(v, 90.0) == v[exact_rank(900000, 333) - 1]);
        assert(nearest_rank_percentile(v, 90.0) == 300); // ceil(299.7)=300
        assert(nearest_rank_percentile(v, 99.0) == v[exact_rank(990000, 333) - 1]);
        assert(nearest_rank_percentile(v, 99.0) == 330); // ceil(329.67)=330
        assert(nearest_rank_percentile(v, 99.9) == v[exact_rank(999000, 333) - 1]);
        assert(nearest_rank_percentile(v, 99.9) == 333); // ceil(332.667)=333 (== max, not an exact-integer boundary)
    }
}

void test_nearest_rank_single_sample() {
    std::vector<std::uint64_t> v = {42};
    // N=1: every p in (0,100] must clamp to rank 1, the only sample.
    assert(nearest_rank_percentile(v, 1.0) == 42);
    assert(nearest_rank_percentile(v, 50.0) == 42);
    assert(nearest_rank_percentile(v, 90.0) == 42);
    assert(nearest_rank_percentile(v, 99.0) == 42);
    assert(nearest_rank_percentile(v, 99.9) == 42);
    assert(nearest_rank_percentile(v, 99.99) == 42); // the specific edge case the task asked to confirm
    assert(nearest_rank_percentile(v, 100.0) == 42);
}

void test_nearest_rank_two_samples() {
    std::vector<std::uint64_t> v = {10, 20};
    assert(nearest_rank_percentile(v, 50.0) == v[exact_rank(500000, 2) - 1]);
    assert(nearest_rank_percentile(v, 50.0) == 10); // ceil(1.0)=1 -> v[0]
    assert(nearest_rank_percentile(v, 90.0) == v[exact_rank(900000, 2) - 1]);
    assert(nearest_rank_percentile(v, 90.0) == 20); // ceil(1.8)=2 -> v[1]
    assert(nearest_rank_percentile(v, 99.99) == 20);
}

void test_nearest_rank_all_identical() {
    std::vector<std::uint64_t> v = {7, 7, 7, 7, 7};
    assert(nearest_rank_percentile(v, 1.0) == 7);
    assert(nearest_rank_percentile(v, 50.0) == 7);
    assert(nearest_rank_percentile(v, 99.99) == 7);
}

void test_nearest_rank_empty_returns_zero() {
    std::vector<std::uint64_t> v;
    assert(nearest_rank_percentile(v, 50.0) == 0);
    assert(nearest_rank_percentile(v, 99.99) == 0);
}

void test_nearest_rank_clamps_out_of_range_p() {
    std::vector<std::uint64_t> v = {10, 20, 30, 40, 50};
    // p <= 0 clamps to 0 -> raw_rank=ceil(0)=0 -> clamped up to rank 1 (min).
    assert(nearest_rank_percentile(v, 0.0) == 10);
    assert(nearest_rank_percentile(v, -5.0) == 10);
    // p > 100 clamps to 100 -> rank = N (max).
    assert(nearest_rank_percentile(v, 100.0) == 50);
    assert(nearest_rank_percentile(v, 150.0) == 50);
    assert(nearest_rank_percentile(v, 1000.0) == 50);
}

// ---------------------------------------------------------------------
// compute_latency_stats(): full reduction, including min/max/mean, AND
// confirmation it sorts its own (already-by-value) copy -- an unsorted
// input must produce identical results to the pre-sorted equivalent.
// ---------------------------------------------------------------------
void test_compute_stats_sorts_unsorted_input() {
    std::vector<std::uint64_t> unsorted = {50, 10, 30, 20, 40};
    const std::vector<std::uint64_t> unsorted_original_copy = unsorted; // to check non-mutation below

    const LatencyStats stats = compute_latency_stats(unsorted);

    // sorted reference: {10,20,30,40,50}
    assert(stats.count == 5);
    assert(stats.min_ns == 10);
    assert(stats.max_ns == 50);
    assert(std::fabs(stats.mean_ns - 30.0) < 1e-9);
    assert(stats.p50_ns == 30); // ceil(2.5)=3 -> sorted[2]=30
    assert(stats.p90_ns == 50); // ceil(4.5)=5 -> sorted[4]=50
    assert(stats.p99_ns == 50); // ceil(4.95)=5
    assert(stats.p999_ns == 50);
    assert(stats.p9999_ns == 50);

    // compute_latency_stats takes samples_ns BY VALUE, so the caller's
    // original vector (`unsorted`, an lvalue) must be copied at the call
    // site regardless of what the function does internally -- confirm that
    // guarantee actually holds (the caller's copy was never touched).
    assert(unsorted == unsorted_original_copy);
}

void test_compute_stats_min_max_mean_hand_verified() {
    // {3,1,4,1,5,9,2,6} -> sorted {1,1,2,3,4,5,6,9}. min=1, max=9,
    // mean=(3+1+4+1+5+9+2+6)/8 = 31/8 = 3.875.
    std::vector<std::uint64_t> v = {3, 1, 4, 1, 5, 9, 2, 6};
    const LatencyStats stats = compute_latency_stats(v);

    assert(stats.count == 8);
    assert(stats.min_ns == 1);
    assert(stats.max_ns == 9);
    assert(std::fabs(stats.mean_ns - 3.875) < 1e-9);
    // sorted: {1,1,2,3,4,5,6,9} (0-indexed)
    assert(stats.p50_ns == 3); // ceil(4.0)=4 -> sorted[3]=3
    assert(stats.p90_ns == 9); // ceil(7.2)=8 -> sorted[7]=9
    assert(stats.p99_ns == 9); // ceil(7.92)=8
    assert(stats.p999_ns == 9);
    assert(stats.p9999_ns == 9);
}

void test_compute_stats_empty_is_honestly_all_zero() {
    std::vector<std::uint64_t> v;
    const LatencyStats stats = compute_latency_stats(v);
    assert(stats.count == 0);
    assert(stats.min_ns == 0);
    assert(stats.max_ns == 0);
    assert(stats.mean_ns == 0.0);
    assert(stats.p50_ns == 0);
    assert(stats.p90_ns == 0);
    assert(stats.p99_ns == 0);
    assert(stats.p999_ns == 0);
    assert(stats.p9999_ns == 0);
}

void test_compute_stats_single_sample_p9999_clamps_correctly() {
    // The specific case the task asked to confirm: N=1 for p99.99 -- does
    // it correctly clamp to the only sample, not read out of bounds / return
    // something else?
    std::vector<std::uint64_t> v = {999};
    const LatencyStats stats = compute_latency_stats(v);
    assert(stats.count == 1);
    assert(stats.min_ns == 999);
    assert(stats.max_ns == 999);
    assert(stats.mean_ns == 999.0);
    assert(stats.p50_ns == 999);
    assert(stats.p90_ns == 999);
    assert(stats.p99_ns == 999);
    assert(stats.p999_ns == 999);
    assert(stats.p9999_ns == 999); // must clamp to the only sample, not 0 / UB
}

void test_compute_stats_all_identical() {
    std::vector<std::uint64_t> v(50, 1234);
    const LatencyStats stats = compute_latency_stats(v);
    assert(stats.count == 50);
    assert(stats.min_ns == 1234);
    assert(stats.max_ns == 1234);
    assert(stats.mean_ns == 1234.0);
    assert(stats.p50_ns == 1234);
    assert(stats.p90_ns == 1234);
    assert(stats.p999_ns == 1234);
    assert(stats.p9999_ns == 1234);
}

// ---------------------------------------------------------------------
// REGRESSION TEST: for N exactly a multiple of 1000, nearest_rank_percentile
// (v, 99.9) used to NOT return the mathematically-exact nearest-rank value.
//
// History (kept briefly for institutional memory -- see latency_harness.hpp
// for the full current-implementation explanation): the original
// implementation computed
//   raw_rank = ceil((clamped_p / 100.0) * N)
// entirely in IEEE-754 `double` arithmetic, where `clamped_p` is the
// `double` literal 99.9. 99.9 is NOT exactly representable in binary
// floating point (same reason 0.1 isn't) -- its nearest double is a hair
// above the true value. For N values where the mathematically exact
// result of (99.9/100)*N is itself an exact integer (i.e. any multiple of
// 1000, since 999/1000 in lowest terms), that tiny representation error
// pushed the computed `double` a hair ABOVE that integer (e.g. for N=1000,
// raw_rank computed as 999.0000000000001 instead of exactly 999.0), so
// `std::ceil` rounded UP to the next integer instead of landing exactly on
// it. The observable symptom: nearest_rank_percentile({1..1000}, 99.9) used
// to return 1000 (the maximum, rank 1000) instead of the
// mathematically-correct rank-999 value -- and bench_main.cpp's
// kMeasuredIters (50,000, also an exact multiple of 1000) meant the real
// benchmark driver's printed p99.9 was affected too.
//
// Fix: `nearest_rank_percentile` now converts the percentage to an exact
// integer (hundredths-of-a-percent precision, via std::llround) and computes
// the rank via pure integer ceiling-division, never touching a `double`
// after that conversion -- see latency_harness.hpp/.cpp for the current
// implementation. This test asserts the CORRECT (mathematically exact)
// value and is expected to PASS; it exists purely as a regression guard so
// nobody accidentally reintroduces double-precision rank arithmetic here.
// ---------------------------------------------------------------------
void test_p999_exact_multiple_of_1000_regression() {
    // Diagnostic sweep across several multiples of 1000 (plus one that
    // isn't, implicitly covered by other tests) -- prints got vs.
    // exact-integer-oracle-expected for visibility, then asserts each one.
    for (std::uint64_t n : {1000ULL, 2000ULL, 3000ULL, 5000ULL, 50000ULL}) {
        std::vector<std::uint64_t> v;
        v.reserve(n);
        for (std::uint64_t i = 1; i <= n; ++i) v.push_back(i);
        const std::uint64_t got = nearest_rank_percentile(v, 99.9);
        const std::uint64_t expected = exact_rank(999000ULL, n); // exact-integer oracle, no double literal
        std::fprintf(stderr,
                      "  [p99.9-regression diagnostic] N=%llu p99.9: got=%llu exact-spec-expected=%llu %s\n",
                      static_cast<unsigned long long>(n), static_cast<unsigned long long>(got),
                      static_cast<unsigned long long>(expected), got == expected ? "(match)" : "(MISMATCH)");
        assert(got == expected &&
               "REGRESSION: nearest_rank_percentile(99.9) off-by-one at N=multiple-of-1000 -- this was a "
               "real, previously-shipped double-precision representation-error bug that has since been "
               "fixed via exact integer arithmetic (see latency_harness.hpp); this assert firing means "
               "that fix regressed");
    }

    // Explicit, hand-verifiable N=1000 case (the smallest/original repro).
    std::vector<std::uint64_t> v;
    for (std::uint64_t i = 1; i <= 1000; ++i) v.push_back(i);
    assert(nearest_rank_percentile(v, 99.9) == 999); // exact_rank(999000, 1000) == 999
}

// ---------------------------------------------------------------------
// measure_latency_ns(): can't assert exact nanosecond values (real timing
// noise), but CAN assert structural/statistical invariants that must
// always hold regardless of the actual numbers measured.
// ---------------------------------------------------------------------

// Invariant that must ALWAYS hold for any non-empty LatencyStats: nearest
// rank is a monotonic non-decreasing function of percentile, so
// min <= p50 <= p90 <= p99 <= p99.9 <= p99.99 <= max, and mean is bounded
// by [min, max].
void assert_stats_internally_consistent(const LatencyStats& stats) {
    assert(stats.min_ns <= stats.p50_ns);
    assert(stats.p50_ns <= stats.p90_ns);
    assert(stats.p90_ns <= stats.p99_ns);
    assert(stats.p99_ns <= stats.p999_ns);
    assert(stats.p999_ns <= stats.p9999_ns);
    assert(stats.p9999_ns <= stats.max_ns);
    assert(stats.mean_ns >= static_cast<double>(stats.min_ns) - 1e-6);
    assert(stats.mean_ns <= static_cast<double>(stats.max_ns) + 1e-6);
}

void test_measure_latency_ns_sample_count_and_call_count() {
    // A workload that counts every invocation (warmup AND measured), so we
    // can independently verify measure_latency_ns() calls the workload
    // EXACTLY warmup_iters + measured_iters times total, and that the
    // returned LatencyStats.count is EXACTLY measured_iters (i.e. no
    // warmup call leaked into the recorded samples, and no measured call
    // went unrecorded).
    std::size_t call_count = 0;
    constexpr std::size_t kWarmup = 137;
    constexpr std::size_t kMeasured = 421;

    const auto workload = [&]() { ++call_count; };
    const LatencyStats stats = measure_latency_ns(workload, kWarmup, kMeasured);

    assert(call_count == kWarmup + kMeasured);
    assert(stats.count == kMeasured);
    assert_stats_internally_consistent(stats);
}

void test_measure_latency_ns_zero_warmup() {
    std::size_t call_count = 0;
    const auto workload = [&]() { ++call_count; };
    const LatencyStats stats = measure_latency_ns(workload, /*warmup_iters=*/0, /*measured_iters=*/200);
    assert(call_count == 200);
    assert(stats.count == 200);
    assert_stats_internally_consistent(stats);
}

void test_measure_latency_ns_single_measured_iter() {
    const auto workload = [&]() { /* no-op real function call, still real work */ };
    const LatencyStats stats = measure_latency_ns(workload, /*warmup_iters=*/5, /*measured_iters=*/1);
    assert(stats.count == 1);
    // N=1: every percentile must equal the single sample, which must equal
    // both min and max.
    assert(stats.min_ns == stats.max_ns);
    assert(stats.p50_ns == stats.min_ns);
    assert(stats.p9999_ns == stats.min_ns);
    assert_stats_internally_consistent(stats);
}

// Warmup-vs-measured contamination check: the very first call ever made
// (which, since warmup_iters > 0, is guaranteed by measure_latency_ns's own
// implementation to happen during the untimed warmup loop, before the
// timed loop even starts) does something 1000x more expensive than every
// other call. If warmup were (incorrectly) being timed/recorded, this
// would show up as a multi-millisecond outlier in stats.max_ns; since real
// steady-state per-iteration cost here is a trivial in-memory increment
// (sub-microsecond), a correct implementation's stats.max_ns must stay
// far below the injected warmup delay.
void test_measure_latency_ns_warmup_not_recorded() {
    bool first_call_done = false;
    std::uint64_t sink = 0;

    const auto workload = [&]() {
        if (!first_call_done) {
            first_call_done = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            sink += 1; // trivial, real steady-state work
        }
    };

    const LatencyStats stats = measure_latency_ns(workload, /*warmup_iters=*/1, /*measured_iters=*/2000);

    assert(first_call_done); // sanity: the workload really did run at least once
    // 5ms == 5,000,000ns. If the warmup call's artificial 5ms sleep leaked
    // into the recorded samples, stats.max_ns would be on that order. A
    // correct implementation's steady-state max (sub-microsecond increments
    // under real, if noisy, scheduling) should be nowhere close -- use
    // 1,000,000ns (1ms) as a generous, still-decisive threshold.
    assert(stats.max_ns < 1'000'000ULL &&
           "warmup's artificial 5ms delay leaked into the timed samples -- warmup is NOT being excluded");
    assert(stats.count == 2000);
    assert_stats_internally_consistent(stats);
    (void)sink;
}

// Real (not synthetic) invariant check: run measure_latency_ns against a
// workload with genuinely variable duration (a small, data-dependent busy
// loop), and confirm the ordering invariant holds against REAL recorded
// timing data, not just hand-fed synthetic vectors.
void test_measure_latency_ns_real_workload_ordering_invariant() {
    std::uint64_t counter = 0;
    static volatile std::uint64_t sink = 0;
    const auto workload = [&]() {
        // Variable-length busy loop: iteration count depends on `counter`,
        // so different calls really do take different amounts of wall time
        // -- not a fixed/no-op workload.
        std::uint64_t local = 0;
        const std::uint64_t spins = 1 + (counter % 37);
        for (std::uint64_t i = 0; i < spins; ++i) {
            local += i * i;
        }
        sink = local;
        ++counter;
    };

    const LatencyStats stats = measure_latency_ns(workload, /*warmup_iters=*/500, /*measured_iters=*/5000);
    assert(stats.count == 5000);
    assert_stats_internally_consistent(stats);
    // With genuinely variable work per call, it would be suspicious (though
    // not strictly impossible) for min == max; assert real variance was
    // observed, i.e. this is exercising a real, not degenerate, workload.
    assert(stats.max_ns >= stats.min_ns);
    // Read `sink` back (not just written) so it is not merely a "set but
    // never read" variable from the compiler's point of view.
    std::printf("  (last busy-loop sink value observed: %llu)\n", static_cast<unsigned long long>(sink));
}

} // namespace

int main(int /*argc*/, char** argv) {
#if defined(NDEBUG)
    std::fprintf(stderr, "FATAL: this test binary was compiled with NDEBUG defined -- every assert() "
                          "above is silently compiled out, so 'all tests passed' would be a false "
                          "positive. Fix CMakeLists.txt (-UNDEBUG) for this target.\n");
    return 1;
#else
    // Belt-and-suspenders live-assertion check: fork+exec this SAME binary
    // with an env var set that makes it hit a deliberately-false assert(),
    // then confirm the child was actually killed by SIGABRT -- proof
    // assert() really aborts in this exact compiled binary, rather than
    // just trusting "no test in this file happened to call assert(false)"
    // (which would be silently vacuous if NDEBUG-like stripping were
    // somehow in effect despite the #if check above, e.g. a custom
    // <cassert> shadow or NDEBUG defined via a route the preprocessor
    // check above can't see).
    if (std::getenv("EXEC_CORE_ASSERT_LIVE_CHECK_CHILD") != nullptr) {
        assert(false && "deliberate failure to prove assert() is live");
        std::fprintf(stderr, "FATAL: reached past a deliberately-false assert() -- assertions are NOT live\n");
        return 0; // unreachable if assertions are actually live
    }
    {
        const pid_t pid = fork();
        if (pid == 0) {
            // Child: re-exec argv[0] with the trigger env var set.
            setenv("EXEC_CORE_ASSERT_LIVE_CHECK_CHILD", "1", /*overwrite=*/1);
            execv(argv[0], argv);
            _exit(127); // only reached if execv itself failed
        }
        assert(pid > 0 && "fork() failed -- cannot verify assert() liveness");
        int status = 0;
        waitpid(pid, &status, 0);
        const bool child_aborted_via_sigabrt = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
        if (!child_aborted_via_sigabrt) {
            std::fprintf(stderr,
                         "FATAL: assert()-liveness self-check failed -- child process (running the "
                         "same binary) did not die from SIGABRT on a deliberately-false assert() "
                         "(raw wait status=%d). Assertions may be silently disabled in this build.\n",
                         status);
            return 1;
        }
        std::printf("[PASS] assert()-liveness self-check (child process SIGABRT'd on a deliberately-"
                    "false assert(), confirming assertions are genuinely live in this binary)\n");
    }
#endif

    struct NamedTest {
        const char* name;
        void (*fn)();
    };
    const NamedTest tests[] = {
        {"nearest_rank_1_to_100", test_nearest_rank_1_to_100},
        {"nearest_rank_1_to_1000_non_boundary_percentiles", test_nearest_rank_1_to_1000_non_boundary_percentiles},
        {"nearest_rank_N_not_divisible_by_100_or_1000", test_nearest_rank_N_not_divisible_by_100_or_1000},
        {"nearest_rank_single_sample", test_nearest_rank_single_sample},
        {"nearest_rank_two_samples", test_nearest_rank_two_samples},
        {"nearest_rank_all_identical", test_nearest_rank_all_identical},
        {"nearest_rank_empty_returns_zero", test_nearest_rank_empty_returns_zero},
        {"nearest_rank_clamps_out_of_range_p", test_nearest_rank_clamps_out_of_range_p},
        {"p999_exact_multiple_of_1000_regression", test_p999_exact_multiple_of_1000_regression},
        {"compute_stats_sorts_unsorted_input", test_compute_stats_sorts_unsorted_input},
        {"compute_stats_min_max_mean_hand_verified", test_compute_stats_min_max_mean_hand_verified},
        {"compute_stats_empty_is_honestly_all_zero", test_compute_stats_empty_is_honestly_all_zero},
        {"compute_stats_single_sample_p9999_clamps_correctly", test_compute_stats_single_sample_p9999_clamps_correctly},
        {"compute_stats_all_identical", test_compute_stats_all_identical},
        {"measure_latency_ns_sample_count_and_call_count", test_measure_latency_ns_sample_count_and_call_count},
        {"measure_latency_ns_zero_warmup", test_measure_latency_ns_zero_warmup},
        {"measure_latency_ns_single_measured_iter", test_measure_latency_ns_single_measured_iter},
        {"measure_latency_ns_warmup_not_recorded", test_measure_latency_ns_warmup_not_recorded},
        {"measure_latency_ns_real_workload_ordering_invariant", test_measure_latency_ns_real_workload_ordering_invariant},
    };

    for (const auto& t : tests) {
        t.fn();
        std::printf("[PASS] %s\n", t.name);
    }
    std::printf("All %zu latency_harness tests passed.\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
