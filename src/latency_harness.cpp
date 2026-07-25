#include "exec_core/latency_harness.hpp"

#include <cstdio>

namespace exec_core {

std::uint64_t nearest_rank_percentile(const std::vector<std::uint64_t>& sorted_ascending, double p) {
    if (sorted_ascending.empty()) {
        return 0;
    }
    const double clamped_p = std::clamp(p, 0.0, 100.0);

    // Nearest-rank method: 1-based rank = ceil(p/100 * N), clamped to
    // [1, N] (see latency_harness.hpp's file header for why this method,
    // not linear interpolation, was chosen). The rank arithmetic itself is
    // done in EXACT INTEGER arithmetic below, not floating point -- see
    // latency_harness.hpp's file header for the full writeup of the bug
    // this fixes (the old `std::ceil((clamped_p / 100.0) * N)` computed a
    // wrong-by-one rank whenever N was an exact multiple of 1000 and
    // p == 99.9, because 99.9 is not exactly representable as a double).
    //
    // Step 1: convert the double percentage to an exact integer scaled by
    // 100 ("hundredths of a percent" -- e.g. 99.9 -> 9990, 99.99 -> 9999).
    // Every percentile this project ever calls with (50.0, 90.0, 99.0,
    // 99.9, 99.99, and the clamped 0.0/100.0 boundaries) has at most 2
    // digits after the decimal point, so this scaling is exact in the
    // reals; the only question is how to get from the `double` back to
    // that exact integer, and the answer MUST be round-to-nearest
    // (std::llround), not truncation and not ceiling:
    //   - Truncation (static_cast<uint64_t>(clamped_p * 100.0)) fails
    //     whenever the nearest representable double for a literal like
    //     99.9 happens to sit a hair BELOW the true decimal value: then
    //     clamped_p * 100.0 can compute to e.g. 9989.999999999998, and
    //     truncating silently drops to 9989 instead of 9990.
    //   - Ceiling has the mirror-image failure: it rounds up even in cases
    //     where the double's tiny representation error already landed a
    //     hair ABOVE the intended value, turning an exact case into a
    //     spurious off-by-one in the other direction.
    //   - Rounding to nearest is correct because the only error a double
    //     literal with <=2 decimal digits can introduce here is a
    //     representation error on the order of 1e-13 relative magnitude
    //     (IEEE-754 double carries ~15-17 significant decimal digits) --
    //     many orders of magnitude smaller than the 0.5 rounding margin --
    //     so round-to-nearest always recovers the exact intended integer
    //     regardless of which direction the tiny fp error fell. This is
    //     the same exact-integer representation the test oracle
    //     (`exact_rank()` in tests/test_latency_harness.cpp) uses, just
    //     derived here from the runtime `double p` argument instead of a
    //     hand-picked literal integer.
    const std::int64_t p_scaled_1e2 = std::llround(clamped_p * 100.0);

    const std::uint64_t n = sorted_ascending.size();

    // Step 2: rank = ceil(p_scaled_1e2 * n / 10000) via pure integer
    // ceiling-division: ceil(a / b) == (a + b - 1) / b for non-negative
    // integers -- no floating point involved anywhere in this step.
    //
    // The multiply is done in `unsigned __int128` rather than
    // `std::uint64_t` as a deliberate, free safety margin: p_scaled_1e2 is
    // at most 10000 (100.00%) and this project's real benchmark uses at
    // most 50,000 samples, so the product would comfortably fit in a
    // uint64_t even at, say, 10 million samples (10000 * 10,000,000 =
    // 1e11, versus uint64_t's max of ~1.8e19) -- __int128 is not strictly
    // required at any sample count this project will plausibly ever run,
    // but costs nothing here (this runs once per percentile during
    // reduction, not in the measured hot loop), so it is used anyway to
    // remove any need to reason about overflow at all, including for
    // hypothetical future callers with much larger N.
    const unsigned __int128 numerator =
        static_cast<unsigned __int128>(p_scaled_1e2) * static_cast<unsigned __int128>(n);
    const unsigned __int128 denominator = 10000;
    const unsigned __int128 rank128 = (numerator + denominator - 1) / denominator;

    std::uint64_t rank = rank128 < 1 ? 1 : static_cast<std::uint64_t>(rank128);
    if (rank > n) {
        rank = n;
    }
    return sorted_ascending[rank - 1];
}

LatencyStats compute_latency_stats(std::vector<std::uint64_t> samples_ns) {
    LatencyStats stats;
    stats.count = samples_ns.size();
    if (samples_ns.empty()) {
        return stats; // all-zero LatencyStats -- honestly reflects "no data", not a fabricated number
    }

    std::sort(samples_ns.begin(), samples_ns.end());

    stats.min_ns = samples_ns.front();
    stats.max_ns = samples_ns.back();

    // Sum as double to avoid overflow-by-construction concerns with very
    // large sample counts * large per-sample ns values, at the cost of
    // losing a little precision versus a 128-bit integer sum -- an
    // acceptable tradeoff for a "mean" that is reported alongside exact
    // percentiles anyway.
    double sum = 0.0;
    for (std::uint64_t v : samples_ns) {
        sum += static_cast<double>(v);
    }
    stats.mean_ns = sum / static_cast<double>(samples_ns.size());

    stats.p50_ns = nearest_rank_percentile(samples_ns, 50.0);
    stats.p90_ns = nearest_rank_percentile(samples_ns, 90.0);
    stats.p99_ns = nearest_rank_percentile(samples_ns, 99.0);
    stats.p999_ns = nearest_rank_percentile(samples_ns, 99.9);
    stats.p9999_ns = nearest_rank_percentile(samples_ns, 99.99);

    return stats;
}

void print_latency_row(const std::string& name, std::size_t warmup_iters, const LatencyStats& stats) {
    std::printf("%-42s n=%-8zu warmup=%-6zu min=%-8llu p50=%-8llu p90=%-8llu p99=%-8llu p99.9=%-8llu "
                "p99.99=%-8llu max=%-10llu mean=%.1f (ns)\n",
                name.c_str(), stats.count, warmup_iters, static_cast<unsigned long long>(stats.min_ns),
                static_cast<unsigned long long>(stats.p50_ns), static_cast<unsigned long long>(stats.p90_ns),
                static_cast<unsigned long long>(stats.p99_ns), static_cast<unsigned long long>(stats.p999_ns),
                static_cast<unsigned long long>(stats.p9999_ns), static_cast<unsigned long long>(stats.max_ns),
                stats.mean_ns);
}

} // namespace exec_core
