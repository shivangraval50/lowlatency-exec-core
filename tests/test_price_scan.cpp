// Real correctness tests for phase-5's SIMD price-scan primitive
// (include/exec_core/price_scan.hpp, src/price_scan.cpp).
//
// Plain assert()-based tests (no framework), consistent with the rest of
// tests/. See price_scan.hpp's file header for the full semantics writeup;
// short version: scan_quantity_at_or_better(levels, n, limit_price, side)
// sums quantity/count over levels crossable by a hypothetical incoming
// order of `side` --
//   Side::Buy  (levels are resting ASKs): crossable when price <= limit
//   Side::Sell (levels are resting BIDs): crossable when price >= limit
// i.e. the limit price itself IS included (at-or-better, not strictly
// better) on both sides.
//
// Honest scope of THIS file, given this is an ARM (Apple Silicon M2) dev
// machine:
//   - scan_quantity_at_or_better_scalar is always compiled/run -- source of
//     truth.
//   - scan_quantity_at_or_better_neon is compiled AND run here (NEON is
//     baseline on AArch64) -- cross-checked against scalar below across a
//     large battery of fixed edge cases plus randomized property tests.
//   - scan_quantity_at_or_better (the public dispatcher) is what every test
//     below actually calls for the "does a real caller get the right
//     answer" checks -- it resolves to _neon on this machine (verified by
//     the guard-check test at the bottom, which confirms __AVX2__ is NOT
//     defined here so the AVX2 branch is genuinely excluded, not silently
//     miscompiled/mistested).
//   - scan_quantity_at_or_better_avx2 is NOT called anywhere in this file.
//     This machine has no x86 hardware; AVX2 correctness is verified on CI
//     (ubuntu-latest) via tools/avx2_smoke_check.cpp, not here. Do not
//     mistake the guard-check test below for an AVX2 correctness test --
//     it only checks the macro is undefined, not that any AVX2 code ran.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "exec_core/price_scan.hpp"

using namespace exec_core;

namespace {

// ---------------------------------------------------------------------
// Cross-check helper: scalar vs the public dispatcher must always agree.
//
// On ARM (__ARM_NEON/__aarch64__) we additionally call the internal
// `_neon` entry point directly and cross-check it too -- this is the
// stronger "neon-vs-scalar-vs-dispatch triple check" that exercises the
// NEON implementation on its own, not just via the dispatcher. That
// internal function is declared only under the same ARM guard in
// price_scan.hpp, so it must not be referenced on other targets (e.g.
// x86_64 CI) -- see price_scan.hpp's `#if defined(__ARM_NEON) ||
// defined(__aarch64__)` guard around `scan_quantity_at_or_better_neon`.
// ---------------------------------------------------------------------
void assert_all_agree(const std::vector<PriceLevel>& levels, Price limit_price, Side side,
                       const char* case_name) {
    const ScanResult scalar =
        scan_quantity_at_or_better_scalar(levels.data(), levels.size(), limit_price, side);
    const ScanResult dispatch =
        scan_quantity_at_or_better(levels.data(), levels.size(), limit_price, side);

#if defined(__ARM_NEON) || defined(__aarch64__)
    const ScanResult neon =
        scan_quantity_at_or_better_neon(levels.data(), levels.size(), limit_price, side);

    if (!(scalar == neon) || !(scalar == dispatch)) {
        std::fprintf(stderr,
                      "MISMATCH in case '%s' (n=%zu, limit=%lld, side=%s): "
                      "scalar={%lld,%zu} neon={%lld,%zu} dispatch={%lld,%zu}\n",
                      case_name, levels.size(), static_cast<long long>(limit_price),
                      side == Side::Buy ? "Buy" : "Sell",
                      static_cast<long long>(scalar.total_quantity), scalar.count,
                      static_cast<long long>(neon.total_quantity), neon.count,
                      static_cast<long long>(dispatch.total_quantity), dispatch.count);
    }
    assert(scalar == neon);
    assert(scalar == dispatch);
#else
    if (!(scalar == dispatch)) {
        std::fprintf(stderr,
                      "MISMATCH in case '%s' (n=%zu, limit=%lld, side=%s): "
                      "scalar={%lld,%zu} dispatch={%lld,%zu}\n",
                      case_name, levels.size(), static_cast<long long>(limit_price),
                      side == Side::Buy ? "Buy" : "Sell",
                      static_cast<long long>(scalar.total_quantity), scalar.count,
                      static_cast<long long>(dispatch.total_quantity), dispatch.count);
    }
    assert(scalar == dispatch);
#endif
}

// ---------------------------------------------------------------------
// Fixed edge cases via the public dispatcher, with hand-verified expected
// values -- this is the "does the contract actually hold" check, not just
// "do scalar and NEON agree with each other" (two buggy-in-the-same-way
// implementations could still agree).
// ---------------------------------------------------------------------
void test_empty_array() {
    std::vector<PriceLevel> levels; // n == 0
    ScanResult r = scan_quantity_at_or_better(levels.data(), levels.size(), 100, Side::Buy);
    assert(r.total_quantity == 0);
    assert(r.count == 0);
    assert_all_agree(levels, 100, Side::Buy, "empty_array_buy");
    assert_all_agree(levels, 100, Side::Sell, "empty_array_sell");
}

void test_single_element_boundary_match() {
    // Side::Buy: level.price <= limit_price is a match. price == limit ->
    // match (at-or-better includes the boundary).
    {
        std::vector<PriceLevel> levels = {{100, 7}};
        ScanResult r = scan_quantity_at_or_better(levels.data(), 1, 100, Side::Buy);
        assert(r.total_quantity == 7);
        assert(r.count == 1);
        assert_all_agree(levels, 100, Side::Buy, "single_boundary_match_buy");
    }
    // Side::Sell: level.price >= limit_price is a match. price == limit ->
    // match.
    {
        std::vector<PriceLevel> levels = {{100, 7}};
        ScanResult r = scan_quantity_at_or_better(levels.data(), 1, 100, Side::Sell);
        assert(r.total_quantity == 7);
        assert(r.count == 1);
        assert_all_agree(levels, 100, Side::Sell, "single_boundary_match_sell");
    }
}

void test_single_element_boundary_no_match() {
    // Buy: price one tick above limit must NOT match (100 > 99 is false for
    // <=).
    {
        std::vector<PriceLevel> levels = {{100, 7}};
        ScanResult r = scan_quantity_at_or_better(levels.data(), 1, 99, Side::Buy);
        assert(r.total_quantity == 0);
        assert(r.count == 0);
        assert_all_agree(levels, 99, Side::Buy, "single_boundary_no_match_buy");
    }
    // Sell: price one tick below limit must NOT match (100 >= 101 is false).
    {
        std::vector<PriceLevel> levels = {{100, 7}};
        ScanResult r = scan_quantity_at_or_better(levels.data(), 1, 101, Side::Sell);
        assert(r.total_quantity == 0);
        assert(r.count == 0);
        assert_all_agree(levels, 101, Side::Sell, "single_boundary_no_match_sell");
    }
}

void test_odd_length_hits_scalar_tail_path() {
    // NEON processes 2 int64 PriceLevels per vector iteration; an odd n
    // forces exactly 1 leftover element through the scalar tail loop inside
    // _neon. Use n=3,5,7 to hit several different tail-remainder states.
    for (std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{7},
                           std::size_t{9}, std::size_t{15}}) {
        std::vector<PriceLevel> levels;
        for (std::size_t i = 0; i < n; ++i) {
            levels.push_back({static_cast<Price>(100 + static_cast<Price>(i)),
                               static_cast<Quantity>(i + 1)});
        }
        char name[64];
        std::snprintf(name, sizeof(name), "odd_length_n%zu_buy", n);
        assert_all_agree(levels, 105, Side::Buy, name);
        std::snprintf(name, sizeof(name), "odd_length_n%zu_sell", n);
        assert_all_agree(levels, 105, Side::Sell, name);

        // Also hand-check the buy count/sum against a manual scalar
        // computation for at least one n, to make sure "agreement" isn't
        // hiding a shared bug.
        if (n == 5) {
            ScanResult r =
                scan_quantity_at_or_better(levels.data(), levels.size(), 102, Side::Buy);
            // levels: (100,1)(101,2)(102,3)(103,4)(104,5); price<=102 ->
            // 100,101,102 -> qty 1+2+3=6, count=3
            assert(r.total_quantity == 6);
            assert(r.count == 3);
        }
    }
}

void test_all_elements_match() {
    std::vector<PriceLevel> levels;
    for (int i = 0; i < 20; ++i) {
        levels.push_back({static_cast<Price>(i), static_cast<Quantity>(i + 1)});
    }
    // Buy with a limit far above every price: all 20 should match.
    ScanResult r = scan_quantity_at_or_better(levels.data(), levels.size(), 10000, Side::Buy);
    Quantity expected_sum = 0;
    for (int i = 0; i < 20; ++i) expected_sum += (i + 1);
    assert(r.count == 20);
    assert(r.total_quantity == expected_sum);
    assert_all_agree(levels, 10000, Side::Buy, "all_match_buy");

    // Sell with a limit far below every price: all 20 should match too.
    ScanResult r2 = scan_quantity_at_or_better(levels.data(), levels.size(), -10000, Side::Sell);
    assert(r2.count == 20);
    assert(r2.total_quantity == expected_sum);
    assert_all_agree(levels, -10000, Side::Sell, "all_match_sell");
}

void test_no_elements_match() {
    std::vector<PriceLevel> levels;
    for (int i = 0; i < 20; ++i) {
        levels.push_back({static_cast<Price>(100 + i), static_cast<Quantity>(i + 1)});
    }
    // Buy with a limit far below every price: nothing should match.
    ScanResult r = scan_quantity_at_or_better(levels.data(), levels.size(), -10000, Side::Buy);
    assert(r.count == 0);
    assert(r.total_quantity == 0);
    assert_all_agree(levels, -10000, Side::Buy, "none_match_buy");

    // Sell with a limit far above every price: nothing should match.
    ScanResult r2 = scan_quantity_at_or_better(levels.data(), levels.size(), 10000, Side::Sell);
    assert(r2.count == 0);
    assert(r2.total_quantity == 0);
    assert_all_agree(levels, 10000, Side::Sell, "none_match_sell");
}

void test_exact_boundary_price_mixed() {
    // A depth with several levels exactly AT the limit price, mixed with
    // levels strictly better/worse -- checks that ties at the boundary are
    // consistently included (at-or-better, not strictly-better) across all
    // implementations, not just isolated single-element cases.
    std::vector<PriceLevel> levels = {
        {95, 10},  // strictly better for Buy (< limit)
        {100, 20}, // exactly at limit -- must count
        {100, 30}, // exactly at limit, duplicate price -- must count
        {105, 40}, // strictly worse for Buy (> limit)
    };
    ScanResult r = scan_quantity_at_or_better(levels.data(), levels.size(), 100, Side::Buy);
    assert(r.count == 3);              // 95, 100, 100
    assert(r.total_quantity == 60);    // 10+20+30
    assert_all_agree(levels, 100, Side::Buy, "boundary_mixed_buy");

    ScanResult r2 = scan_quantity_at_or_better(levels.data(), levels.size(), 100, Side::Sell);
    assert(r2.count == 3);             // 100, 100, 105
    assert(r2.total_quantity == 90);   // 20+30+40
    assert_all_agree(levels, 100, Side::Sell, "boundary_mixed_sell");
}

void test_realistic_depth_mixed() {
    // A realistic L2-style depth (best price first), 32 levels, one tick
    // apart, queried at a limit partway through the book.
    std::vector<PriceLevel> asks;
    for (int i = 0; i < 32; ++i) {
        asks.push_back({static_cast<Price>(1000 + i), static_cast<Quantity>(100 + 5 * i)});
    }
    ScanResult r = scan_quantity_at_or_better(asks.data(), asks.size(), 1015, Side::Buy);
    // Levels 0..15 (prices 1000..1015) match: 16 levels.
    Quantity expected = 0;
    for (int i = 0; i <= 15; ++i) expected += (100 + 5 * i);
    assert(r.count == 16);
    assert(r.total_quantity == expected);
    assert_all_agree(asks, 1015, Side::Buy, "depth32_buy_mid");

    std::vector<PriceLevel> bids;
    for (int i = 0; i < 32; ++i) {
        bids.push_back({static_cast<Price>(1000 - i), static_cast<Quantity>(100 + 5 * i)});
    }
    ScanResult r2 = scan_quantity_at_or_better(bids.data(), bids.size(), 985, Side::Sell);
    // Levels 0..15 (prices 1000..985) match: 16 levels.
    assert(r2.count == 16);
    assert(r2.total_quantity == expected);
    assert_all_agree(bids, 985, Side::Sell, "depth32_sell_mid");
}

void test_negative_prices_and_quantities() {
    // Price/Quantity are both std::int64_t (see types.hpp) -- signed, so
    // negative values are representable and must be handled correctly by
    // NEON's signed-compare intrinsics (vcleq_s64/vcgeq_s64), not just
    // treated as huge unsigned values.
    std::vector<PriceLevel> levels = {
        {-50, 100}, {-10, -5} /* negative quantity, e.g. a correction/adjustment */,
        {0, 3}, {10, 4}, {50, -1},
    };
    ScanResult r = scan_quantity_at_or_better(levels.data(), levels.size(), -10, Side::Buy);
    // price <= -10: -50 (100), -10 (-5) -> count=2, sum=95
    assert(r.count == 2);
    assert(r.total_quantity == 95);
    assert_all_agree(levels, -10, Side::Buy, "negative_buy");

    ScanResult r2 = scan_quantity_at_or_better(levels.data(), levels.size(), -10, Side::Sell);
    // price >= -10: -10(-5), 0(3), 10(4), 50(-1) -> count=4, sum=1
    assert(r2.count == 4);
    assert(r2.total_quantity == 1);
    assert_all_agree(levels, -10, Side::Sell, "negative_sell");
}

void test_large_arrays_vector_width_remainder() {
    // Sweep sizes around and past several multiples of the NEON vector
    // width (2 int64 lanes) up into the hundreds/thousands, to catch any
    // remainder-handling bug that only manifests once the vectorized loop
    // runs many iterations (e.g. an off-by-one in vec_end computation, or a
    // reduction bug that only shows up with enough accumulation).
    std::mt19937_64 rng(0xC0FFEEu);
    std::uniform_int_distribution<Price> price_dist(-1000, 1000);
    std::uniform_int_distribution<Quantity> qty_dist(-500, 500);

    for (std::size_t n : {std::size_t{0},   std::size_t{1},   std::size_t{2},
                           std::size_t{63},  std::size_t{64},  std::size_t{65},
                           std::size_t{127}, std::size_t{128}, std::size_t{129},
                           std::size_t{500}, std::size_t{999}, std::size_t{1000},
                           std::size_t{2001}}) {
        std::vector<PriceLevel> levels;
        levels.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            levels.push_back({price_dist(rng), qty_dist(rng)});
        }
        char name[64];
        std::snprintf(name, sizeof(name), "large_n%zu_buy", n);
        assert_all_agree(levels, 0, Side::Buy, name);
        std::snprintf(name, sizeof(name), "large_n%zu_sell", n);
        assert_all_agree(levels, 0, Side::Sell, name);
    }
}

// ---------------------------------------------------------------------
// Randomized property-based cross-check: for many random array lengths,
// random level contents, and random limit prices, scalar and NEON (and the
// public dispatcher) must agree exactly. This is the strongest single
// check in this file for a masking/reduction bug that fixed cases could
// miss (e.g. a bug that only appears for specific bit patterns or a
// specific alignment of match/no-match runs relative to vector lanes).
// ---------------------------------------------------------------------
void test_randomized_property_cross_check() {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<std::size_t> len_dist(0, 300);
    std::uniform_int_distribution<Price> price_dist(-10'000, 10'000);
    std::uniform_int_distribution<Quantity> qty_dist(-1'000'000, 1'000'000);
    std::uniform_int_distribution<int> side_dist(0, 1);

    constexpr int kTrials = 500;
    for (int trial = 0; trial < kTrials; ++trial) {
        const std::size_t n = len_dist(rng);
        std::vector<PriceLevel> levels;
        levels.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            levels.push_back({price_dist(rng), qty_dist(rng)});
        }
        const Price limit = price_dist(rng);
        const Side side = side_dist(rng) == 0 ? Side::Buy : Side::Sell;

        char name[32];
        std::snprintf(name, sizeof(name), "random_trial_%d", trial);
        assert_all_agree(levels, limit, side, name);
    }
}

// ---------------------------------------------------------------------
// Guard-check: confirm this build genuinely does NOT define __AVX2__ (so
// the AVX2 branch of the dispatcher is compiled out, not silently
// mis-selected on this ARM machine) and DOES define the NEON guard macros,
// so the public dispatcher is actually resolving to _neon here, not
// silently falling back to _scalar (which would make every "cross-check"
// above vacuous -- scalar vs scalar always agrees).
// ---------------------------------------------------------------------
void test_avx2_guarded_out_on_this_arm_machine() {
#if defined(__AVX2__)
    assert(false && "AVX2 must not be defined on this ARM (Apple Silicon) machine -- "
                     "if this fires, the dispatcher would silently be selecting/compiling "
                     "the AVX2 path here, which has never been runtime-verified locally.");
#endif
#if !(defined(__ARM_NEON) || defined(__aarch64__))
    assert(false && "Expected NEON guards to be active on this machine; if not, the "
                     "public dispatcher used by every test above is silently exercising "
                     "the scalar fallback instead of NEON, making the cross-checks vacuous.");
#endif
    // If we get here, __AVX2__ is undefined and NEON guards are active --
    // exactly the expected configuration for this ARM dev machine.
    std::printf("  (guard check: __AVX2__ undefined, NEON guards active -- as expected on ARM)\n");
}

} // namespace

int main() {
    struct NamedTest {
        const char* name;
        void (*fn)();
    };
    const NamedTest tests[] = {
        {"empty_array", test_empty_array},
        {"single_element_boundary_match", test_single_element_boundary_match},
        {"single_element_boundary_no_match", test_single_element_boundary_no_match},
        {"odd_length_hits_scalar_tail_path", test_odd_length_hits_scalar_tail_path},
        {"all_elements_match", test_all_elements_match},
        {"no_elements_match", test_no_elements_match},
        {"exact_boundary_price_mixed", test_exact_boundary_price_mixed},
        {"realistic_depth_mixed", test_realistic_depth_mixed},
        {"negative_prices_and_quantities", test_negative_prices_and_quantities},
        {"large_arrays_vector_width_remainder", test_large_arrays_vector_width_remainder},
        {"randomized_property_cross_check", test_randomized_property_cross_check},
        {"avx2_guarded_out_on_this_arm_machine", test_avx2_guarded_out_on_this_arm_machine},
    };

    for (const auto& t : tests) {
        t.fn();
        std::printf("[PASS] %s\n", t.name);
    }
    std::printf("All %zu price_scan tests passed.\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
