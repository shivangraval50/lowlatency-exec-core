// Interim, CI-only smoke check for phase 5's AVX2 price-scan path
// (include/exec_core/price_scan.hpp, src/price_scan.cpp).
//
// This is NOT the authoritative test suite (that's
// tests/test_price_scan.cpp, added by the tester phase, wired into the
// normal CMake/ctest build). It exists because the AVX2 implementation was
// written by hand on an ARM machine with no x86 hardware to run it on --
// see price_scan.hpp's file header. This file is compiled with -mavx2 and
// *executed* directly by .github/workflows/ci.yml, on ubuntu-latest (real
// x86_64 hardware with AVX2), so that this phase's AVX2 path gets a real
// run-and-compare-against-scalar check on real hardware as soon as this
// change reaches CI, rather than waiting on a separate PR.
//
// Mirrors the same edge cases the author manually ran locally against the
// NEON path (empty, single-element boundary/no-match, odd-length tail,
// all-match, none-match, a 16-level depth scan, and negative
// prices/quantities) -- same inputs, just checked against
// scan_quantity_at_or_better_avx2 instead of _neon.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "exec_core/price_scan.hpp"

#if !defined(__AVX2__)
#error "avx2_smoke_check.cpp must be compiled with -mavx2 (see ci.yml)"
#endif

using namespace exec_core;

namespace {
bool g_all_ok = true;

void check(const char* name, const std::vector<PriceLevel>& levels, Price limit, Side side) {
    ScanResult scalar =
        scan_quantity_at_or_better_scalar(levels.data(), levels.size(), limit, side);
    ScanResult avx2 = scan_quantity_at_or_better_avx2(levels.data(), levels.size(), limit, side);
    ScanResult dispatch =
        scan_quantity_at_or_better(levels.data(), levels.size(), limit, side);
    bool ok = (scalar == avx2) && (scalar == dispatch);
    std::printf("%-28s scalar={%lld,%zu} avx2={%lld,%zu} dispatch={%lld,%zu} %s\n", name,
                (long long)scalar.total_quantity, scalar.count, (long long)avx2.total_quantity,
                avx2.count, (long long)dispatch.total_quantity, dispatch.count,
                ok ? "OK" : "MISMATCH");
    if (!ok) {
        g_all_ok = false;
    }
}
} // namespace

int main() {
    check("empty", {}, 100, Side::Buy);
    check("single_boundary_buy", {{100, 5}}, 100, Side::Buy);
    check("single_boundary_sell", {{100, 5}}, 100, Side::Sell);
    check("single_no_match_buy", {{150, 5}}, 100, Side::Buy);
    check("single_no_match_sell", {{50, 5}}, 100, Side::Sell);
    // Non-multiple-of-4 length (3 elements: one AVX2 quad partially filled,
    // exercises the scalar tail loop in the AVX2 implementation).
    check("odd_length_buy", {{90, 1}, {95, 2}, {200, 3}}, 100, Side::Buy);
    check("all_match_buy", {{10, 1}, {20, 2}, {30, 3}, {40, 4}}, 1000, Side::Buy);
    check("none_match_sell", {{10, 1}, {20, 2}, {30, 3}, {40, 4}}, 1000, Side::Sell);

    {
        std::vector<PriceLevel> asks;
        for (int i = 0; i < 16; ++i) {
            asks.push_back({100 + i, 10 + i});
        }
        check("depth16_buy_limit_105", asks, 105, Side::Buy);
        check("depth16_buy_limit_below_all", asks, 50, Side::Buy);
        check("depth16_buy_limit_above_all", asks, 999, Side::Buy);
    }

    check("negative_prices_buy", {{-50, 1}, {-10, 2}, {0, 3}, {10, 4}}, -10, Side::Buy);

    if (!g_all_ok) {
        std::fprintf(stderr, "AVX2 SMOKE CHECK FAILED -- see MISMATCH lines above\n");
        return 1;
    }
    std::printf("ALL AVX2 VS SCALAR SMOKE CHECKS PASSED (ubuntu-latest, real x86_64 AVX2)\n");
    return 0;
}
