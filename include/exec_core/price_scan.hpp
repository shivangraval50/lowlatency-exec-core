// SIMD price-scan primitive: "how much resting quantity is available at or
// better than a given limit price, on the side of the book an incoming order
// would cross?"
//
// ---------------------------------------------------------------------------
// Why this operation, and why it's a good SIMD candidate
// ---------------------------------------------------------------------------
// This is exactly the question a marketable limit order (or a slippage/
// impact estimator) needs answered before it touches the matching logic:
// "if I sent a buy at price P right now, how much of the resting ask side
// could it immediately fill against, and how deep would it have to walk?"
// Given a flat, contiguous snapshot of one side's price levels (best price
// first, as a real L2 book snapshot would be laid out), the answer is a
// linear scan: compare each level's price against a broadcast threshold,
// and reduce (sum + count) over the levels that satisfy it. That
// compare-mask-reduce shape is the textbook case for data-parallel SIMD --
// no data-dependent branching, no early-exit dependency between lanes.
//
// ---------------------------------------------------------------------------
// Honest scope: this is a standalone utility, NOT wired into OrderBook yet
// ---------------------------------------------------------------------------
// `OrderBook` (phase 1) still matches against `std::map<Price,
// std::list<RestingOrder>>` / an intrusive list carved from `SlabAllocator`
// (phase 3) -- that correctness-first design is deliberately untouched here.
// This header operates on a *separate*, deliberately simple flat array type
// (`PriceLevel[]`) that nothing in this codebase currently populates from a
// live `OrderBook`. It is a SIMD-accelerated scan primitive that could feed
// a future flat-array/SoA order-book variant (e.g. a periodic snapshot of
// top-N levels exported for a risk/impact calculation) -- not a claim that
// the live matching path has been rewritten to use it. Do not read this file
// as "the order book is now SIMD"; it isn't.
//
// ---------------------------------------------------------------------------
// Semantics
// ---------------------------------------------------------------------------
// `side` names the side of the *incoming, hypothetical order*, not the side
// of the resting `levels` array (the caller is expected to pass the resting
// levels from the opposite side of the book):
//   - Side::Buy  -- caller passes resting ASK levels. A level is "at or
//                   better" (crossable) when level.price <= limit_price.
//   - Side::Sell -- caller passes resting BID levels. A level is "at or
//                   better" (crossable) when level.price >= limit_price.
// The scan does NOT assume `levels` is sorted (it happens to be, in a real
// book snapshot, but all three implementations below scan every element and
// do not rely on sortedness or try to early-exit at a prefix boundary) --
// this keeps scalar/NEON/AVX2 behavior identical and simple to reason about,
// at the cost of not exploiting a sorted-prefix early-out. That's a
// deliberate simplicity-over-speed tradeoff for a first version; documented,
// not hidden.
//
// ---------------------------------------------------------------------------
// Three implementations, one dispatcher
// ---------------------------------------------------------------------------
// 1. `_scalar` -- portable C++, compiles/runs everywhere, and is the source
//    of truth every other implementation is checked against.
// 2. `_neon` -- ARM NEON (AArch64), guarded by `__ARM_NEON`/`__aarch64__`.
//    This is the path actually compiled *and run* on this project's local
//    dev machine (Apple Silicon M2) -- see tests/test_price_scan.cpp.
// 3. `_avx2` -- x86 AVX2, guarded by `__AVX2__`. Written by hand against the
//    AVX2 intrinsics reference; compiles under a `-target x86_64-apple-
//    darwin -mavx2` cross-compile *compile-check* on this machine (no
//    linux sysroot available locally to go further), but has NOT been run
//    or verified for correctness on real x86 hardware from this machine --
//    that verification happens on CI (ubuntu-latest, see
//    .github/workflows/ci.yml), which is real x86_64 hardware with AVX2.
//    Do not treat `_avx2` as verified until CI has actually run its test.
//
// `scan_quantity_at_or_better` is a compile-time dispatcher (chosen via the
// same `#ifdef`s as above) that picks NEON on ARM, AVX2 on x86 w/ AVX2
// enabled at compile time, and scalar otherwise -- so ordinary callers don't
// need to know which path they're getting.
#pragma once

#include <cstddef>

#include "exec_core/types.hpp"

namespace exec_core {

// One resting price level: aggregate quantity available at `price`. Plain
// AoS layout (not the live OrderBook's map/intrusive-list structure) --
// deliberately simple so a flat snapshot of it is trivial to build and scan.
struct PriceLevel {
    Price price = 0;
    Quantity quantity = 0;
};

struct ScanResult {
    Quantity total_quantity = 0; // sum of quantity over levels satisfying the threshold
    std::size_t count = 0;       // number of levels satisfying the threshold

    friend bool operator==(const ScanResult&, const ScanResult&) = default;
};

// Scalar reference implementation. Always available; used both as a real
// fallback (platforms without NEON/AVX2) and as the correctness oracle the
// NEON/AVX2 paths are checked against.
ScanResult scan_quantity_at_or_better_scalar(const PriceLevel* levels, std::size_t n,
                                              Price limit_price, Side side);

#if defined(__ARM_NEON) || defined(__aarch64__)
// ARM NEON implementation (AArch64). Compiled and run on this project's
// local machine (Apple Silicon M2) -- see tests/test_price_scan.cpp for the
// scalar-vs-NEON cross-check across edge cases.
ScanResult scan_quantity_at_or_better_neon(const PriceLevel* levels, std::size_t n,
                                            Price limit_price, Side side);
#endif

#if defined(__AVX2__)
// x86 AVX2 implementation. Correct per the Intel intrinsics reference to the
// best of this author's/environment's ability, but NOT runtime-verified on
// this machine (no x86 hardware here) -- see file header above and
// .github/workflows/ci.yml for where this actually gets exercised.
ScanResult scan_quantity_at_or_better_avx2(const PriceLevel* levels, std::size_t n,
                                            Price limit_price, Side side);
#endif

// Dispatches, at compile time, to whichever of the above the current
// translation unit was compiled for: NEON on ARM/AArch64, else AVX2 if
// `__AVX2__` is defined (x86 with AVX2 enabled), else the scalar reference.
ScanResult scan_quantity_at_or_better(const PriceLevel* levels, std::size_t n,
                                       Price limit_price, Side side);

} // namespace exec_core
