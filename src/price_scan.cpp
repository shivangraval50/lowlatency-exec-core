// See include/exec_core/price_scan.hpp for the full design writeup (what the
// scan does, why it's a good SIMD candidate, and the honesty notes on which
// implementation is actually verified where).
#include "exec_core/price_scan.hpp"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace exec_core {

// ---------------------------------------------------------------------------
// Scalar reference implementation
// ---------------------------------------------------------------------------
// Deliberately the simplest possible correct code: a plain linear scan, no
// early-exit on sortedness (see header comment). This is the oracle every
// other implementation must match bit-for-bit.
ScanResult scan_quantity_at_or_better_scalar(const PriceLevel* levels, std::size_t n,
                                              Price limit_price, Side side) {
    ScanResult result;
    if (side == Side::Buy) {
        // Caller passed resting ASK levels; crossable when price <= limit.
        for (std::size_t i = 0; i < n; ++i) {
            if (levels[i].price <= limit_price) {
                result.total_quantity += levels[i].quantity;
                ++result.count;
            }
        }
    } else {
        // Caller passed resting BID levels; crossable when price >= limit.
        for (std::size_t i = 0; i < n; ++i) {
            if (levels[i].price >= limit_price) {
                result.total_quantity += levels[i].quantity;
                ++result.count;
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// ARM NEON implementation (AArch64) -- compiled AND run locally (M2)
// ---------------------------------------------------------------------------
#if defined(__ARM_NEON) || defined(__aarch64__)
ScanResult scan_quantity_at_or_better_neon(const PriceLevel* levels, std::size_t n,
                                            Price limit_price, Side side) {
    static_assert(sizeof(PriceLevel) == 2 * sizeof(std::int64_t),
                  "NEON path assumes PriceLevel is exactly {int64 price; int64 quantity;} "
                  "with no padding, so vld2q_s64's stride-2 deinterleave lines up.");

    const int64x2_t limit_vec = vdupq_n_s64(limit_price);
    const int64x2_t ones_vec = vdupq_n_s64(1);
    int64x2_t sum_vec = vdupq_n_s64(0);
    int64x2_t count_vec = vdupq_n_s64(0);

    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 2); // largest multiple of 2 <= n
    for (; i < vec_end; i += 2) {
        // vld2q_s64 deinterleaves 4 consecutive int64s (2 PriceLevel structs)
        // into val[0] = {price[i], price[i+1]}, val[1] = {qty[i], qty[i+1]}
        // -- exactly PriceLevel's AoS layout, stride 2.
        const int64x2x2_t deint =
            vld2q_s64(reinterpret_cast<const std::int64_t*>(levels + i));
        const int64x2_t prices = deint.val[0];
        const int64x2_t quantities = deint.val[1];

        // uint64x2_t mask: all-ones lane where the predicate holds, else 0.
        const uint64x2_t mask_u =
            (side == Side::Buy) ? vcleq_s64(prices, limit_vec)   // price <= limit
                                 : vcgeq_s64(prices, limit_vec); // price >= limit
        const int64x2_t mask = vreinterpretq_s64_u64(mask_u);

        sum_vec = vaddq_s64(sum_vec, vandq_s64(mask, quantities));
        count_vec = vaddq_s64(count_vec, vandq_s64(mask, ones_vec));
    }

    Quantity total = vgetq_lane_s64(sum_vec, 0) + vgetq_lane_s64(sum_vec, 1);
    std::size_t count = static_cast<std::size_t>(vgetq_lane_s64(count_vec, 0) +
                                                  vgetq_lane_s64(count_vec, 1));

    // Scalar tail for the (at most 1) leftover element when n is odd.
    for (; i < n; ++i) {
        const bool hit = (side == Side::Buy) ? (levels[i].price <= limit_price)
                                              : (levels[i].price >= limit_price);
        if (hit) {
            total += levels[i].quantity;
            ++count;
        }
    }

    return ScanResult{total, count};
}
#endif // __ARM_NEON || __aarch64__

// ---------------------------------------------------------------------------
// x86 AVX2 implementation -- written against the AVX2 intrinsics reference,
// compile-checked locally with a `-target x86_64-apple-darwin -mavx2`
// cross-compile (this machine has no x86 hardware and no linux sysroot to go
// further), but NOT runtime-verified on this machine. See
// .github/workflows/ci.yml for the real x86_64 (ubuntu-latest) job that
// actually builds and runs tests/test_price_scan.cpp's AVX2 case.
// ---------------------------------------------------------------------------
#if defined(__AVX2__)
ScanResult scan_quantity_at_or_better_avx2(const PriceLevel* levels, std::size_t n,
                                            Price limit_price, Side side) {
    static_assert(sizeof(PriceLevel) == 2 * sizeof(std::int64_t),
                  "AVX2 path assumes PriceLevel is exactly {int64 price; int64 quantity;} "
                  "with no padding.");

    const __m256i limit_vec = _mm256_set1_epi64x(limit_price);
    const __m256i ones_vec = _mm256_set1_epi64x(1);
    __m256i sum_vec = _mm256_setzero_si256();
    __m256i count_vec = _mm256_setzero_si256();

    std::size_t i = 0;
    const std::size_t vec_end = n - (n % 4); // largest multiple of 4 <= n
    for (; i < vec_end; i += 4) {
        // Two raw 256-bit loads, each holding 2 PriceLevel structs
        // interleaved as {p,q,p,q}: lo = {p0,q0,p1,q1}, hi = {p2,q2,p3,q3}.
        const __m256i lo =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(levels + i));
        const __m256i hi =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(levels + i + 2));

        // _mm256_unpacklo/hi_epi64 operate within each 128-bit lane, so:
        //   unpacklo(lo, hi) = {lo[0], hi[0], lo[2], hi[2]} = {p0, p2, p1, p3}
        //   unpackhi(lo, hi) = {lo[1], hi[1], lo[3], hi[3]} = {q0, q2, q1, q3}
        // Lane j of `prices` corresponds to lane j of `quantities` (p0<->q0,
        // p2<->q2, p1<->q1, p3<->q3) even though the overall order within
        // the vector is shuffled relative to the original array -- that's
        // fine because the reduction below (masked sum/count) is
        // order-independent.
        const __m256i prices = _mm256_unpacklo_epi64(lo, hi);
        const __m256i quantities = _mm256_unpackhi_epi64(lo, hi);

        // AVX2 only has signed 64-bit *greater-than* (_mm256_cmpgt_epi64);
        // build <= and >= from it plus a bitwise NOT (xor against an
        // all-ones vector, the standard AVX2 idiom since there's no andnot-
        // complement instruction for this width).
        const __m256i all_ones = _mm256_cmpeq_epi64(prices, prices);
        __m256i mask;
        if (side == Side::Buy) {
            // price <= limit  <=>  NOT(price > limit)
            const __m256i gt = _mm256_cmpgt_epi64(prices, limit_vec);
            mask = _mm256_xor_si256(gt, all_ones);
        } else {
            // price >= limit  <=>  NOT(limit > price)
            const __m256i gt = _mm256_cmpgt_epi64(limit_vec, prices);
            mask = _mm256_xor_si256(gt, all_ones);
        }

        sum_vec = _mm256_add_epi64(sum_vec, _mm256_and_si256(mask, quantities));
        count_vec = _mm256_add_epi64(count_vec, _mm256_and_si256(mask, ones_vec));
    }

    alignas(32) std::int64_t sum_lanes[4];
    alignas(32) std::int64_t count_lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(sum_lanes), sum_vec);
    _mm256_store_si256(reinterpret_cast<__m256i*>(count_lanes), count_vec);

    Quantity total = sum_lanes[0] + sum_lanes[1] + sum_lanes[2] + sum_lanes[3];
    std::size_t count = static_cast<std::size_t>(count_lanes[0] + count_lanes[1] +
                                                  count_lanes[2] + count_lanes[3]);

    // Scalar tail for the (at most 3) leftover elements when n is not a
    // multiple of 4.
    for (; i < n; ++i) {
        const bool hit = (side == Side::Buy) ? (levels[i].price <= limit_price)
                                              : (levels[i].price >= limit_price);
        if (hit) {
            total += levels[i].quantity;
            ++count;
        }
    }

    return ScanResult{total, count};
}
#endif // __AVX2__

// ---------------------------------------------------------------------------
// Compile-time dispatcher
// ---------------------------------------------------------------------------
ScanResult scan_quantity_at_or_better(const PriceLevel* levels, std::size_t n,
                                       Price limit_price, Side side) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    return scan_quantity_at_or_better_neon(levels, n, limit_price, side);
#elif defined(__AVX2__)
    return scan_quantity_at_or_better_avx2(levels, n, limit_price, side);
#else
    return scan_quantity_at_or_better_scalar(levels, n, limit_price, side);
#endif
}

} // namespace exec_core
