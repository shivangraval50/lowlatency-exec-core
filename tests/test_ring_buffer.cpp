// Real correctness tests for the phase-2 lock-free SPSC ring buffer.
//
// Plain assert()-based tests (no framework), consistent with
// tests/test_order_book.cpp. Covers single-threaded boundary/wrap-around
// behavior and a genuine two-thread concurrent producer/consumer stress
// test with a strict per-item sequence check that would fail on any
// drop/corruption/duplication/reorder.

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "exec_core/ring_buffer.hpp"

using namespace exec_core;

namespace {

// ---------------------------------------------------------------------
// Basic push/pop, empty/full boundary checks.
// ---------------------------------------------------------------------
void test_basic_push_pop() {
    SpscRingBuffer<int, 4> rb;
    assert(rb.capacity() == 4);
    assert(rb.empty_approx());
    assert(rb.size_approx() == 0);

    int out = -1;
    assert(rb.try_pop(out) == false); // empty: pop must fail, out untouched contract-wise
    assert(!rb.try_pop().has_value());

    assert(rb.try_push(10) == true);
    assert(rb.try_push(20) == true);
    assert(rb.size_approx() == 2);
    assert(!rb.empty_approx());

    assert(rb.try_pop(out) == true);
    assert(out == 10);
    assert(rb.try_pop(out) == true);
    assert(out == 20);
    assert(rb.try_pop(out) == false);
    assert(rb.empty_approx());
}

// ---------------------------------------------------------------------
// Push until genuinely full: capacity-th push succeeds, (capacity+1)-th
// push must fail (return false), per the documented "full when
// tail - head == Capacity" contract -- there is no wasted slot.
// ---------------------------------------------------------------------
void test_fill_to_capacity_and_reject_overflow() {
    constexpr std::size_t kCap = 8;
    SpscRingBuffer<int, kCap> rb;

    for (std::size_t i = 0; i < kCap; ++i) {
        assert(rb.try_push(static_cast<int>(i)) == true);
    }
    assert(rb.size_approx() == kCap);

    // Buffer is genuinely full now -- further pushes must be rejected, not
    // silently overwrite/corrupt existing slots.
    assert(rb.try_push(999) == false);
    assert(rb.try_push(998) == false);
    assert(rb.size_approx() == kCap); // unaffected by the rejected pushes

    // Drain and verify FIFO order + values are exactly what was pushed
    // (nothing corrupted by the rejected overflow attempts).
    for (std::size_t i = 0; i < kCap; ++i) {
        int out = -1;
        assert(rb.try_pop(out) == true);
        assert(out == static_cast<int>(i));
    }
    assert(rb.try_pop().has_value() == false);
    assert(rb.empty_approx());
}

// ---------------------------------------------------------------------
// Wrap-around reuse of slots: run many more full push/pop cycles than the
// capacity so that the underlying index arithmetic wraps around the
// backing array multiple times, and check FIFO ordering/values hold
// across every wrap.
// ---------------------------------------------------------------------
void test_wraparound_many_cycles() {
    constexpr std::size_t kCap = 4;
    SpscRingBuffer<std::uint64_t, kCap> rb;

    std::uint64_t next_expected_push_value = 0;
    std::uint64_t next_expected_pop_value = 0;

    // Do enough full push/pop cycles to wrap the mod-Capacity index many
    // times over (this exercises tail/head incrementing well past 2^k
    // multiples of Capacity, not just a single wrap).
    constexpr int kCycles = 10000;
    for (int cycle = 0; cycle < kCycles; ++cycle) {
        // Fill completely.
        for (std::size_t i = 0; i < kCap; ++i) {
            assert(rb.try_push(next_expected_push_value) == true);
            ++next_expected_push_value;
        }
        assert(rb.try_push(0xDEADBEEFULL) == false); // full, must reject

        // Partially drain (2 of 4), then push 2 more (still respecting
        // capacity), then fully drain -- exercises interleaved
        // wrap-around, not just "fill fully then empty fully".
        for (std::size_t i = 0; i < kCap / 2; ++i) {
            std::uint64_t out = 0xFFFFFFFFULL;
            assert(rb.try_pop(out) == true);
            assert(out == next_expected_pop_value);
            ++next_expected_pop_value;
        }
        for (std::size_t i = 0; i < kCap / 2; ++i) {
            assert(rb.try_push(next_expected_push_value) == true);
            ++next_expected_push_value;
        }
        for (std::size_t i = 0; i < kCap; ++i) {
            std::uint64_t out = 0xFFFFFFFFULL;
            assert(rb.try_pop(out) == true);
            assert(out == next_expected_pop_value);
            ++next_expected_pop_value;
        }
        assert(rb.try_pop().has_value() == false); // empty at end of cycle
        assert(rb.empty_approx());
    }
    assert(next_expected_push_value == next_expected_pop_value);
    // Per cycle: kCap pushed on the initial fill, then kCap/2 pushed again
    // after the partial drain -- i.e. kCap + kCap/2 total pushes/pops per
    // cycle, not kCap*2 (the rejected overflow push and the final full
    // drain don't add extra pushed items).
    assert(next_expected_push_value == static_cast<std::uint64_t>(kCycles) * (kCap + kCap / 2));
}

// ---------------------------------------------------------------------
// Concurrent correctness: real std::thread producer + std::thread
// consumer. Producer pushes strictly-increasing sequence numbers;
// consumer verifies every value received is exactly the next expected
// one (catches drops, duplication, reordering, and corruption -- any of
// those would fail the strict equality check below). Uses a busy-spin
// retry on try_push/try_pop failure (expected/documented non-blocking
// behavior when full/empty).
// ---------------------------------------------------------------------
void test_concurrent_spsc_sequence_integrity() {
    constexpr std::size_t kCap = 1024;
    constexpr std::uint64_t kNumItems = 2'000'000;

    SpscRingBuffer<std::uint64_t, kCap> rb;
    std::atomic<bool> mismatch{false};
    std::atomic<std::uint64_t> consumed_count{0};

    std::thread producer([&rb, &mismatch]() {
        for (std::uint64_t i = 0; i < kNumItems && !mismatch.load(std::memory_order_relaxed); ++i) {
            while (!rb.try_push(i)) {
                // full -- spin until the consumer frees a slot
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&rb, &mismatch, &consumed_count]() {
        std::uint64_t expected = 0;
        std::uint64_t out = 0;
        while (expected < kNumItems) {
            if (rb.try_pop(out)) {
                if (out != expected) {
                    mismatch.store(true, std::memory_order_relaxed);
                    std::fprintf(stderr,
                                 "MISMATCH at expected=%llu, got=%llu\n",
                                 static_cast<unsigned long long>(expected),
                                 static_cast<unsigned long long>(out));
                    return;
                }
                ++expected;
                consumed_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    assert(mismatch.load() == false);
    assert(consumed_count.load() == kNumItems);
    assert(rb.empty_approx());
}

// ---------------------------------------------------------------------
// Move-only payload sanity check: the ring buffer's element type just
// needs to be movable (per the static_assert in the header), so verify a
// move-only struct works correctly through try_push(T&&)/try_pop(T&),
// including that moved-from source values don't alias/corrupt what comes
// out the other end.
// ---------------------------------------------------------------------
struct MoveOnlyPayload {
    int value;
    MoveOnlyPayload(int v = 0) : value(v) {} // NOLINT(google-explicit-constructor)
    MoveOnlyPayload(const MoveOnlyPayload&) = delete;
    MoveOnlyPayload& operator=(const MoveOnlyPayload&) = delete;
    MoveOnlyPayload(MoveOnlyPayload&& other) noexcept : value(other.value) { other.value = -1; }
    MoveOnlyPayload& operator=(MoveOnlyPayload&& other) noexcept {
        value = other.value;
        other.value = -1;
        return *this;
    }
};

void test_move_only_payload() {
    SpscRingBuffer<MoveOnlyPayload, 4> rb;

    for (int i = 0; i < 4; ++i) {
        MoveOnlyPayload p(i * 10);
        assert(rb.try_push(std::move(p)) == true);
    }
    assert(rb.try_push(MoveOnlyPayload(999)) == false); // full

    for (int i = 0; i < 4; ++i) {
        MoveOnlyPayload out;
        assert(rb.try_pop(out) == true);
        assert(out.value == i * 10);
    }
    MoveOnlyPayload drained;
    assert(rb.try_pop(drained) == false); // empty now
}

} // namespace

int main() {
    struct NamedTest {
        const char* name;
        void (*fn)();
    };
    const NamedTest tests[] = {
        {"basic_push_pop", test_basic_push_pop},
        {"fill_to_capacity_and_reject_overflow", test_fill_to_capacity_and_reject_overflow},
        {"wraparound_many_cycles", test_wraparound_many_cycles},
        {"concurrent_spsc_sequence_integrity", test_concurrent_spsc_sequence_integrity},
        {"move_only_payload", test_move_only_payload},
    };

    for (const auto& t : tests) {
        t.fn();
        std::printf("[PASS] %s\n", t.name);
    }
    std::printf("All %zu ring_buffer tests passed.\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
