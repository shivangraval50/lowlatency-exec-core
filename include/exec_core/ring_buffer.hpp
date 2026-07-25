// Lock-free ring buffer intended as the inbound order-ingestion queue feeding
// the phase-1 matching engine (one thread parses/decodes inbound order
// commands and pushes them; the matching-engine thread pops and applies them
// to `OrderBook`). Header-only, since that's idiomatic for a small template
// data structure like this.
//
// ---------------------------------------------------------------------------
// SPSC, not MPMC -- and that's an honest, deliberate choice, not a shortcut.
// ---------------------------------------------------------------------------
// This is a **single-producer/single-consumer** queue ONLY. It is only
// correct if exactly one thread ever calls try_push() and exactly one
// (possibly different) thread ever calls try_pop(). It does NOT support
// multiple concurrent producers or multiple concurrent consumers -- if two
// threads call try_push() concurrently they will race on writing the same
// slot and on the read-modify-write of `tail_`, corrupting the buffer. This
// is intentional and matches the target use case (one ingestion/decode
// thread, one matching-engine thread); a correct MPMC ring buffer needs a
// materially different algorithm (e.g. per-slot sequence numbers, as in
// Dmitry Vyukov's bounded MPMC queue) and is NOT implemented here. Do not
// use this type from more than one producer or more than one consumer
// thread.
//
// ---------------------------------------------------------------------------
// Algorithm
// ---------------------------------------------------------------------------
// Two monotonically-increasing (mod 2^64, never explicitly wrapped) counters:
//   - `tail`: number of items ever pushed. Written only by the producer.
//   - `head`: number of items ever popped.  Written only by the consumer.
// Slot for logical index `i` lives at `buffer_[i & kMask]`, so Capacity must
// be a power of two (checked with static_assert) so the mask is a cheap `&`
// instead of a `%`. The queue is empty when `tail == head` and full when
// `tail - head == Capacity`; because both counters only ever increase,
// there's no need to reserve a wasted "always empty" slot the way naive
// wrap-around-index designs do.
//
// ---------------------------------------------------------------------------
// Memory ordering
// ---------------------------------------------------------------------------
// Each side reads its OWN counter with memory_order_relaxed: only that thread
// ever writes it, so there is no cross-thread data race on that particular
// load and program order within the single writer thread already gives it
// the value it last stored. The interesting synchronization is:
//
//   - Producer: after writing an item into buffer_[tail & kMask], it
//     publishes with `producer_.tail.store(tail + 1, memory_order_release)`.
//     This release prevents the compiler/CPU from reordering the buffer
//     write to *after* the counter becomes visible to the consumer.
//   - Consumer: before reading buffer_[head & kMask], if it needs a fresher
//     view of how many items are available it does
//     `producer_.tail.load(memory_order_acquire)`. This acquire pairs with
//     the producer's release store, so once the consumer observes the
//     incremented tail, it is guaranteed to also observe the item the
//     producer wrote into that slot (release-acquire happens-before).
//   - Symmetrically, the consumer publishes with
//     `consumer_.head.store(head + 1, memory_order_release)` after it has
//     finished reading a slot, and the producer does
//     `consumer_.head.load(memory_order_acquire)` before reusing that slot
//     for a new item. This is not just about the counter's numeric value --
//     it is what prevents the producer from overwriting a slot the consumer
//     has not finished reading yet (release/acquire, not just relaxed, is
//     required here for correctness, not merely as a style preference).
//
// This is acquire/release throughout, not "seq_cst everywhere and hope" --
// on a single-producer/single-consumer queue like this, acquire/release is
// sufficient to establish the happens-before edges we actually need (no
// third thread needs to observe a single total order across both counters),
// and it is measurably cheaper than seq_cst on most architectures (notably
// ARM/AArch64, i.e. this machine's M2, where seq_cst stores require an extra
// barrier that plain release stores don't).
//
// ---------------------------------------------------------------------------
// Cache-line padding: done here, not deferred
// ---------------------------------------------------------------------------
// PLAN.md phase 4 ("cache-line alignment") is about *general* alignment work
// across the codebase (e.g. resting-order / slab-allocator node layout). The
// producer/consumer index padding below is different: it is inherent to this
// specific lock-free algorithm (false sharing between the hot producer index
// and the hot consumer index is a correctness-adjacent performance bug in
// *any* SPSC ring buffer, not a later optimization pass), so it is done now
// rather than deferred.
//
// Layout below groups each atomic with the *cached copy the other side
// doesn't own* onto the same cache line:
//   - "producer line": `tail` (written by producer, read by consumer) and
//     `head_cached` (read+written only by producer).
//   - "consumer line": `head` (written by consumer, read by producer) and
//     `tail_cached` (read+written only by consumer).
// This does not eliminate all cross-core cache traffic (the consumer's
// acquire-load of `tail` will still occasionally miss when the producer has
// just written it), but it avoids *true* false sharing, i.e. two threads
// both actively writing distinct variables that happen to share a line and
// ping-ponging cache-line ownership back and forth. Here only one thread
// ever writes to each line; the other thread only reads it, which is a much
// cheaper pattern (MESI Shared-state read, not Modified/Exclusive contention).
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace exec_core {

#if defined(__cpp_lib_hardware_interference_size)
inline constexpr std::size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
// libc++/libstdc++ on most of our target platforms (macOS/arm64, Linux/x86_64
// CI) don't ship std::hardware_destructive_interference_size in a way we can
// rely on portably yet; 64 bytes is the correct line size for both Apple
// Silicon and current x86-64 server/desktop parts, so it's used as a plain
// constant instead.
inline constexpr std::size_t kCacheLineSize = 64;
#endif

// Fixed-capacity, single-producer/single-consumer lock-free ring buffer.
// `T` is meant to hold a small POD-ish "inbound order command" (or any
// trivially-movable payload); `Capacity` must be a power of two.
template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two (cheap mask, not modulo, for slot indexing)");
    static_assert(std::is_nothrow_move_constructible_v<T> || std::is_copy_constructible_v<T>,
                  "T must be movable or copyable to be stored in the ring buffer");

  public:
    SpscRingBuffer() = default;

    // Non-copyable, non-movable: this type owns a fixed in-place buffer and
    // is meant to be shared by reference/pointer between exactly two threads
    // for its whole lifetime, not passed around by value.
    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;
    SpscRingBuffer(SpscRingBuffer&&) = delete;
    SpscRingBuffer& operator=(SpscRingBuffer&&) = delete;

    // Producer-side only. Attempts to push `item` into the queue.
    // Returns false without blocking if the queue is full.
    bool try_push(const T& item) { return emplace_impl(item); }
    bool try_push(T&& item) { return emplace_impl(std::move(item)); }

    // Consumer-side only. Attempts to pop the oldest item into `out`.
    // Returns false without blocking if the queue is empty.
    bool try_pop(T& out) {
        const std::size_t head = consumer_.head.load(std::memory_order_relaxed);
        if (head == consumer_.tail_cached) {
            // Local cache is stale (or genuinely empty) -- refresh from the
            // producer's published tail. Acquire pairs with the producer's
            // release store in emplace_impl(), so if we observe the new
            // tail value here we are also guaranteed to observe the slot
            // write that happened-before it.
            consumer_.tail_cached = producer_.tail.load(std::memory_order_acquire);
            if (head == consumer_.tail_cached) {
                return false; // genuinely empty
            }
        }
        out = std::move(buffer_[head & kMask]);
        // Release: publishes both the new head value AND (per the memory
        // model) guarantees the read of buffer_[...] above is not reordered
        // to *after* this store -- i.e. the producer, once it observes this
        // new head via its own acquire-load, will never see a stale slot it
        // is about to overwrite as still "in use" by the consumer.
        consumer_.head.store(head + 1, std::memory_order_release);
        return true;
    }

    // Convenience wrapper returning std::optional instead of an out-param.
    std::optional<T> try_pop() {
        T out;
        if constexpr (std::is_default_constructible_v<T>) {
            if (try_pop(out)) {
                return out;
            }
            return std::nullopt;
        } else {
            static_assert(std::is_default_constructible_v<T>,
                          "try_pop() with no out-param requires T to be default-constructible; "
                          "use try_pop(T&) otherwise");
        }
    }

    static constexpr std::size_t capacity() { return Capacity; }

    // Diagnostics only: these are inherently racy under concurrent
    // push/pop (the value can be stale the instant it's returned) and must
    // not be used to decide whether try_push/try_pop will succeed -- call
    // those directly and check their return value instead.
    bool empty_approx() const {
        return producer_.tail.load(std::memory_order_relaxed) == consumer_.head.load(std::memory_order_relaxed);
    }
    std::size_t size_approx() const {
        return producer_.tail.load(std::memory_order_relaxed) - consumer_.head.load(std::memory_order_relaxed);
    }

  private:
    static constexpr std::size_t kMask = Capacity - 1;

    template <typename U>
    bool emplace_impl(U&& item) {
        const std::size_t tail = producer_.tail.load(std::memory_order_relaxed);
        if (tail - producer_.head_cached == Capacity) {
            // Local cache says full -- refresh from the consumer's published
            // head before giving up. Acquire pairs with the consumer's
            // release store in try_pop(), so if we observe the slot as freed
            // we are guaranteed the consumer's read of it has completed.
            producer_.head_cached = consumer_.head.load(std::memory_order_acquire);
            if (tail - producer_.head_cached == Capacity) {
                return false; // genuinely full
            }
        }
        buffer_[tail & kMask] = std::forward<U>(item);
        // Release: publishes both the new tail value AND guarantees the
        // slot write above is visible to the consumer once it observes this
        // store via its acquire-load of tail.
        producer_.tail.store(tail + 1, std::memory_order_release);
        return true;
    }

    // "Producer line": tail is the producer's own published position
    // (written here, read by the consumer); head_cached is the producer's
    // private, unsynchronized cache of the last head value it observed, so
    // it doesn't need to issue an atomic load on every push once there's
    // headroom. See file header for why grouping these two on one cache
    // line is fine (only the producer ever writes to this line).
    struct alignas(kCacheLineSize) ProducerLine {
        std::atomic<std::size_t> tail{0};
        std::size_t head_cached{0};
    };

    // "Consumer line": mirror image of ProducerLine.
    struct alignas(kCacheLineSize) ConsumerLine {
        std::atomic<std::size_t> head{0};
        std::size_t tail_cached{0};
    };

    ProducerLine producer_{};
    ConsumerLine consumer_{};

    // The actual storage. Not padded/cache-line-aligned per element -- that
    // sort of per-element layout tuning (and any slab-allocator-driven
    // packing of the payload type itself) is explicitly left to phase 4
    // ("cache-line alignment") and phase 3 ("custom slab allocator"); this
    // phase's scope is the lock-free index/synchronization logic.
    alignas(kCacheLineSize) std::array<T, Capacity> buffer_{};
};

} // namespace exec_core
