// Fixed-block-size slab/pool allocator, used by OrderBook (see
// order_book.hpp) to carve resting-order list nodes out of a single
// pre-reserved arena instead of calling the general-purpose allocator
// (`new`/`delete`) on every order add/fill/cancel.
//
// ---------------------------------------------------------------------------
// Design
// ---------------------------------------------------------------------------
//  - `SlabAllocator<T>` reserves `initial_capacity` blocks of
//    `sizeof(T)`/`alignof(T)` up front, in one contiguous heap allocation
//    ("chunk"), at construction time. Blocks are handed out via an
//    intrusive singly-linked free list threaded directly through each
//    free block's own (otherwise-unused) storage -- there is no separate
//    bookkeeping array/bitmap, so the only per-block overhead is whatever
//    padding alignment requires.
//  - `allocate()` / `deallocate()` are O(1): pop, respectively push, the
//    head of the free list. Once the pool is warmed up (every block that
//    will ever be needed has been touched at least once -- e.g. after a
//    burst that fills the book to its steady-state depth), further
//    alloc/deallocate cycles do NOT call into the general-purpose
//    allocator at all: no `new`, no `malloc`, no `free`, no `delete` on the
//    hot path.
//  - `allocate()` returns raw, uninitialized storage sized/aligned for `T`
//    (it does not construct a `T`); `deallocate()` expects the object
//    already destroyed (it does not call `~T()`). This mirrors
//    `std::allocator`/`std::pmr::memory_resource` convention: the *caller*
//    (here, `OrderBook`) is the "container" responsible for construction
//    and destruction; the allocator only owns raw memory. See
//    order_book.cpp's `push_back`/`pop_front`/`erase` helpers, which
//    placement-new an `OrderNode` into the returned storage and explicitly
//    call `~OrderNode()` before returning it.
//
// ---------------------------------------------------------------------------
// Exhaustion policy -- explicit and documented, not silent UB
// ---------------------------------------------------------------------------
// If the free list is empty when `allocate()` is called (every previously
// reserved block is currently in use), the pool allocates ONE more chunk of
// `initial_capacity` additional blocks via `::operator new` (this is
// tracked in `growth_events()` so a caller can tell whether this ever
// happened) and continues. This is a deliberate choice over either (a)
// throwing/returning failure and rejecting a valid order, or (b) silently
// overrunning the arena and corrupting memory:
//   - A hard fixed-capacity failure would mean a legitimate incoming order
//     could be rejected purely because of a pool-sizing guess, which is a
//     worse failure mode for a matching engine than one extra (rare,
//     off-the-common-path) heap allocation.
//   - Never growing and instead writing past the reserved arena would be
//     undefined behavior / heap corruption -- unacceptable regardless of
//     how "rare" it might be at some scale.
// The tradeoff being made explicit: after growth, alloc/deallocate are
// still O(1) and heap-free again (the new chunk's blocks just join the
// same free list), so this only costs one allocation per growth event, not
// a permanent slowdown -- but it does mean "no heap calls on the hot path"
// is a *steady-state* property, not an absolute one, if the initial
// capacity guess turns out to be too small. Call `capacity()` /
// `growth_events()` to check whether that happened for a given workload.
//
// Every chunk this pool allocates is freed in `~SlabAllocator()`; the pool
// never leaks, but it also never shrinks while alive (per-chunk memory is
// only released at pool destruction, not when blocks are freed back to the
// pool) -- a deliberate simplicity/latency tradeoff for a bounded-lifetime
// exec-core process, not something a general-purpose allocator would do.
//
// ---------------------------------------------------------------------------
// Thread-safety
// ---------------------------------------------------------------------------
// NOT thread-safe: no locks, no atomics. Like the rest of this project
// through phase 3 (see README.md: "single-threaded matching engine"), this
// is meant to be used from one thread at a time. A concurrent version would
// need CAS-based free-list operations and a solution to the classic ABA
// problem; that is out of scope here and would undercut the project's
// honest single-threaded framing.
//
// ---------------------------------------------------------------------------
// Phase 4 alignment audit: no `alignas(kCacheLineSize)` added here either
// ---------------------------------------------------------------------------
// Same reasoning as order_book.hpp's OrderNode/PriceLevel/OrderLocation
// audit: false sharing requires two threads actively contending for the
// same cache line, and this allocator's free_list_ pointer, chunk_size_,
// total_capacity_, in_use_, growth_events_ counters, and chunks_ vector are
// (per the paragraph above) only ever touched by whichever single thread
// owns this SlabAllocator instance. Aligning any of that bookkeeping would
// only inflate the object and, for FreeNode specifically, work directly
// against the point of this allocator: FreeNode is deliberately as small as
// `T` needs it to be (see kBlockSize below) so that blocks pack tightly and
// a chunk allocated for, say, 4096 OrderNodes actually fits in the memory
// footprint that implies -- padding every block out to a cache line would
// multiply the arena's memory/cache footprint for a benefit (avoiding
// false sharing) that doesn't exist in a single-threaded allocator.
//
// ---------------------------------------------------------------------------
// Sanitizer coverage caveat
// ---------------------------------------------------------------------------
// AddressSanitizer's redzones are placed around each `::operator new` chunk,
// not around individual blocks within it -- so a "logical" use-after-free or
// double-free that stays within a still-live chunk (e.g. reading a block
// after `deallocate()` but before it's reused) is invisible to ASan. Real
// chunk-level overruns/UAF/double-free at the `::operator new`/`delete`
// boundary are still caught. This is a general limitation of any hand-rolled
// pool allocator under ASan, not specific to sloppy code here; call-site
// alloc/dealloc pairing in order_book.cpp was additionally reviewed by hand
// for this reason. (Separately, LeakSanitizer is unsupported on macOS/ASan,
// so leak-freedom on this platform relies on the same manual review, not an
// automated leak-check run.)
#pragma once

#include <cstddef>
#include <new>
#include <vector>

namespace exec_core {

template <typename T>
class SlabAllocator {
  public:
    // Not measured against a real workload -- a plain starting guess (TODO:
    // revisit once phase 6's latency harness gives an actual order-arrival
    // workload to size against). Exceeding it costs one growth event, not
    // a failure; see file header.
    static constexpr std::size_t kDefaultInitialCapacity = 4096;

    explicit SlabAllocator(std::size_t initial_capacity = kDefaultInitialCapacity)
        : chunk_size_(initial_capacity == 0 ? 1 : initial_capacity) {
        add_chunk(chunk_size_);
    }

    ~SlabAllocator() {
        for (void* chunk : chunks_) {
            ::operator delete(chunk, std::align_val_t(kBlockAlign));
        }
    }

    // This type owns raw heap chunks and the pointer-linked free list
    // threaded through them; copying would either duplicate the arena
    // (surprising, expensive) or alias it (double-free on double
    // destruction). Moving is intentionally disallowed too, matching this
    // project's other owned-memory types (see SpscRingBuffer in
    // ring_buffer.hpp): callers are expected to hold this by reference
    // from wherever it's declared, not pass it around by value.
    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;
    SlabAllocator(SlabAllocator&&) = delete;
    SlabAllocator& operator=(SlabAllocator&&) = delete;

    // Returns raw storage for one `T`, sized/aligned correctly but NOT
    // constructed -- caller must placement-new a `T` into it. Grows the
    // pool (see file header) if the free list is currently empty.
    T* allocate() {
        if (free_list_ == nullptr) {
            add_chunk(chunk_size_);
            ++growth_events_;
        }
        FreeNode* node = free_list_;
        free_list_ = free_list_->next;
        ++in_use_;
        return reinterpret_cast<T*>(node);
    }

    // Returns storage previously obtained from allocate() back to the
    // pool. Caller must have already destroyed the `T` living there (this
    // does not call ~T()); O(1), never calls into the general-purpose
    // allocator.
    void deallocate(T* p) noexcept {
        FreeNode* node = new (p) FreeNode;
        node->next = free_list_;
        free_list_ = node;
        --in_use_;
    }

    // Total blocks currently reserved (initial capacity plus any growth
    // chunks). Monotonically non-decreasing over the pool's lifetime.
    std::size_t capacity() const noexcept { return total_capacity_; }

    // Blocks currently handed out via allocate() and not yet returned via
    // deallocate().
    std::size_t in_use() const noexcept { return in_use_; }

    // Number of times allocate() had to grow the pool beyond its most
    // recent capacity (i.e. the free list was observed empty). Zero means
    // the initial reservation was never exceeded and every allocate() call
    // over this pool's lifetime was a pure free-list pop with no heap
    // traffic.
    std::size_t growth_events() const noexcept { return growth_events_; }

  private:
    // Free blocks store a `FreeNode` (just a next-pointer) directly in
    // their own (otherwise-unused, not-yet-constructed) bytes -- the
    // classic intrusive-free-list pool-allocator trick, avoiding a
    // separate bookkeeping array. Placement-new'd into the block's storage
    // (see allocate()/deallocate()/add_chunk()) so this is a real object
    // with a real lifetime for the purposes of the C++ object model, not a
    // raw reinterpret_cast onto unstarted storage -- this matters for
    // strict-aliasing/lifetime correctness under UBSan.
    struct FreeNode {
        FreeNode* next;
    };
    // Regression guard for the "no alignas(kCacheLineSize)" audit above: a
    // bare single-pointer struct has alignof <= alignof(void*); if a future
    // edit ever padded FreeNode out to a cache line (directly or by way of
    // some added member), this fails at compile time for every T this
    // allocator is instantiated with, instead of silently multiplying the
    // arena's memory/cache footprint.
    static_assert(alignof(FreeNode) <= alignof(void*),
                  "FreeNode's alignment exceeds a plain pointer's -- see the "
                  "phase-4 alignment audit comment above before adding alignas here.");

    static constexpr std::size_t kRawBlockSize = sizeof(T) > sizeof(FreeNode) ? sizeof(T) : sizeof(FreeNode);
    static constexpr std::size_t kBlockAlign = alignof(T) > alignof(FreeNode) ? alignof(T) : alignof(FreeNode);
    // Round the per-block stride up to a multiple of the alignment so an
    // array of blocks keeps every block correctly aligned.
    static constexpr std::size_t kBlockSize = ((kRawBlockSize + kBlockAlign - 1) / kBlockAlign) * kBlockAlign;

    void add_chunk(std::size_t count) {
        void* raw = ::operator new(kBlockSize * count, std::align_val_t(kBlockAlign));
        chunks_.push_back(raw);
        auto* bytes = static_cast<std::byte*>(raw);
        for (std::size_t i = 0; i < count; ++i) {
            void* block = bytes + i * kBlockSize;
            FreeNode* node = new (block) FreeNode;
            node->next = free_list_;
            free_list_ = node;
        }
        total_capacity_ += count;
    }

    std::vector<void*> chunks_;
    FreeNode* free_list_ = nullptr;
    std::size_t chunk_size_;
    std::size_t total_capacity_ = 0;
    std::size_t in_use_ = 0;
    std::size_t growth_events_ = 0;
};

} // namespace exec_core
