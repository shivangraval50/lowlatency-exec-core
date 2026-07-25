// Real tests for SlabAllocator<T> (phase 3's fixed-block-size pool
// allocator), consistent with tests/test_order_book.cpp and
// tests/test_ring_buffer.cpp: plain assert()-based, no framework, each test
// a free function, main() runs them all and prints a summary.
//
// Covers: basic alloc/free, free-list reuse (LIFO), exhaustion/growth
// behavior (including that growth doesn't corrupt already-live blocks), and
// alignment correctness for an over-aligned, non-trivial type -- including
// after a growth chunk is added, since add_chunk() is the code path that
// actually computes block stride/alignment.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "exec_core/slab_allocator.hpp"

using namespace exec_core;

namespace {

// Small trivial type -- exercises the common case (comparable in size to
// OrderNode's pointer members, but simpler to reason about in a test).
struct Small {
    int x;
};

// Deliberately over-aligned (32, stricter than alignof(std::max_align_t) on
// this platform and stricter than the FreeNode pointer the allocator
// threads through free blocks) and non-trivial in the sense that it holds
// real data a misaligned/corrupted write would visibly break.
struct alignas(32) Aligned32 {
    double data[4];
    int tag;
};

// ---------------------------------------------------------------------
// Basic allocate/construct/use/destroy/deallocate round trip.
// ---------------------------------------------------------------------
void test_basic_alloc_free() {
    SlabAllocator<Small> pool(8);
    assert(pool.capacity() == 8);
    assert(pool.in_use() == 0);
    assert(pool.growth_events() == 0);

    Small* p = pool.allocate();
    assert(p != nullptr);
    new (p) Small{42};
    assert(p->x == 42);
    assert(pool.in_use() == 1);
    assert(pool.capacity() == 8); // single allocate() within initial capacity: no growth
    assert(pool.growth_events() == 0);

    p->~Small();
    pool.deallocate(p);
    assert(pool.in_use() == 0);
    assert(pool.capacity() == 8);
}

// ---------------------------------------------------------------------
// Free-list reuse: freeing a block and immediately allocating again must
// hand back that exact block (LIFO free list, no leak, no fresh growth).
// ---------------------------------------------------------------------
void test_free_list_reuse() {
    SlabAllocator<Small> pool(4);

    Small* a = pool.allocate();
    new (a) Small{1};
    Small* b = pool.allocate();
    new (b) Small{2};
    assert(pool.in_use() == 2);

    a->~Small();
    pool.deallocate(a);
    assert(pool.in_use() == 1);

    // Next allocate() must reuse `a`'s slot: the free list is LIFO and
    // nothing else was freed in between.
    Small* c = pool.allocate();
    assert(c == a);
    new (c) Small{3};
    assert(pool.in_use() == 2);
    assert(pool.growth_events() == 0); // reuse only -- never touched the GP allocator again
    assert(pool.capacity() == 4);

    // Cycle through many more free/alloc pairs on the same two live slots
    // to make sure repeated reuse doesn't drift capacity or leak.
    for (int i = 0; i < 100; ++i) {
        b->~Small();
        pool.deallocate(b);
        b = pool.allocate();
        new (b) Small{2 + i};
        assert(pool.in_use() == 2);
        assert(pool.capacity() == 4);
        assert(pool.growth_events() == 0);
    }

    c->~Small();
    pool.deallocate(c);
    b->~Small();
    pool.deallocate(b);
    assert(pool.in_use() == 0);
    assert(pool.capacity() == 4);
}

// ---------------------------------------------------------------------
// Exhaustion/growth: allocate past the initial capacity and confirm the
// pool grows (documented policy) rather than crashing/corrupting, that
// growth_events() reflects each growth, and that blocks allocated *before*
// a growth event still hold their original values afterward (i.e. growth
// allocates a fresh chunk rather than doing something that could alias or
// invalidate already-live blocks).
// ---------------------------------------------------------------------
void test_exhaustion_triggers_growth_without_corruption() {
    constexpr std::size_t kInit = 4;
    SlabAllocator<Small> pool(kInit);

    std::vector<std::pair<Small*, int>> live; // (pointer, value it should hold)
    const auto alloc_one = [&](int value) {
        Small* p = pool.allocate();
        new (p) Small{value};
        live.emplace_back(p, value);
    };

    for (std::size_t i = 0; i < kInit; ++i) {
        alloc_one(static_cast<int>(i));
    }
    assert(pool.capacity() == kInit);
    assert(pool.growth_events() == 0);
    assert(pool.in_use() == kInit);

    // One more allocate() beyond the initial reservation must grow, not
    // crash and not silently overrun the arena.
    alloc_one(999);
    assert(pool.growth_events() == 1);
    assert(pool.capacity() == kInit * 2);
    assert(pool.in_use() == kInit + 1);

    // Force a second growth event by allocating well past the new capacity.
    for (std::size_t i = 0; i < kInit * 3; ++i) {
        alloc_one(static_cast<int>(2000 + i));
    }
    assert(pool.growth_events() >= 2);
    assert(pool.capacity() >= kInit + 1 + kInit * 3);
    assert(pool.in_use() == live.size());

    // Every block must be a distinct address (no aliasing across chunks).
    std::set<Small*> unique_ptrs;
    for (auto& [ptr, value] : live) {
        unique_ptrs.insert(ptr);
    }
    assert(unique_ptrs.size() == live.size());

    // Every block, including ones allocated before either growth event,
    // must still hold the value it was constructed with -- growth must not
    // move, reuse, or corrupt already-live blocks.
    for (auto& [ptr, value] : live) {
        assert(ptr->x == value);
    }

    for (auto& [ptr, value] : live) {
        (void)value;
        ptr->~Small();
        pool.deallocate(ptr);
    }
    assert(pool.in_use() == 0);
}

// ---------------------------------------------------------------------
// Alignment correctness for a non-trivial, over-aligned type, both within
// the initial chunk and after growth chunks are added (add_chunk() is
// where block stride/alignment is actually computed, so growth chunks are
// exactly where an off-by-one in that arithmetic would first show up).
// ---------------------------------------------------------------------
void test_alignment_for_overaligned_type() {
    constexpr std::size_t kInit = 3;
    SlabAllocator<Aligned32> pool(kInit);

    std::vector<Aligned32*> ptrs;
    const auto alloc_and_check = [&](int tag) {
        Aligned32* p = pool.allocate();
        const auto addr = reinterpret_cast<std::uintptr_t>(p);
        assert(addr % alignof(Aligned32) == 0);
        new (p) Aligned32{{1.5, 2.5, 3.5, 4.5}, tag};
        // Read/write through the object to give UBSan/ASan a real chance to
        // flag a misaligned or out-of-bounds access if the stride/alignment
        // math were wrong.
        assert(p->data[0] == 1.5 && p->data[3] == 4.5 && p->tag == tag);
        p->data[1] += 10.0;
        assert(p->data[1] == 12.5);
        ptrs.push_back(p);
    };

    // Fill the initial chunk...
    for (std::size_t i = 0; i < kInit; ++i) {
        alloc_and_check(static_cast<int>(i));
    }
    assert(pool.growth_events() == 0);

    // ...then force growth chunks and re-check alignment on the newly
    // reserved blocks.
    for (int i = 0; i < 20; ++i) {
        alloc_and_check(100 + i);
    }
    assert(pool.growth_events() > 0);

    for (Aligned32* p : ptrs) {
        p->~Aligned32();
        pool.deallocate(p);
    }
    assert(pool.in_use() == 0);
}

// ---------------------------------------------------------------------
// Zero initial capacity is documented to be clamped to 1 rather than
// producing a zero-size allocation (which would be UB-adjacent territory
// for ::operator new sizing math). Confirm it still works correctly and
// still grows on demand.
// ---------------------------------------------------------------------
void test_zero_initial_capacity_clamped() {
    SlabAllocator<Small> pool(0);
    assert(pool.capacity() == 1);

    Small* a = pool.allocate();
    new (a) Small{7};
    assert(pool.in_use() == 1);
    assert(pool.growth_events() == 0);

    Small* b = pool.allocate(); // must grow: only 1 block was reserved
    new (b) Small{8};
    assert(pool.growth_events() == 1);
    assert(a != b);
    assert(a->x == 7 && b->x == 8);

    a->~Small();
    pool.deallocate(a);
    b->~Small();
    pool.deallocate(b);
}

} // namespace

int main() {
    struct NamedTest {
        const char* name;
        void (*fn)();
    };
    const NamedTest tests[] = {
        {"basic_alloc_free", test_basic_alloc_free},
        {"free_list_reuse", test_free_list_reuse},
        {"exhaustion_triggers_growth_without_corruption", test_exhaustion_triggers_growth_without_corruption},
        {"alignment_for_overaligned_type", test_alignment_for_overaligned_type},
        {"zero_initial_capacity_clamped", test_zero_initial_capacity_clamped},
    };

    for (const auto& t : tests) {
        t.fn();
        std::printf("[PASS] %s\n", t.name);
    }
    std::printf("All %zu slab_allocator tests passed.\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
