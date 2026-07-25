#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "exec_core/order.hpp"
#include "exec_core/slab_allocator.hpp"
#include "exec_core/trade.hpp"
#include "exec_core/types.hpp"

namespace exec_core {

// Single-threaded limit order book with price-time priority matching.
//
// Design (phase 1: correctness first; phase 3 replaces the node allocation
// strategy described below -- no lock-free tricks, no SIMD yet; those are
// later phases in PLAN.md):
//
//  - Each side of the book is a std::map<Price, PriceLevel>, keyed so that
//    map::begin() is always the best price: bids_ compares with
//    std::greater (highest price first), asks_ with std::less (lowest
//    price first). `PriceLevel` is a minimal intrusive doubly-linked list
//    (just head/tail raw pointers over `OrderNode`s) giving O(1)
//    push-to-back on arrival and O(1) erase-by-pointer on cancel/fill,
//    while preserving FIFO (time priority) order via list traversal order
//    -- the same complexity `std::list<RestingOrder>` gave in phase 1.
//  - Phase 3 change: `OrderNode`s are no longer individually `new`/`delete`d
//    (that was phase 1's `std::list<RestingOrder>`'s default per-node heap
//    traffic). Instead every `OrderNode`, on either side of the book, at
//    any price level, is carved out of ONE shared `SlabAllocator<OrderNode>`
//    pre-reserved arena (`node_pool_` below) via O(1) free-list pop/push --
//    see slab_allocator.hpp for the allocator itself and its exhaustion
//    policy. One shared pool (not one pool per price level) is deliberate:
//    total resting-order count is bounded by book depth, not by how many
//    distinct price levels happen to be occupied, so a single arena sized
//    to expected book depth is the realistic design (a pool per price
//    level would either over-reserve per level or defeat the point of
//    pre-reservation).
//  - An unordered_map<OrderId, location> gives O(1) average lookup from an
//    order id to its (side, price, node pointer) so cancel doesn't need to
//    scan the book.
//  - A monotonic sequence counter timestamps orders (for tie-breaking / test
//    visibility) and trades (for a stable global trade ordering).
//
// Complexity:
//  - add_limit_order: O(log P) to locate/create the relevant price level(s),
//    plus O(k) where k is the number of resting orders it fully consumes
//    while matching (each such order is O(1) to remove). Never touches
//    price levels it doesn't cross.
//  - cancel_order: O(1) average (hash lookup + intrusive-list erase by
//    pointer), plus O(log P) if removing the order empties its price level
//    and that level must be erased from the map.
//  - best_bid/best_ask: O(1) (map::begin already the extreme element).
//
// P = number of distinct price levels currently resting on that side.
//
// Not covered by this phase (later phases / explicitly out of scope here):
// lock-free/concurrent access, SIMD price scans, market orders / stop
// orders / iceberg orders, order modification (cancel+replace is done by
// the caller as cancel() + add_limit_order()).
class OrderBook {
public:
    OrderBook() = default;

    // Owns a pool of raw memory (node_pool_) and pointer-linked intrusive
    // lists into it -- copying would require deep-copying every node and
    // relinking every list, which nothing in this codebase needs; deleting
    // it is honest rather than leaving a silently-wrong-by-default copy
    // that shares the wrong pool. Not moved either, for the same reason
    // SlabAllocator itself isn't (see slab_allocator.hpp): this owns raw
    // memory and is meant to be held by reference/pointer, not passed by
    // value. (Phase 1/2 code never copied or moved an OrderBook, so this
    // is not a behavior change for any existing caller/test.)
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;
    OrderBook(OrderBook&&) = delete;
    OrderBook& operator=(OrderBook&&) = delete;

    // Destroys any still-resting orders' nodes and returns their storage to
    // node_pool_ before the pool itself is torn down (mirrors how e.g.
    // std::vector's destructor destroys its elements before freeing their
    // storage -- SlabAllocator, like std::allocator, only owns raw memory,
    // not object lifetime).
    ~OrderBook();

    // Submits a new limit order. It first matches against resting orders on
    // the opposite side while prices cross (price-time priority: best price
    // first, then oldest order at that price first). Any unfilled remainder
    // rests in the book at `price`. Returns the trades generated, in the
    // order they occurred; empty if nothing crossed.
    //
    // Throws std::invalid_argument if quantity <= 0 or `id` is already
    // resting in (or was previously used and is still tracked by) the book.
    std::vector<Trade> add_limit_order(OrderId id, Side side, Price price, Quantity quantity);

    // Cancels a resting order. Returns true if it was found and removed,
    // false if no such order is currently resting (already filled,
    // cancelled, or never existed).
    bool cancel_order(OrderId id);

    // Best (highest) resting bid price, if any.
    std::optional<Price> best_bid() const;
    // Best (lowest) resting ask price, if any.
    std::optional<Price> best_ask() const;

    // Total remaining quantity resting at `price` on `side`; 0 if there is
    // no such level.
    Quantity quantity_at(Side side, Price price) const;

    // Number of resting orders currently in the book, all price levels.
    std::size_t order_count() const;

    // Aggregate (price, total quantity) per level, best price first. Mainly
    // for tests / introspection, not a hot-path API.
    std::vector<std::pair<Price, Quantity>> bid_levels() const;
    std::vector<std::pair<Price, Quantity>> ask_levels() const;

    // Order ids resting at `price` on `side`, in time priority (oldest /
    // next-to-match first). Empty if there is no such level. For tests.
    std::vector<OrderId> orders_at(Side side, Price price) const;

private:
    // -----------------------------------------------------------------------
    // Phase 4 alignment audit: OrderNode / PriceLevel / OrderLocation
    // deliberately do NOT get `alignas(kCacheLineSize)`.
    // -----------------------------------------------------------------------
    // False sharing (two cores repeatedly bouncing ownership of the same
    // cache line because they're writing *different* variables that happen
    // to land on it) is strictly a multi-core, concurrent-access phenomenon.
    // OrderBook is, by this project's explicit design (see the class-level
    // comment above and README.md: "single-threaded matching engine"),
    // touched by exactly one thread. Even in the intended future pipeline
    // (ring_buffer.hpp's inbound-command queue feeding a matching-engine
    // thread), the producer thread that decodes commands never reaches into
    // OrderBook's internals itself -- it only pushes into the ring buffer;
    // the single consumer thread is the only thing that ever calls
    // add_limit_order()/cancel_order()/etc. and therefore the only thing
    // that ever touches an OrderNode, PriceLevel, or OrderLocation. There is
    // no second thread to false-share *with*.
    //
    // Given that, `alignas(kCacheLineSize)` here would be pure cost with no
    // matching benefit:
    //   - OrderNode is ~48 bytes (RestingOrder: 4 x int64/uint64, plus two
    //     8-byte pointers); padding it to 64 bytes wouldn't just waste
    //     memory per node, it would actively hurt performance, since the
    //     hot loop that matches orders at a price level (see
    //     add_limit_order() in order_book.cpp) walks OrderNode::next
    //     pointer-chasing through however many resting orders are at the
    //     best price -- fewer nodes per cache line means *more* cache
    //     misses during that walk, the opposite of what alignment is meant
    //     to buy you here.
    //   - PriceLevel is 16 bytes (two pointers) and is default-constructed
    //     on every new price level via std::map::operator[] (see the
    //     existing comment on PriceLevel below) -- padding it out would
    //     bloat every std::map<Price, PriceLevel> node for no reason.
    //   - OrderLocation is small and only ever read/written by the single
    //     matching-engine thread via locations_ (also single-threaded).
    //
    // If a later, materially different design ever ran multiple matching
    // threads over sharded/partitioned order books (out of scope today --
    // see README.md's "Limitations" section), *that* would be the point to
    // revisit this, not before.

    // Intrusive doubly-linked-list node for one resting order. Allocated
    // from / freed back to node_pool_ (never via bare new/delete) -- see
    // push_back()/pop_front()/erase() in order_book.cpp.
    struct OrderNode {
        RestingOrder order;
        OrderNode* prev = nullptr;
        OrderNode* next = nullptr;
    };
    // Regression guard for the "no alignas(kCacheLineSize)" audit above: if
    // that alignas were ever accidentally added back (or added to a member
    // type), OrderNode's alignment would jump to kCacheLineSize (64); a
    // plain struct of int64/pointer members has alignof <= alignof(void*)
    // (8 on every platform this project targets), so this static_assert
    // fails immediately rather than silently bloating every node.
    static_assert(alignof(OrderNode) <= alignof(void*),
                  "OrderNode's alignment exceeds a plain pointer's -- see the "
                  "phase-4 alignment audit comment above before adding alignas here.");

    // One FIFO (time-priority) queue of resting orders at a single price.
    // Deliberately just two raw pointers: PriceLevel itself owns no
    // memory (node_pool_ does), so it stays cheap to default-construct,
    // which matters because std::map<Price, PriceLevel>::operator[]
    // default-constructs a new one every time a fresh price level appears.
    struct PriceLevel {
        OrderNode* head = nullptr; // next to match (oldest / time priority)
        OrderNode* tail = nullptr; // most recently arrived
    };
    static_assert(alignof(PriceLevel) <= alignof(void*),
                  "PriceLevel's alignment exceeds a plain pointer's -- see the "
                  "phase-4 alignment audit comment above before adding alignas here.");

    struct OrderLocation {
        Side side;
        Price price;
        OrderNode* node;
    };

    // Intrusive-list helpers. These are OrderBook member functions (not
    // free functions on PriceLevel) because they need node_pool_ to
    // allocate/deallocate OrderNodes; see order_book.cpp.
    OrderNode* push_back(PriceLevel& level, const RestingOrder& order);
    void pop_front(PriceLevel& level);
    void erase(PriceLevel& level, OrderNode* node);
    static Quantity sum_quantity(const PriceLevel& level);
    static std::vector<OrderId> collect_ids(const PriceLevel& level);

    std::map<Price, PriceLevel, std::greater<Price>> bids_; // best (highest) price first
    std::map<Price, PriceLevel, std::less<Price>> asks_;    // best (lowest) price first
    std::unordered_map<OrderId, OrderLocation> locations_;

    // Phase 3: shared arena for OrderNodes across both sides of the book
    // and every price level (see class-level comment above for why one
    // shared pool, not one per level). See slab_allocator.hpp for the
    // allocation/exhaustion policy -- in short: O(1) alloc/free once
    // warmed up, and if resting-order count ever exceeds the initial
    // reservation the pool grows by one more chunk rather than failing an
    // order or corrupting memory.
    SlabAllocator<OrderNode> node_pool_;

    Sequence next_sequence_ = 0;
};

} // namespace exec_core
