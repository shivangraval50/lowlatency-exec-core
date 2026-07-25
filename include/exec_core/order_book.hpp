#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "exec_core/order.hpp"
#include "exec_core/trade.hpp"
#include "exec_core/types.hpp"

namespace exec_core {

// Single-threaded limit order book with price-time priority matching.
//
// Design (phase 1: correctness first -- no lock-free tricks, no custom
// allocators, no SIMD; those are later phases in PLAN.md):
//
//  - Each side of the book is a std::map<Price, std::list<RestingOrder>>,
//    keyed so that map::begin() is always the best price: bids_ compares
//    with std::greater (highest price first), asks_ with std::less (lowest
//    price first). A std::list per price level gives O(1) push-to-back on
//    arrival and O(1) erase-by-iterator on cancel/fill, while preserving
//    FIFO (time priority) order via list iteration order.
//  - An unordered_map<OrderId, location> gives O(1) average lookup from an
//    order id to its (side, price, list-iterator) so cancel doesn't need to
//    scan the book.
//  - A monotonic sequence counter timestamps orders (for tie-breaking / test
//    visibility) and trades (for a stable global trade ordering).
//
// Complexity:
//  - add_limit_order: O(log P) to locate/create the relevant price level(s),
//    plus O(k) where k is the number of resting orders it fully consumes
//    while matching (each such order is O(1) to remove). Never touches
//    price levels it doesn't cross.
//  - cancel_order: O(1) average (hash lookup + list::erase by iterator),
//    plus O(log P) if removing the order empties its price level and that
//    level must be erased from the map.
//  - best_bid/best_ask: O(1) (map::begin already the extreme element).
//
// P = number of distinct price levels currently resting on that side.
//
// Not covered by this phase (later phases / explicitly out of scope here):
// lock-free/concurrent access, custom allocators, SIMD price scans, market
// orders / stop orders / iceberg orders, order modification (cancel+replace
// is done by the caller as cancel() + add_limit_order()).
class OrderBook {
public:
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
    using LevelOrders = std::list<RestingOrder>;

    struct OrderLocation {
        Side side;
        Price price;
        LevelOrders::iterator it;
    };

    std::map<Price, LevelOrders, std::greater<Price>> bids_; // best (highest) price first
    std::map<Price, LevelOrders, std::less<Price>> asks_;    // best (lowest) price first
    std::unordered_map<OrderId, OrderLocation> locations_;

    Sequence next_sequence_ = 0;
};

} // namespace exec_core
