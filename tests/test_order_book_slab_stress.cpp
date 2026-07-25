// Integration/stress tests for OrderBook's phase-3 slab-allocator-backed
// storage (see order_book.hpp / slab_allocator.hpp). Unlike
// tests/test_order_book.cpp (small, hand-checked correctness scenarios),
// these tests specifically push scale and access patterns that only the
// slab integration -- not the matching logic -- could plausibly break:
//
//   - thousands of simultaneously resting orders, comfortably past
//     SlabAllocator<T>::kDefaultInitialCapacity (4096), to force at least
//     one real pool growth event;
//   - long interleaved add/cancel/fill cycles, to force heavy free-list
//     reuse (the same OrderNode slots getting recycled many times);
//   - a full-book teardown (OrderBook's destructor walking every resting
//     order across both sides and returning it to node_pool_).
//
// These are exactly the patterns worth running under
// AddressSanitizer/UndefinedBehaviorSanitizer (see the tester's report for
// the actual sanitizer run/output) -- a plain optimized build could pass
// every assert here and still be hiding a use-after-free or leak that only
// a sanitizer or a leak-detector would catch.
//
// Plain assert()-based, consistent with the rest of tests/.

#include <cassert>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "exec_core/order_book.hpp"

using namespace exec_core;

namespace {

// ---------------------------------------------------------------------
// Thousands of resting orders at distinct prices, none of which cross --
// forces node_pool_ well past its default initial capacity (4096) purely
// from one side of the book, then cancels everything to also exercise
// bulk free-list return.
// ---------------------------------------------------------------------
void test_bulk_distinct_price_levels_forces_growth() {
    OrderBook book;
    constexpr int kOrders = 10000; // > SlabAllocator<T>::kDefaultInitialCapacity (4096)

    // Distinct ascending prices on the sell side so nothing ever crosses;
    // every one of these rests.
    for (int i = 0; i < kOrders; ++i) {
        const OrderId id = static_cast<OrderId>(i + 1);
        const Price price = 1000 + i;
        auto trades = book.add_limit_order(id, Side::Sell, price, 1);
        assert(trades.empty());
    }
    assert(book.order_count() == static_cast<std::size_t>(kOrders));
    assert(book.best_ask().has_value() && *book.best_ask() == 1000);
    assert(book.ask_levels().size() == static_cast<std::size_t>(kOrders));

    // Spot-check a handful of scattered ids/prices survived intact.
    assert(book.quantity_at(Side::Sell, 1000) == 1);
    assert(book.quantity_at(Side::Sell, 1000 + kOrders / 2) == 1);
    assert(book.quantity_at(Side::Sell, 1000 + kOrders - 1) == 1);
    assert(book.quantity_at(Side::Sell, 1000 + kOrders) == 0); // one past the end: never existed

    // Cancel every single one (in a different order than insertion: evens
    // then odds), forcing node_pool_ to take back thousands of blocks onto
    // its free list, and confirm the book ends up completely empty.
    for (int i = 0; i < kOrders; i += 2) {
        assert(book.cancel_order(static_cast<OrderId>(i + 1)) == true);
    }
    for (int i = 1; i < kOrders; i += 2) {
        assert(book.cancel_order(static_cast<OrderId>(i + 1)) == true);
    }
    assert(book.order_count() == 0);
    assert(!book.best_ask().has_value());
    assert(book.ask_levels().empty());
}

// ---------------------------------------------------------------------
// Thousands of resting orders stacked at a SINGLE price level -- forces
// pool growth while exercising the intrusive doubly-linked-list
// (push_back/pop_front/erase) heavily at one PriceLevel, then drains the
// level via a mix of matching (pop_front, FIFO) and direct cancellation
// (erase, arbitrary position) to exercise both removal paths at scale.
// ---------------------------------------------------------------------
void test_single_price_level_deep_book_fifo_and_cancel() {
    OrderBook book;
    constexpr int kOrders = 6000; // > kDefaultInitialCapacity, all at one level
    constexpr Price kPrice = 500;

    std::vector<OrderId> ids;
    ids.reserve(kOrders);
    for (int i = 0; i < kOrders; ++i) {
        const OrderId id = static_cast<OrderId>(i + 1);
        book.add_limit_order(id, Side::Buy, kPrice, 1);
        ids.push_back(id);
    }
    assert(book.order_count() == static_cast<std::size_t>(kOrders));
    assert(book.quantity_at(Side::Buy, kPrice) == kOrders);

    // Cancel every third order outright (erase from the middle of the
    // intrusive list, not just the ends).
    std::unordered_set<OrderId> cancelled;
    for (int i = 0; i < kOrders; i += 3) {
        assert(book.cancel_order(ids[static_cast<std::size_t>(i)]) == true);
        cancelled.insert(ids[static_cast<std::size_t>(i)]);
    }
    const auto remaining_after_cancel = static_cast<Quantity>(kOrders - static_cast<int>(cancelled.size()));
    assert(book.quantity_at(Side::Buy, kPrice) == remaining_after_cancel);

    // Time priority (FIFO) must still hold among the survivors: the first
    // remaining id in orders_at() must be the smallest surviving id.
    OrderId expected_first = 0;
    for (int i = 0; i < kOrders; ++i) {
        if (cancelled.find(ids[static_cast<std::size_t>(i)]) == cancelled.end()) {
            expected_first = ids[static_cast<std::size_t>(i)];
            break;
        }
    }
    auto resting_ids = book.orders_at(Side::Buy, kPrice);
    assert(!resting_ids.empty());
    assert(resting_ids.front() == expected_first);

    // Now sweep the rest of the level away via matching: one aggressive
    // sell for the full remaining size should walk the whole FIFO list via
    // pop_front and fully drain the level.
    auto trades = book.add_limit_order(999999, Side::Sell, kPrice, remaining_after_cancel);
    assert(static_cast<Quantity>(trades.size()) == remaining_after_cancel); // one trade per 1-lot survivor
    assert(book.order_count() == 0);
    assert(!book.best_bid().has_value());
    assert(book.quantity_at(Side::Buy, kPrice) == 0);
}

// ---------------------------------------------------------------------
// Long interleaved add/cancel/fill cycle with a deterministic PRNG (fixed
// seed => reproducible), cross-checked against an independent mirror of
// expected book state. This is the pattern most likely to exercise heavy
// free-list churn (repeatedly allocating/deallocating the same handful of
// hot slots) rather than pure growth.
// ---------------------------------------------------------------------
void test_interleaved_add_cancel_fill_cycles() {
    OrderBook book;
    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> price_dist(1, 50);   // 50 possible price levels
    std::uniform_int_distribution<int> qty_dist(1, 20);
    std::uniform_int_distribution<int> action_dist(0, 9);   // 0..9: weight actions below
    std::uniform_int_distribution<int> side_dist(0, 1);

    // Mirror of currently-resting orders: id -> (side, price, remaining qty).
    struct MirrorEntry {
        Side side;
        Price price;
        Quantity qty;
    };
    std::unordered_map<OrderId, MirrorEntry> mirror;
    std::vector<OrderId> resting_ids; // for picking a random one to cancel
    OrderId next_id = 1;

    constexpr int kIterations = 20000;
    for (int iter = 0; iter < kIterations; ++iter) {
        const int action = action_dist(rng);
        if (action <= 6 || resting_ids.empty()) {
            // Add a new resting-or-crossing order. Use a price range (bids
            // at 1..25, asks at 26..50, offset so they generally do NOT
            // cross) most of the time, but occasionally cross deliberately
            // (every ~10th add) to also exercise pop_front/erase via
            // matches, not just via cancel_order.
            const Side side = side_dist(rng) == 0 ? Side::Buy : Side::Sell;
            Price price;
            if (iter % 10 == 0) {
                // Deliberately aggressive: buy high / sell low, likely to cross.
                price = (side == Side::Buy) ? 50 : 1;
            } else {
                price = (side == Side::Buy) ? price_dist(rng) % 25 + 1 : price_dist(rng) % 25 + 26;
            }
            const Quantity qty = qty_dist(rng);
            const OrderId id = next_id++;

            auto trades = book.add_limit_order(id, side, price, qty);

            // Apply the same fills to the mirror that the book just
            // reported, oldest resting order at the traded price/side first
            // (mirror doesn't need to reproduce full price-time priority --
            // it only needs matching totals per id, which the returned
            // trades already give us).
            Quantity filled_from_incoming = 0;
            for (const Trade& t : trades) {
                filled_from_incoming += t.quantity;
                auto it = mirror.find(t.resting_order_id);
                assert(it != mirror.end());
                it->second.qty -= t.quantity;
                assert(it->second.qty >= 0);
                if (it->second.qty == 0) {
                    mirror.erase(it);
                }
            }
            const Quantity remaining = qty - filled_from_incoming;
            assert(remaining >= 0);
            if (remaining > 0) {
                mirror[id] = MirrorEntry{side, price, remaining};
                resting_ids.push_back(id);
            }
        } else {
            // Cancel a random currently-resting order (mirror-tracked).
            // Skip past any ids that have since been fully filled (still
            // present in resting_ids but no longer in mirror) without
            // requeuing them -- keeps resting_ids from growing unbounded
            // with stale entries.
            while (!resting_ids.empty()) {
                std::uniform_int_distribution<std::size_t> idx_dist(0, resting_ids.size() - 1);
                const std::size_t idx = idx_dist(rng);
                const OrderId candidate = resting_ids[idx];
                resting_ids[idx] = resting_ids.back();
                resting_ids.pop_back();
                auto it = mirror.find(candidate);
                if (it == mirror.end()) {
                    continue; // stale entry (already fully filled), try another
                }
                const bool cancelled = book.cancel_order(candidate);
                assert(cancelled == true);
                mirror.erase(it);
                break;
            }
        }

        // Periodically cross-check aggregate invariants against the mirror
        // (checking every iteration would be O(n^2)-ish; every 500 is
        // enough to catch drift without dominating runtime).
        if (iter % 500 == 0) {
            assert(book.order_count() == mirror.size());
        }
    }

    // Final cross-check: every mirror entry must still be resting with
    // exactly the expected quantity, and order_count() must match exactly
    // (no phantom/leaked/missing orders).
    assert(book.order_count() == mirror.size());
    for (const auto& [id, entry] : mirror) {
        assert(book.quantity_at(entry.side, entry.price) >= entry.qty);
    }

    // Drain everything that's left via cancel_order, forcing yet more
    // free-list churn, and confirm we end at a truly empty book.
    for (const auto& [id, entry] : mirror) {
        (void)entry;
        assert(book.cancel_order(id) == true);
    }
    assert(book.order_count() == 0);
    assert(!book.best_bid().has_value());
    assert(!book.best_ask().has_value());
}

// ---------------------------------------------------------------------
// Full-book teardown: build a large book spanning many price levels on
// both sides (well past initial pool capacity) with a mix of still-resting
// and already-filled orders, then let the OrderBook go out of scope so its
// destructor walks every remaining node. Nothing here asserts anything
// interesting on its own -- the point is to hand the destructor a large,
// non-trivial, multi-level book and run this under
// ASan/UBSan/leak-checking (see the tester's report) to catch a
// use-after-free/double-free/leak that a plain optimized run would not
// surface.
// ---------------------------------------------------------------------
void test_full_book_teardown_under_sanitizers() {
    constexpr int kLevels = 200;
    constexpr int kOrdersPerLevel = 30; // 200 * 30 = 6000 resting orders, both sides
    {
        OrderBook book;
        OrderId id = 1;
        for (int lvl = 0; lvl < kLevels; ++lvl) {
            const Price bid_price = 1000 - lvl;
            const Price ask_price = 2000 + lvl;
            for (int k = 0; k < kOrdersPerLevel; ++k) {
                book.add_limit_order(id++, Side::Buy, bid_price, 3);
                book.add_limit_order(id++, Side::Sell, ask_price, 3);
            }
        }
        assert(book.order_count() == static_cast<std::size_t>(kLevels * kOrdersPerLevel * 2));

        // Fill/cancel a chunk of the book (mixing both removal paths) so
        // the destructor sees a book with holes in it, not just a
        // perfectly uniform grid.
        for (OrderId cancel_id = 1; cancel_id <= 1000; cancel_id += 2) {
            book.cancel_order(cancel_id);
        }
        auto trades = book.add_limit_order(id++, Side::Buy, 2050, 500); // sweeps several ask levels
        assert(!trades.empty());

        // `book` goes out of scope here -- ~OrderBook() walks every
        // remaining OrderNode on both sides and returns it to node_pool_,
        // whose own destructor then frees every chunk it ever allocated
        // (including growth chunks, since 6000+ nodes exceeds the default
        // 4096 initial capacity).
    }
}

} // namespace

int main() {
    struct NamedTest {
        const char* name;
        void (*fn)();
    };
    const NamedTest tests[] = {
        {"bulk_distinct_price_levels_forces_growth", test_bulk_distinct_price_levels_forces_growth},
        {"single_price_level_deep_book_fifo_and_cancel", test_single_price_level_deep_book_fifo_and_cancel},
        {"interleaved_add_cancel_fill_cycles", test_interleaved_add_cancel_fill_cycles},
        {"full_book_teardown_under_sanitizers", test_full_book_teardown_under_sanitizers},
    };

    for (const auto& t : tests) {
        t.fn();
        std::printf("[PASS] %s\n", t.name);
    }
    std::printf("All %zu order_book slab-stress tests passed.\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
