// Real correctness tests for the phase-1 single-threaded order book.
//
// Plain assert()-based tests (no framework), consistent with
// tests/test_smoke.cpp. Each test is a free function; main() runs them all
// and prints a summary. Any failed assert aborts immediately with a file/line
// pointing at the exact failing check, which is enough to diagnose failures
// without a framework.

#include <cassert>
#include <cstdio>
#include <stdexcept>

#include "exec_core/order_book.hpp"

using namespace exec_core;

namespace {

// ---------------------------------------------------------------------
// Basic add + exact-price cross.
// ---------------------------------------------------------------------
void test_basic_cross_exact_price() {
    OrderBook book;

    // Resting sell at 100 x 10.
    auto t1 = book.add_limit_order(1, Side::Sell, 100, 10);
    assert(t1.empty());
    assert(book.best_ask().has_value() && *book.best_ask() == 100);
    assert(!book.best_bid().has_value());

    // Incoming buy at 100 x 10 -- exact match, should fully cross and leave
    // an empty book.
    auto t2 = book.add_limit_order(2, Side::Buy, 100, 10);
    assert(t2.size() == 1);
    const Trade& trade = t2.front();
    assert(trade.aggressor_order_id == 2);
    assert(trade.resting_order_id == 1);
    assert(trade.aggressor_side == Side::Buy);
    assert(trade.price == 100);
    assert(trade.quantity == 10);

    assert(!book.best_bid().has_value());
    assert(!book.best_ask().has_value());
    assert(book.order_count() == 0);
    assert(book.quantity_at(Side::Sell, 100) == 0);
}

// ---------------------------------------------------------------------
// Partial fill: resting order larger than incoming order.
// ---------------------------------------------------------------------
void test_partial_fill_resting_larger() {
    OrderBook book;

    book.add_limit_order(1, Side::Sell, 50, 20); // resting: 20 @ 50

    auto trades = book.add_limit_order(2, Side::Buy, 50, 7); // incoming: 7 @ 50
    assert(trades.size() == 1);
    assert(trades.front().quantity == 7);
    assert(trades.front().resting_order_id == 1);
    assert(trades.front().price == 50);

    // Remainder of the resting order (13) should still be in the book, and
    // the incoming order should be fully filled and NOT resting.
    assert(book.order_count() == 1);
    assert(book.best_ask().has_value() && *book.best_ask() == 50);
    assert(book.quantity_at(Side::Sell, 50) == 13);
    auto ids = book.orders_at(Side::Sell, 50);
    assert(ids.size() == 1 && ids.front() == 1);

    // The incoming order id (2) should not be cancellable / resting.
    assert(book.cancel_order(2) == false);
}

// ---------------------------------------------------------------------
// FIFO / time priority: two resting orders at same price, earlier one
// fills first.
// ---------------------------------------------------------------------
void test_fifo_time_priority() {
    OrderBook book;

    book.add_limit_order(1, Side::Sell, 100, 5);  // arrives first
    book.add_limit_order(2, Side::Sell, 100, 5);  // arrives second, same price
    assert(book.quantity_at(Side::Sell, 100) == 10);

    // Incoming buy for 5 should fill order 1 entirely (oldest first), not
    // order 2, and not split across both.
    auto trades = book.add_limit_order(3, Side::Buy, 100, 5);
    assert(trades.size() == 1);
    assert(trades.front().resting_order_id == 1);
    assert(trades.front().quantity == 5);

    // Order 1 is gone, order 2 still resting untouched.
    assert(book.cancel_order(1) == false);
    assert(book.order_count() == 1);
    auto ids = book.orders_at(Side::Sell, 100);
    assert(ids.size() == 1 && ids.front() == 2);

    // Now send a buy for 3 -- should start eating into order 2 (partial).
    auto trades2 = book.add_limit_order(4, Side::Buy, 100, 3);
    assert(trades2.size() == 1);
    assert(trades2.front().resting_order_id == 2);
    assert(trades2.front().quantity == 3);
    assert(book.quantity_at(Side::Sell, 100) == 2);
}

// ---------------------------------------------------------------------
// Multi-level sweep: incoming order crosses multiple price levels in
// correct price order (best price first).
// ---------------------------------------------------------------------
void test_multi_level_sweep() {
    OrderBook book;

    // Three ask levels: 100 (qty 5), 101 (qty 5), 102 (qty 5).
    book.add_limit_order(1, Side::Sell, 101, 5);
    book.add_limit_order(2, Side::Sell, 100, 5); // best (lowest) ask
    book.add_limit_order(3, Side::Sell, 102, 5);

    assert(book.best_ask().has_value() && *book.best_ask() == 100);

    // Incoming aggressive buy at 102 for 12 should sweep 100 -> 101 -> 102
    // in that order, leaving 3 resting at 102.
    auto trades = book.add_limit_order(4, Side::Buy, 102, 12);
    assert(trades.size() == 3);
    assert(trades[0].price == 100 && trades[0].resting_order_id == 2 && trades[0].quantity == 5);
    assert(trades[1].price == 101 && trades[1].resting_order_id == 1 && trades[1].quantity == 5);
    assert(trades[2].price == 102 && trades[2].resting_order_id == 3 && trades[2].quantity == 2);

    // Trade sequence numbers should be strictly increasing (stable global
    // ordering).
    assert(trades[0].sequence < trades[1].sequence);
    assert(trades[1].sequence < trades[2].sequence);

    // Level 102 now has 3 remaining (5 - 2), and order 4 (the aggressor,
    // fully filled with 12) should not rest.
    assert(book.quantity_at(Side::Sell, 102) == 3);
    assert(book.quantity_at(Side::Sell, 100) == 0);
    assert(book.quantity_at(Side::Sell, 101) == 0);
    assert(book.best_ask().has_value() && *book.best_ask() == 102);
    assert(book.cancel_order(4) == false); // fully filled, nothing resting
    assert(book.order_count() == 1);       // just order 3's remainder
}

// A sweep where the incoming order's limit price stops the sweep partway
// through the levels available (limit price doesn't cross the top level).
void test_sweep_respects_limit_price() {
    OrderBook book;

    book.add_limit_order(1, Side::Sell, 100, 5);
    book.add_limit_order(2, Side::Sell, 101, 5);
    book.add_limit_order(3, Side::Sell, 102, 5);

    // Buy limited to 101 should only cross 100 and 101, never touch 102.
    auto trades = book.add_limit_order(4, Side::Buy, 101, 20);
    assert(trades.size() == 2);
    assert(trades[0].price == 100);
    assert(trades[1].price == 101);

    // Remainder (20 - 10 = 10) rests as a new bid at 101; level 102 untouched.
    assert(book.best_bid().has_value() && *book.best_bid() == 101);
    assert(book.quantity_at(Side::Buy, 101) == 10);
    assert(book.quantity_at(Side::Sell, 102) == 5);
    assert(book.best_ask().has_value() && *book.best_ask() == 102);
}

// ---------------------------------------------------------------------
// Cancel: removed from book and no longer matchable. Cancel of a
// nonexistent / already-canceled order returns false (per header contract).
// ---------------------------------------------------------------------
void test_cancel_behavior() {
    OrderBook book;

    book.add_limit_order(1, Side::Buy, 90, 10);
    assert(book.order_count() == 1);

    // Cancel of a nonexistent id.
    assert(book.cancel_order(42) == false);

    // Successful cancel.
    assert(book.cancel_order(1) == true);
    assert(book.order_count() == 0);
    assert(!book.best_bid().has_value());
    assert(book.quantity_at(Side::Buy, 90) == 0);

    // Cancel of the same id again -- already removed, should be false, not
    // throw.
    assert(book.cancel_order(1) == false);

    // A cancelled order is no longer matchable: an incoming sell at 90
    // should find nothing to cross and simply rest.
    auto trades = book.add_limit_order(2, Side::Sell, 90, 10);
    assert(trades.empty());
    assert(book.best_ask().has_value() && *book.best_ask() == 90);

    // Cancelling a resting order removes only that order, leaving siblings
    // at the same price level intact.
    book.add_limit_order(3, Side::Sell, 90, 5);
    assert(book.quantity_at(Side::Sell, 90) == 15);
    assert(book.cancel_order(2) == true);
    assert(book.quantity_at(Side::Sell, 90) == 5);
    auto ids = book.orders_at(Side::Sell, 90);
    assert(ids.size() == 1 && ids.front() == 3);

    // Since order id 1 was fully cancelled (and thus no longer tracked),
    // reusing that id should be allowed (duplicate check is about ids
    // currently resting in the book, not the id space historically).
    bool reuse_threw = false;
    try {
        book.add_limit_order(1, Side::Buy, 80, 3);
    } catch (const std::invalid_argument&) {
        reuse_threw = true;
    }
    assert(!reuse_threw);
    assert(book.order_count() == 2); // ids 3 and 1
}

// ---------------------------------------------------------------------
// Invalid input: non-positive quantity and duplicate order id.
// ---------------------------------------------------------------------
void test_invalid_input() {
    OrderBook book;

    // Zero quantity throws.
    bool threw_zero = false;
    try {
        book.add_limit_order(1, Side::Buy, 100, 0);
    } catch (const std::invalid_argument&) {
        threw_zero = true;
    }
    assert(threw_zero);
    assert(book.order_count() == 0); // rejected, nothing left behind

    // Negative quantity throws.
    bool threw_negative = false;
    try {
        book.add_limit_order(2, Side::Sell, 100, -5);
    } catch (const std::invalid_argument&) {
        threw_negative = true;
    }
    assert(threw_negative);
    assert(book.order_count() == 0);

    // Duplicate id while the original is still resting throws, and does not
    // corrupt the existing resting order.
    book.add_limit_order(10, Side::Buy, 100, 5);
    bool threw_duplicate = false;
    try {
        book.add_limit_order(10, Side::Buy, 101, 3);
    } catch (const std::invalid_argument&) {
        threw_duplicate = true;
    }
    assert(threw_duplicate);
    assert(book.order_count() == 1);
    assert(book.quantity_at(Side::Buy, 100) == 5); // untouched by the rejected duplicate
    assert(book.quantity_at(Side::Buy, 101) == 0);

    // Duplicate id check also applies across sides (id namespace is global,
    // not per-side).
    bool threw_duplicate_other_side = false;
    try {
        book.add_limit_order(10, Side::Sell, 100, 3);
    } catch (const std::invalid_argument&) {
        threw_duplicate_other_side = true;
    }
    assert(threw_duplicate_other_side);
    assert(book.order_count() == 1);
}

// ---------------------------------------------------------------------
// best_bid / best_ask correctness after adds/fills/cancels.
// ---------------------------------------------------------------------
void test_best_bid_ask_tracking() {
    OrderBook book;

    assert(!book.best_bid().has_value());
    assert(!book.best_ask().has_value());

    book.add_limit_order(1, Side::Buy, 99, 10);
    assert(*book.best_bid() == 99);

    // A higher bid becomes the new best bid.
    book.add_limit_order(2, Side::Buy, 100, 5);
    assert(*book.best_bid() == 100);

    // A lower bid does not change best bid.
    book.add_limit_order(3, Side::Buy, 95, 5);
    assert(*book.best_bid() == 100);

    // Cancel the best bid -- best bid should fall back to the next-highest
    // level (99), not disappear entirely and not skip to 95.
    assert(book.cancel_order(2) == true);
    assert(*book.best_bid() == 99);

    // Ask side, mirrored: lowest ask wins.
    book.add_limit_order(4, Side::Sell, 110, 5);
    assert(*book.best_ask() == 110);
    book.add_limit_order(5, Side::Sell, 108, 5);
    assert(*book.best_ask() == 108);
    book.add_limit_order(6, Side::Sell, 120, 5);
    assert(*book.best_ask() == 108);

    // Fully filling the best ask should advance best_ask to the next level.
    auto trades = book.add_limit_order(7, Side::Buy, 108, 5);
    assert(trades.size() == 1);
    assert(*book.best_ask() == 110);

    // Draining the whole book (bids and asks) should return best_bid/best_ask
    // to nullopt.
    assert(book.cancel_order(1) == true);
    assert(book.cancel_order(3) == true);
    assert(!book.best_bid().has_value());

    assert(book.cancel_order(4) == true);
    assert(book.cancel_order(6) == true);
    assert(!book.best_ask().has_value());
    assert(book.order_count() == 0);
}

} // namespace

int main() {
    struct NamedTest {
        const char* name;
        void (*fn)();
    };
    const NamedTest tests[] = {
        {"basic_cross_exact_price", test_basic_cross_exact_price},
        {"partial_fill_resting_larger", test_partial_fill_resting_larger},
        {"fifo_time_priority", test_fifo_time_priority},
        {"multi_level_sweep", test_multi_level_sweep},
        {"sweep_respects_limit_price", test_sweep_respects_limit_price},
        {"cancel_behavior", test_cancel_behavior},
        {"invalid_input", test_invalid_input},
        {"best_bid_ask_tracking", test_best_bid_ask_tracking},
    };

    for (const auto& t : tests) {
        t.fn();
        std::printf("[PASS] %s\n", t.name);
    }
    std::printf("All %zu order_book tests passed.\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
