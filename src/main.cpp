// Minimal manual sanity-check driver for the phase-1 order book -- not the
// authoritative test suite (that lives under tests/), just something to eyeball
// while developing. Run: ./exec_core_demo
#include <cstdio>

#include "exec_core/order_book.hpp"

using namespace exec_core;

namespace {

const char* side_name(Side s) { return s == Side::Buy ? "BUY" : "SELL"; }

void print_trades(const std::vector<Trade>& trades) {
    for (const Trade& t : trades) {
        std::printf("  TRADE aggressor=%llu (%s) resting=%llu price=%lld qty=%lld seq=%llu\n",
                    static_cast<unsigned long long>(t.aggressor_order_id), side_name(t.aggressor_side),
                    static_cast<unsigned long long>(t.resting_order_id), static_cast<long long>(t.price),
                    static_cast<long long>(t.quantity), static_cast<unsigned long long>(t.sequence));
    }
}

} // namespace

int main() {
    OrderBook book;

    std::printf("-- resting orders, no cross --\n");
    print_trades(book.add_limit_order(1, Side::Buy, 100, 10));
    print_trades(book.add_limit_order(2, Side::Sell, 105, 5));
    std::printf("best_bid=%lld best_ask=%lld order_count=%zu\n",
                static_cast<long long>(*book.best_bid()), static_cast<long long>(*book.best_ask()),
                book.order_count());

    std::printf("-- crossing order, partial fill --\n");
    print_trades(book.add_limit_order(3, Side::Buy, 105, 8)); // should fill order 2 (5) then rest 3 at 105
    std::printf("best_bid=%lld best_ask=%s order_count=%zu\n", static_cast<long long>(*book.best_bid()),
                book.best_ask() ? "set" : "none", book.order_count());

    std::printf("-- cancel --\n");
    std::printf("cancel(1) = %s\n", book.cancel_order(1) ? "true" : "false");
    std::printf("cancel(999) = %s\n", book.cancel_order(999) ? "true" : "false");
    std::printf("order_count=%zu\n", book.order_count());

    return 0;
}
