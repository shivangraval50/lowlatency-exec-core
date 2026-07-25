#pragma once

#include "exec_core/types.hpp"

namespace exec_core {

// A single execution produced by matching an incoming ("aggressor") order
// against a resting order already in the book. Trades always execute at the
// resting order's price (price-time priority: the order that was already in
// the book gets its price honored).
struct Trade {
    OrderId aggressor_order_id;
    OrderId resting_order_id;
    Side aggressor_side; // side of the order that arrived and caused the trade
    Price price;
    Quantity quantity;
    Sequence sequence; // monotonic, assigned in the order trades occurred
};

} // namespace exec_core
