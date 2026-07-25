#pragma once

#include "exec_core/types.hpp"

namespace exec_core {

// A resting order sitting in the book at a given price level. `quantity` is
// the *remaining* (unfilled) quantity -- it is decremented in place as fills
// happen so the book never needs to reconstruct order state from a log.
struct RestingOrder {
    OrderId id;
    Price price;
    Quantity quantity;
    Sequence sequence; // arrival order, used for FIFO time priority within a price level
};

} // namespace exec_core
