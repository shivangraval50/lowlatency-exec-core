// Basic value types shared across the matching engine.
//
// Prices and quantities are plain integers (ticks / lots), never floating
// point -- floating point price comparisons are a classic source of subtle
// matching bugs, and real venues quote in integer ticks anyway.
#pragma once

#include <cstdint>

namespace exec_core {

enum class Side : std::uint8_t {
    Buy,
    Sell,
};

using OrderId = std::uint64_t;
using Price = std::int64_t;     // integer ticks
using Quantity = std::int64_t;  // integer lots/shares
using Sequence = std::uint64_t; // monotonic arrival/event counter, used for
                                 // time priority and stable trade ordering

} // namespace exec_core
