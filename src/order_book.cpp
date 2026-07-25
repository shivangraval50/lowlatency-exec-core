#include "exec_core/order_book.hpp"

#include <algorithm>
#include <stdexcept>

namespace exec_core {

std::vector<Trade> OrderBook::add_limit_order(OrderId id, Side side, Price price, Quantity quantity) {
    if (quantity <= 0) {
        throw std::invalid_argument("OrderBook::add_limit_order: quantity must be positive");
    }
    if (locations_.find(id) != locations_.end()) {
        throw std::invalid_argument("OrderBook::add_limit_order: duplicate order id");
    }

    std::vector<Trade> trades;
    Quantity remaining = quantity;

    if (side == Side::Buy) {
        // Match against resting asks while the best ask is at or below our
        // limit price (i.e. we're willing to pay at least that much).
        while (remaining > 0 && !asks_.empty()) {
            auto level_it = asks_.begin();
            const Price level_price = level_it->first;
            if (level_price > price) {
                break; // best ask too expensive, nothing more to cross
            }
            LevelOrders& orders = level_it->second;
            while (remaining > 0 && !orders.empty()) {
                RestingOrder& resting = orders.front();
                const Quantity traded = std::min(remaining, resting.quantity);
                trades.push_back(Trade{id, resting.id, side, level_price, traded, next_sequence_++});
                remaining -= traded;
                resting.quantity -= traded;
                if (resting.quantity == 0) {
                    locations_.erase(resting.id);
                    orders.pop_front();
                }
            }
            if (orders.empty()) {
                asks_.erase(level_it);
            }
        }
    } else {
        // Match against resting bids while the best bid is at or above our
        // limit price (i.e. someone's willing to pay at least what we ask).
        while (remaining > 0 && !bids_.empty()) {
            auto level_it = bids_.begin();
            const Price level_price = level_it->first;
            if (level_price < price) {
                break; // best bid too low, nothing more to cross
            }
            LevelOrders& orders = level_it->second;
            while (remaining > 0 && !orders.empty()) {
                RestingOrder& resting = orders.front();
                const Quantity traded = std::min(remaining, resting.quantity);
                trades.push_back(Trade{id, resting.id, side, level_price, traded, next_sequence_++});
                remaining -= traded;
                resting.quantity -= traded;
                if (resting.quantity == 0) {
                    locations_.erase(resting.id);
                    orders.pop_front();
                }
            }
            if (orders.empty()) {
                bids_.erase(level_it);
            }
        }
    }

    if (remaining > 0) {
        const Sequence seq = next_sequence_++;
        if (side == Side::Buy) {
            LevelOrders& orders = bids_[price];
            orders.push_back(RestingOrder{id, price, remaining, seq});
            locations_.emplace(id, OrderLocation{side, price, std::prev(orders.end())});
        } else {
            LevelOrders& orders = asks_[price];
            orders.push_back(RestingOrder{id, price, remaining, seq});
            locations_.emplace(id, OrderLocation{side, price, std::prev(orders.end())});
        }
    }

    return trades;
}

bool OrderBook::cancel_order(OrderId id) {
    auto loc_it = locations_.find(id);
    if (loc_it == locations_.end()) {
        return false;
    }
    const OrderLocation loc = loc_it->second;
    locations_.erase(loc_it);

    if (loc.side == Side::Buy) {
        auto level_it = bids_.find(loc.price);
        level_it->second.erase(loc.it);
        if (level_it->second.empty()) {
            bids_.erase(level_it);
        }
    } else {
        auto level_it = asks_.find(loc.price);
        level_it->second.erase(loc.it);
        if (level_it->second.empty()) {
            asks_.erase(level_it);
        }
    }
    return true;
}

std::optional<Price> OrderBook::best_bid() const {
    if (bids_.empty()) {
        return std::nullopt;
    }
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const {
    if (asks_.empty()) {
        return std::nullopt;
    }
    return asks_.begin()->first;
}

Quantity OrderBook::quantity_at(Side side, Price price) const {
    const auto sum = [](const LevelOrders& orders) {
        Quantity total = 0;
        for (const RestingOrder& o : orders) {
            total += o.quantity;
        }
        return total;
    };
    if (side == Side::Buy) {
        auto it = bids_.find(price);
        return it == bids_.end() ? 0 : sum(it->second);
    }
    auto it = asks_.find(price);
    return it == asks_.end() ? 0 : sum(it->second);
}

std::size_t OrderBook::order_count() const {
    return locations_.size();
}

std::vector<std::pair<Price, Quantity>> OrderBook::bid_levels() const {
    std::vector<std::pair<Price, Quantity>> levels;
    levels.reserve(bids_.size());
    for (const auto& [price, orders] : bids_) {
        Quantity total = 0;
        for (const RestingOrder& o : orders) {
            total += o.quantity;
        }
        levels.emplace_back(price, total);
    }
    return levels;
}

std::vector<std::pair<Price, Quantity>> OrderBook::ask_levels() const {
    std::vector<std::pair<Price, Quantity>> levels;
    levels.reserve(asks_.size());
    for (const auto& [price, orders] : asks_) {
        Quantity total = 0;
        for (const RestingOrder& o : orders) {
            total += o.quantity;
        }
        levels.emplace_back(price, total);
    }
    return levels;
}

std::vector<OrderId> OrderBook::orders_at(Side side, Price price) const {
    std::vector<OrderId> ids;
    const auto collect = [&ids](const LevelOrders& orders) {
        for (const RestingOrder& o : orders) {
            ids.push_back(o.id);
        }
    };
    if (side == Side::Buy) {
        auto it = bids_.find(price);
        if (it != bids_.end()) {
            collect(it->second);
        }
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end()) {
            collect(it->second);
        }
    }
    return ids;
}

} // namespace exec_core
