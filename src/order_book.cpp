#include "exec_core/order_book.hpp"

#include <algorithm>
#include <new>
#include <stdexcept>

namespace exec_core {

OrderBook::~OrderBook() {
    const auto destroy_level = [this](PriceLevel& level) {
        OrderNode* node = level.head;
        while (node != nullptr) {
            OrderNode* next = node->next;
            node->~OrderNode();
            node_pool_.deallocate(node);
            node = next;
        }
    };
    for (auto& [price, level] : bids_) {
        destroy_level(level);
    }
    for (auto& [price, level] : asks_) {
        destroy_level(level);
    }
}

OrderBook::OrderNode* OrderBook::push_back(PriceLevel& level, const RestingOrder& order) {
    OrderNode* node = node_pool_.allocate();
    new (node) OrderNode{order, level.tail, nullptr};
    if (level.tail != nullptr) {
        level.tail->next = node;
    } else {
        level.head = node;
    }
    level.tail = node;
    return node;
}

void OrderBook::pop_front(PriceLevel& level) {
    OrderNode* node = level.head;
    level.head = node->next;
    if (level.head != nullptr) {
        level.head->prev = nullptr;
    } else {
        level.tail = nullptr;
    }
    node->~OrderNode();
    node_pool_.deallocate(node);
}

void OrderBook::erase(PriceLevel& level, OrderNode* node) {
    if (node->prev != nullptr) {
        node->prev->next = node->next;
    } else {
        level.head = node->next;
    }
    if (node->next != nullptr) {
        node->next->prev = node->prev;
    } else {
        level.tail = node->prev;
    }
    node->~OrderNode();
    node_pool_.deallocate(node);
}

Quantity OrderBook::sum_quantity(const PriceLevel& level) {
    Quantity total = 0;
    for (const OrderNode* node = level.head; node != nullptr; node = node->next) {
        total += node->order.quantity;
    }
    return total;
}

std::vector<OrderId> OrderBook::collect_ids(const PriceLevel& level) {
    std::vector<OrderId> ids;
    for (const OrderNode* node = level.head; node != nullptr; node = node->next) {
        ids.push_back(node->order.id);
    }
    return ids;
}

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
            PriceLevel& level = level_it->second;
            while (remaining > 0 && level.head != nullptr) {
                RestingOrder& resting = level.head->order;
                const Quantity traded = std::min(remaining, resting.quantity);
                trades.push_back(Trade{id, resting.id, side, level_price, traded, next_sequence_++});
                remaining -= traded;
                resting.quantity -= traded;
                if (resting.quantity == 0) {
                    locations_.erase(resting.id);
                    pop_front(level);
                }
            }
            if (level.head == nullptr) {
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
            PriceLevel& level = level_it->second;
            while (remaining > 0 && level.head != nullptr) {
                RestingOrder& resting = level.head->order;
                const Quantity traded = std::min(remaining, resting.quantity);
                trades.push_back(Trade{id, resting.id, side, level_price, traded, next_sequence_++});
                remaining -= traded;
                resting.quantity -= traded;
                if (resting.quantity == 0) {
                    locations_.erase(resting.id);
                    pop_front(level);
                }
            }
            if (level.head == nullptr) {
                bids_.erase(level_it);
            }
        }
    }

    if (remaining > 0) {
        const Sequence seq = next_sequence_++;
        if (side == Side::Buy) {
            PriceLevel& level = bids_[price];
            OrderNode* node = push_back(level, RestingOrder{id, price, remaining, seq});
            locations_.emplace(id, OrderLocation{side, price, node});
        } else {
            PriceLevel& level = asks_[price];
            OrderNode* node = push_back(level, RestingOrder{id, price, remaining, seq});
            locations_.emplace(id, OrderLocation{side, price, node});
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
        erase(level_it->second, loc.node);
        if (level_it->second.head == nullptr) {
            bids_.erase(level_it);
        }
    } else {
        auto level_it = asks_.find(loc.price);
        erase(level_it->second, loc.node);
        if (level_it->second.head == nullptr) {
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
    if (side == Side::Buy) {
        auto it = bids_.find(price);
        return it == bids_.end() ? 0 : sum_quantity(it->second);
    }
    auto it = asks_.find(price);
    return it == asks_.end() ? 0 : sum_quantity(it->second);
}

std::size_t OrderBook::order_count() const {
    return locations_.size();
}

std::vector<std::pair<Price, Quantity>> OrderBook::bid_levels() const {
    std::vector<std::pair<Price, Quantity>> levels;
    levels.reserve(bids_.size());
    for (const auto& [price, level] : bids_) {
        levels.emplace_back(price, sum_quantity(level));
    }
    return levels;
}

std::vector<std::pair<Price, Quantity>> OrderBook::ask_levels() const {
    std::vector<std::pair<Price, Quantity>> levels;
    levels.reserve(asks_.size());
    for (const auto& [price, level] : asks_) {
        levels.emplace_back(price, sum_quantity(level));
    }
    return levels;
}

std::vector<OrderId> OrderBook::orders_at(Side side, Price price) const {
    if (side == Side::Buy) {
        auto it = bids_.find(price);
        return it == bids_.end() ? std::vector<OrderId>{} : collect_ids(it->second);
    }
    auto it = asks_.find(price);
    return it == asks_.end() ? std::vector<OrderId>{} : collect_ids(it->second);
}

} // namespace exec_core
