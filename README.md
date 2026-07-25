# lowlatency-exec-core

> Low-latency order-matching execution core in C++; verified by tooling; per-phase latency percentiles.

**Stack:** C++20 / CMake; ARM NEON locally, AVX2 + io_uring benchmarked on x86 CI

> Scale is deliberately small and hardware-constrained (built on an M2 / 8GB, no local GPU).
> The point is the mechanics and honest measurement, not raw scale.

## Problem
Matching engines sit on the hot path of every exchange/venue: every microsecond spent
adding, cancelling, or crossing an order is microseconds an HFT competitor doesn't spend.
This project builds one from scratch and measures, honestly, where the time actually goes
as complexity (lock-free structures, custom allocation, SIMD) is added -- rather than
asserting it's fast.

## Approach
Phased build, each phase independently correctness-checked before the next starts (see
PLAN.md):
1. **Single-threaded matching engine** (done) -- price-time priority limit order book.
   Bids/asks are `std::map<Price, std::list<RestingOrder>>` (map keeps best-price O(1) access,
   list preserves FIFO time priority per level and gives O(1) erase-by-iterator); an
   `unordered_map<OrderId, ...>` gives O(1)-average cancel lookup. Plain STL containers,
   no lock-free tricks or custom allocation yet -- correctness first.
2. **Lock-free ring buffer** (done) -- `SpscRingBuffer<T, Capacity>`, single-producer/
   single-consumer only (explicitly, not MPMC -- a correct MPMC needs a different
   algorithm). Power-of-2 capacity, acquire/release atomics (not seq_cst) with the
   producer/consumer counters each cache-line-padded to avoid false sharing. Verified
   under ThreadSanitizer with a real 2-thread producer/consumer stress test.
3. **Custom slab allocator** (done) -- `SlabAllocator<T>`, a fixed-block pool with an
   intrusive free list, pre-reserves blocks up front so `OrderBook`'s resting-order nodes
   (now an intrusive list, not `std::list`) are carved out of one arena instead of calling
   `new`/`delete` per order. Grows by one more chunk (logged via `growth_events()`) if the
   initial capacity guess is exceeded, rather than rejecting an order or corrupting memory.
   Verified under ASan+UBSan, including a stress test forcing real pool growth.
4. Cache-line alignment + core pinning (planned).
5. SIMD price scans (planned) -- NEON locally, AVX2 on x86 CI.
6. Latency-percentile harness (planned) -- p50/p99/p99.9/p99.99, re-measured after each
   phase so every optimization's actual effect (or lack of one) is on the record.

## Results
| Metric | Value |
| ------ | ----- |
| -      | TODO -- not yet measured |

## Limitations / what's unrealistic
- Small scale by design; see note above.
- Single-threaded, single-process, in-memory only -- no persistence, no networking, no
  multi-symbol routing. This is the matching-engine core, not an exchange.
- No market/stop/iceberg order types yet; order modify is cancel + re-add, not in-place.
- This machine's Xcode Command Line Tools has a broken/incomplete libc++ header install
  (missing `<vector>`, `<map>`, etc.). Local builds work around it by pointing at the
  full macOS SDK's own libc++ headers; this workaround is local-machine-only and is not
  baked into `CMakeLists.txt` or CI (CI builds cleanly on `ubuntu-latest`).
- <!-- TODO: fill in as the build progresses. -->
