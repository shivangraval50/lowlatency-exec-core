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
4. **Cache-line alignment + core pinning** (done) -- audited `OrderBook`/`SlabAllocator`
   hot-path structs and deliberately did NOT cache-line-align them: they're only ever
   touched by the single matching-engine thread, so there's no false sharing to prevent,
   and padding would only bloat memory/cache footprint (`static_assert`s guard against this
   regressing later). The ring buffer's existing phase-2 padding is unchanged, now sourced
   from a shared `cache_line.hpp`. Core pinning is honestly platform-split: Linux uses real
   `pthread_setaffinity_np` (kernel-enforced, verified via `sched_getcpu()` on CI); macOS
   calls `thread_policy_set(THREAD_AFFINITY_POLICY)` but maps its real return code to an
   honest `BestEffortHint`/`Unsupported` outcome rather than claiming true pinning --
   measured on this M2, it actually returns `KERN_NOT_SUPPORTED`, i.e. no pinning at all.
5. **SIMD price scans** (NEON done and verified locally; AVX2 TODO -- see below) --
   `scan_quantity_at_or_better`, a standalone (not yet wired into `OrderBook`'s live
   matching path) primitive answering "how much resting quantity is at-or-better than a
   given limit price" over a flat price/quantity array, with scalar/NEON/AVX2
   implementations and platform dispatch. NEON cross-checked against the scalar reference
   with a 500-trial randomized property test plus fixed edge cases (empty, boundary ties,
   vector-width/remainder sizes) -- all pass on this M2. AVX2 only compile-checked
   (inspected generated assembly to confirm real `vpcmpgtq`/`vpaddq` etc. were emitted) --
   this machine has no x86 hardware to actually run it on, so it is NOT claimed as verified;
   PLAN.md leaves this phase unchecked pending a real run on x86 CI.
6. **Latency-percentile harness** (done) -- `measure_latency_ns` runs a caller-supplied
   workload for a warmup period (untimed, discarded) then a measured period, recording each
   iteration's `steady_clock` duration into a buffer allocated once up front (no in-loop
   allocation, so the measurement loop doesn't inject its own outliers); `compute_latency_stats`
   reduces the samples to min/p50/p90/p99/p99.9/p99.99/max/mean via the nearest-rank method.
   `src/bench_main.cpp` wires this up against real phase 1-5 code (`OrderBook::add_limit_order`
   resting and crossing, `OrderBook::cancel_order`, `SpscRingBuffer` push+pop, `scan_quantity_at_or_better`
   at two depths) and prints an honest results table -- see that file's header for exactly what
   each workload does and doesn't measure. While building this, the tester's synthetic-distribution
   tests caught a real bug: `nearest_rank_percentile` computed the rank via `double` arithmetic
   (`ceil((p/100.0) * N)`), which silently returned an off-by-one rank whenever the mathematically
   exact result was itself an integer (e.g. p99.9 at any N that's an exact multiple of 1000 --
   this includes `bench_main.cpp`'s own `kMeasuredIters=50000`), because literals like `99.9`
   aren't exactly representable in binary floating point. Fixed by switching the whole
   computation to exact integer arithmetic (round the percentage to hundredths-of-a-percent via
   `std::llround`, then ceiling-divide as an integer, widened to `unsigned __int128` to rule out
   overflow) -- verified against a hand-computed exact-integer oracle in `tests/test_latency_harness.cpp`.
   Running `exec_core_bench` once on this laptop is a sanity check that the harness works
   end-to-end against real code, NOT the authoritative benchmark run this README's Results table
   should be filled in from -- see bench_main.cpp's file header for why (single-machine noise, no
   real core pinning on this M2, sample counts too small for genuine p99.99 resolution).

## Results
| Metric | Value |
| ------ | ----- |
| -      | TODO -- not yet measured |

## Limitations / what's unrealistic
- Small scale by design; see note above.
- Single-threaded, single-process, in-memory only -- no persistence, no networking, no
  multi-symbol routing. This is the matching-engine core, not an exchange.
- No market/stop/iceberg order types yet; order modify is cancel + re-add, not in-place.
- AVX2 SIMD price-scan path is compile-checked only, not runtime-verified -- no x86
  hardware on this local machine. TODO: run on x86 CI (ubuntu-latest) before claiming it
  works; PLAN.md leaves that phase unchecked until then.
- Real CPU core pinning only happens on Linux. On this local machine (macOS/Apple Silicon
  M2) `thread_policy_set` returns `KERN_NOT_SUPPORTED` -- there is no working core-affinity
  mechanism to benchmark against locally; that part of the design is only actually verified
  on CI (`ubuntu-latest`).
- This machine's Xcode Command Line Tools has a broken/incomplete libc++ header install
  (missing `<vector>`, `<map>`, etc.). Local builds work around it by pointing at the
  full macOS SDK's own libc++ headers; this workaround is local-machine-only and is not
  baked into `CMakeLists.txt` or CI (CI builds cleanly on `ubuntu-latest`).
- The latency-percentile harness (`exec_core_bench`) has only been run as a local sanity check
  on this one laptop, a handful of times, not as a controlled benchmark (no isolated/dedicated
  machine, no repeated-runs-with-statistics methodology, no verification of `steady_clock`'s own
  call overhead). The Results table below stays "TODO" until a real, reviewed benchmark run
  happens -- a single noisy laptop run is not that.
- <!-- TODO: fill in as the build progresses. -->
