# Build plan -- lowlatency-exec-core

Each phase is independently verifiable. Check off as completed.

- [x] Single-threaded matching engine (correctness first)
- [x] Lock-free ring buffer
- [x] Custom slab allocator
- [x] Cache-line alignment + core pinning
- [ ] SIMD price scans -- NEON path implemented + verified locally (this M2). AVX2 path
      implemented and compile-checked only; TODO: run on x86 CI (ubuntu-latest) to actually
      verify it -- this machine has no x86 hardware, so it cannot be checked done here.
- [ ] Latency-percentile harness (p50/p99/p99.9/p99.99), measured after each phase
