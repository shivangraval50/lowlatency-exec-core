# Build plan -- lowlatency-exec-core

Each phase is independently verifiable. Check off as completed.

- [ ] Single-threaded matching engine (correctness first)
- [ ] Lock-free ring buffer
- [ ] Custom slab allocator
- [ ] Cache-line alignment + core pinning
- [ ] SIMD price scans
- [ ] Latency-percentile harness (p50/p99/p99.9/p99.99), measured after each phase
