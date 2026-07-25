cmake_minimum_required(VERSION 3.16)
project(lowlatency_exec_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Needed by anything using std::thread (phase 2's SPSC ring buffer test will).
find_package(Threads REQUIRED)

# Core matching-engine library (phase 1: single-threaded, correctness first).
# core_affinity.cpp (phase 4) has real per-platform implementations (Linux:
# sched_setaffinity; macOS: best-effort thread_policy_set) behind
# __linux__/__APPLE__, so it's a regular translation unit here, not
# header-only like ring_buffer.hpp/slab_allocator.hpp (which are templates).
# latency_harness.cpp (phase 6) holds the non-template pieces of
# latency_harness.hpp (percentile computation, pretty-printing) -- same
# split as core_affinity.hpp/.cpp: the templated measure_latency_ns() stays
# header-only (it must, since Fn is a caller-supplied type), but
# compute_latency_stats()/print_latency_row() are ordinary functions and so
# get a real translation unit like the rest of this project's non-template
# code.
add_library(exec_core
  src/order_book.cpp
  src/core_affinity.cpp
  src/price_scan.cpp
  src/latency_harness.cpp
)
target_include_directories(exec_core PUBLIC include)
target_compile_options(exec_core PRIVATE -Wall -Wextra -Wpedantic)
# core_affinity.cpp's Linux path calls pthread_setaffinity_np(), so the
# library itself needs to link pthread there (harmless/no-op on macOS).
target_link_libraries(exec_core PUBLIC Threads::Threads)

# Phase 5 (SIMD price scans, src/price_scan.cpp): the ARM NEON path there
# compiles automatically on AArch64 (NEON is baseline on ARM64, no special
# flag needed), but the AVX2 path needs an explicit -mavx2 -- AVX2 is NOT
# implied merely by targeting x86_64. Apply that flag ONLY to this one
# source file, and ONLY when actually building for an x86 target, so this
# doesn't try (and fail) to pass an x86-only flag to arm64/AArch64 clang on
# this project's local dev machine (M2). CI (.github/workflows/ci.yml) runs
# on ubuntu-latest, which is real x86_64 hardware with AVX2 -- that's where
# this flag actually matters and where the AVX2 path gets compiled and
# tested. It has NOT been runtime-verified on this local (ARM) machine; see
# price_scan.hpp's file header for the honest status of each path.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
  set_source_files_properties(src/price_scan.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

# Manual sanity-check driver (not the authoritative test suite).
add_executable(exec_core_demo src/main.cpp)
target_link_libraries(exec_core_demo PRIVATE exec_core)

# Phase 6 latency-percentile benchmark driver (src/bench_main.cpp). Links
# against the SAME exec_core library as everything else (Release by default,
# per CMAKE_BUILD_TYPE above) -- this is a real runnable program, not a
# results file; see that file's header for how to read/not over-trust its
# output, and README.md's Results section for the actual reviewed numbers.
add_executable(exec_core_bench src/bench_main.cpp)
target_link_libraries(exec_core_bench PRIVATE exec_core)
target_compile_options(exec_core_bench PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

# Placeholder smoke test (kept green until real tests land).
add_executable(smoke_test tests/test_smoke.cpp)
add_test(NAME smoke_test COMMAND smoke_test)

# Test binaries rely on assert() for correctness checking, so they must never
# be compiled with NDEBUG defined -- regardless of CMAKE_BUILD_TYPE (which
# defaults to Release above and would otherwise silently strip every
# assert(), turning "all tests passed" into a false positive). -UNDEBUG is a
# compiler *option* (not a -D definition), so it must go through
# target_compile_options; it is appended after CXX_FLAGS_<CONFIG> on the
# command line and so wins over an earlier -DNDEBUG from the Release/
# RelWithDebInfo build-type flags.
target_compile_options(smoke_test PRIVATE -UNDEBUG)

# Real order-book tests are added by the tester phase; wire them in
# automatically once the file exists so this CMakeLists doesn't need to
# change again.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_order_book.cpp)
  add_executable(order_book_tests tests/test_order_book.cpp)
  target_link_libraries(order_book_tests PRIVATE exec_core)
  target_compile_options(order_book_tests PRIVATE -Wall -Wextra -UNDEBUG)
  add_test(NAME order_book_tests COMMAND order_book_tests)
endif()

# Phase 2's ring_buffer.hpp is header-only (include/ is already a public
# include dir of exec_core above), so no new library target is needed -- just
# wire in the test executable once the tester phase adds it. Links against
# Threads::Threads because a real test needs actual concurrent
# producer/consumer std::thread's, not just single-threaded API checks.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_ring_buffer.cpp)
  add_executable(ring_buffer_tests tests/test_ring_buffer.cpp)
  target_link_libraries(ring_buffer_tests PRIVATE exec_core Threads::Threads)
  target_compile_options(ring_buffer_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME ring_buffer_tests COMMAND ring_buffer_tests)
endif()

# Phase 3's slab_allocator.hpp is header-only, same as ring_buffer.hpp above
# -- just wire in the test executable once the tester phase adds it.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_slab_allocator.cpp)
  add_executable(slab_allocator_tests tests/test_slab_allocator.cpp)
  target_link_libraries(slab_allocator_tests PRIVATE exec_core)
  target_compile_options(slab_allocator_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME slab_allocator_tests COMMAND slab_allocator_tests)
endif()

# Phase 3 order-book/slab integration stress tests (large resting-order
# counts, interleaved add/cancel/fill, full-book teardown) -- see
# tests/test_order_book_slab_stress.cpp. Meant to be run under
# ASan/UBSan too; this target itself just builds a plain instrumented-or-not
# binary depending on whatever CMAKE_CXX_FLAGS the invoking configure passed.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_order_book_slab_stress.cpp)
  add_executable(order_book_slab_stress_tests tests/test_order_book_slab_stress.cpp)
  target_link_libraries(order_book_slab_stress_tests PRIVATE exec_core)
  target_compile_options(order_book_slab_stress_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME order_book_slab_stress_tests COMMAND order_book_slab_stress_tests)
endif()

# Phase 4's core_affinity.hpp/.cpp: real, verifiable pinning on Linux
# (sched_getcpu() after pthread_setaffinity_np() can confirm it actually
# took effect -- this is what runs on CI, ubuntu-latest); honestly
# best-effort/unverifiable on macOS (see core_affinity.hpp's file header --
# Apple Silicon does not honor THREAD_AFFINITY_POLICY at all). The test
# itself (added by the tester phase) is expected to assert the strong
# Linux guarantee only where the platform actually provides it, and assert
# only "did not crash / returned an honest status" on macOS.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_core_affinity.cpp)
  add_executable(core_affinity_tests tests/test_core_affinity.cpp)
  target_link_libraries(core_affinity_tests PRIVATE exec_core Threads::Threads)
  target_compile_options(core_affinity_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME core_affinity_tests COMMAND core_affinity_tests)
endif()

# Phase 5's price_scan.hpp/.cpp: scalar reference implementation always
# compiles/runs; NEON path compiles+runs here (AArch64, this project's local
# machine); AVX2 path only compiles (and only gets exercised) on an x86
# target -- i.e. CI (ubuntu-latest), via the -mavx2 flag applied to
# src/price_scan.cpp above. Wired in automatically once the tester phase
# adds the test file, same pattern as every other phase above.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_price_scan.cpp)
  add_executable(price_scan_tests tests/test_price_scan.cpp)
  target_link_libraries(price_scan_tests PRIVATE exec_core)
  target_compile_options(price_scan_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME price_scan_tests COMMAND price_scan_tests)
endif()

# Phase 6's latency_harness.hpp/.cpp: percentile-computation correctness
# (nearest_rank_percentile/compute_latency_stats, against hand-computable
# synthetic distributions) plus structural invariants on measure_latency_ns
# itself. Uses fork()/exec()/waitpid() (POSIX) for one self-check (confirm
# assert() is genuinely live in this exact binary) and std::this_thread::
# sleep_for for another (confirm warmup iterations are excluded) -- links
# Threads::Threads for the latter, same as ring_buffer_tests above.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_latency_harness.cpp)
  add_executable(latency_harness_tests tests/test_latency_harness.cpp)
  target_link_libraries(latency_harness_tests PRIVATE exec_core Threads::Threads)
  target_compile_options(latency_harness_tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME latency_harness_tests COMMAND latency_harness_tests)
endif()
