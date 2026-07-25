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
add_library(exec_core
  src/order_book.cpp
  src/core_affinity.cpp
)
target_include_directories(exec_core PUBLIC include)
target_compile_options(exec_core PRIVATE -Wall -Wextra -Wpedantic)
# core_affinity.cpp's Linux path calls pthread_setaffinity_np(), so the
# library itself needs to link pthread there (harmless/no-op on macOS).
target_link_libraries(exec_core PUBLIC Threads::Threads)

# Manual sanity-check driver (not the authoritative test suite).
add_executable(exec_core_demo src/main.cpp)
target_link_libraries(exec_core_demo PRIVATE exec_core)

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
