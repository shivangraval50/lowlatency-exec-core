cmake_minimum_required(VERSION 3.16)
project(lowlatency_exec_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Core matching-engine library (phase 1: single-threaded, correctness first).
add_library(exec_core
  src/order_book.cpp
)
target_include_directories(exec_core PUBLIC include)
target_compile_options(exec_core PRIVATE -Wall -Wextra -Wpedantic)

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
