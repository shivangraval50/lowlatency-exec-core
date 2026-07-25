// Shared cache-line-size constant, used wherever this codebase needs to
// `alignas(...)` a hot-path structure to avoid false sharing.
//
// ---------------------------------------------------------------------------
// Why this is its own header (phase 4)
// ---------------------------------------------------------------------------
// Phase 2's ring_buffer.hpp originally defined `kCacheLineSize` locally,
// because the producer/consumer index padding it needs is inherent to that
// specific lock-free algorithm, not a "general alignment pass" concern (see
// the comment in that file). Phase 4 ("cache-line alignment") is exactly the
// general pass, and more than one header now wants the same constant, so it
// is pulled out here as the single source of truth; ring_buffer.hpp now
// includes this header instead of keeping its own copy. Nothing about
// ring_buffer.hpp's own padding scheme changes.
//
// ---------------------------------------------------------------------------
// Where the number comes from
// ---------------------------------------------------------------------------
// `std::hardware_destructive_interference_size` (C++17, <new>) is the
// standard-library-provided answer to "how far apart do two objects need to
// be to not share a cache line", when a standard library ships it. As of
// this writing, neither libc++ (macOS/arm64, this project's local dev
// machine) nor the libstdc++ shipped by ubuntu-latest's default GCC/Clang
// (this project's CI, see .github/workflows/ci.yml) define
// `__cpp_lib_hardware_interference_size` reliably enough to depend on
// portably, so both platforms fall back to a plain `64`, which is the
// correct line size for Apple Silicon (M-series) and for essentially all
// current x86-64 server/desktop parts. If a future target ever has a
// different real line size (e.g. some big.LITTLE ARM parts mix 64B/128B
// lines), this constant would need revisiting for that target -- it is not
// auto-detected at runtime.
#pragma once

#include <cstddef>

namespace exec_core {

#if defined(__cpp_lib_hardware_interference_size)
inline constexpr std::size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLineSize = 64;
#endif

} // namespace exec_core
