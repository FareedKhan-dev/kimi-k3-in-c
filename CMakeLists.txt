# Kimi K3 inference engine.
#
# The Makefile is the reference build and what CI uses. This exists for IDE integration
# and for platforms where GNU make is awkward. Both produce the same binary.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.16)

project(kimi-k3-in-c
        VERSION 0.1.0
        DESCRIPTION "Kimi K3 inference engine in portable C"
        LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -march=native gives a real speedup on the expert matmuls but produces a binary tied to
# the build machine. Off by default so a packaged build is portable.
option(K3_NATIVE_ARCH "Optimise for the building CPU (-march/-mcpu=native)" OFF)
option(K3_SANITIZE    "Build with ASan + UBSan"                        OFF)

find_package(OpenMP)
find_package(Threads REQUIRED)

add_library(k3_flags INTERFACE)
target_compile_options(k3_flags INTERFACE
  -Wall -Wextra
  # Not cosmetic: weight pointers are `const void *`, and pointer arithmetic on void is a
  # GNU extension that strides by one byte. Without this the mistake compiles silently.
  -Wpointer-arith
  -Wshadow -Wvla
  -Wno-unused-parameter
  # Keeps floating-point results reproducible across compilers by disabling automatic FMA
  # contraction; the op tests compare against a reference at a fixed tolerance.
  -ffp-contract=off)

if(K3_NATIVE_ARCH)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    # Apple Silicon / AArch64 uses -mcpu rather than the x86 -march spelling.
    target_compile_options(k3_flags INTERFACE -mcpu=native)
  else()
    target_compile_options(k3_flags INTERFACE -march=native)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  # Documented x86 baseline. The engine assumes AVX2 + FMA are available.
  target_compile_options(k3_flags INTERFACE -mavx2 -mfma)
endif()

# --------------------------------------------------------------------------- engine --
add_library(k3 STATIC
  src/core/k3_ops.c
  src/io/k3_st.c
  src/io/k3_load.c
  src/io/k3_trunk.c
  src/cache/k3_cache.c
  src/model/k3_bind.c)

target_include_directories(k3 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/k3
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cache
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer)

target_link_libraries(k3 PUBLIC k3_flags m Threads::Threads)
if(OpenMP_C_FOUND)
  target_link_libraries(k3 PUBLIC OpenMP::OpenMP_C)
endif()

# ------------------------------------------------------------------------------ cli --
add_executable(k3_cli src/cli/k3_run.c)
set_target_properties(k3_cli PROPERTIES OUTPUT_NAME k3)
target_link_libraries(k3_cli PRIVATE k3)

# ---------------------------------------------------------------------------- tests --
# Every test below runs WITHOUT model weights. The checkpoint is 1.56 TB; if correctness
# needed it, correctness would go unchecked.
enable_testing()

function(k3_add_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE k3)
endfunction()

k3_add_test(test_ops    tests/unit/test_ops.c)
k3_add_test(test_cache  tests/unit/test_cache.c)
k3_add_test(test_st     tests/unit/test_st.c)
k3_add_test(test_model_stream tests/unit/test_model_stream.c)
k3_add_test(test_cfg    tests/unit/test_cfg.c)
k3_add_test(test_tok    tests/unit/test_tok.c)
k3_add_test(scale_test  tests/unit/scale_test.c)
k3_add_test(test_trunk  tests/unit/test_trunk.c)
k3_add_test(k3_model    tests/unit/k3_model.c)

set(FIX ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)

# Every test the Makefile's `test` target runs is registered here too. The two build
# systems are documented as interchangeable, so a test present in one and absent from
# the other silently gives ctest less coverage than `make test`.
add_test(NAME ops        COMMAND test_ops   ${FIX}/ops)
add_test(NAME cache      COMMAND test_cache ${FIX}/cache)
add_test(NAME config     COMMAND test_cfg   fixture ${FIX}/ref_k3.json)
add_test(NAME scale      COMMAND scale_test)
add_test(NAME trunk      COMMAND test_trunk)
add_test(NAME oracle     COMMAND k3_model   ${FIX})

# The negative half of the config contract. Each fixture is the working flat config with
# exactly one field mutated, so a rejection can only come from that field. Without these
# the refuse-rather-than-guess policy -- the reason this reader exists -- has no executed
# test at all, in either build system.
add_test(NAME config_reject_no_layermap
         COMMAND test_cfg reject ${FIX}/cfg/no_layermap.json)
add_test(NAME config_reject_bad_layer_index
         COMMAND test_cfg reject ${FIX}/cfg/bad_layer_index.json)
add_test(NAME config_reject_bad_topk
         COMMAND test_cfg reject ${FIX}/cfg/bad_topk.json)
add_test(NAME safetensors COMMAND test_st   ${FIX}/st ${CMAKE_CURRENT_BINARY_DIR}/idx.json
         plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32)
add_test(NAME model_stream COMMAND test_model_stream ${FIX}/st)
add_test(NAME ultra_cli_contract
         COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:k3_cli>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_ultra_cli.cmake)

# The tokenizer test needs the released vocabulary, which is not in this repository.
# Registered UNCONDITIONALLY so ctest reports it as Skipped rather than omitting it:
# wrapping this in if(DEFINED ENV{K3_TOK_FILES}) did the opposite of what that sentence
# claims, and made `ctest` quietly report one fewer test than `make test` runs. The
# wrapper resolves K3_TOK_FILES at TEST time, not configure time, so exporting it before
# ctest now works as the docs say.
add_test(NAME tokenizer
         COMMAND ${CMAKE_COMMAND}
                 -DTEST_TOK=$<TARGET_FILE:test_tok>
                 -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/README.md
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_tok_test.cmake)
set_tests_properties(tokenizer PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")

add_executable(bench_kernels benchmarks/bench_kernels.c)
target_link_libraries(bench_kernels PRIVATE k3)

install(TARGETS k3_cli RUNTIME DESTINATION bin)
install(DIRECTORY include/k3 DESTINATION include)
# k3_cfg.h includes json.h and exposes jval in its signatures, so an installed header
# set without it does not compile. Kept in step with the Makefile's install target.
install(FILES third_party/json.h DESTINATION include/k3)
