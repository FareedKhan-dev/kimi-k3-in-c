# Kimi K3 inference engine.
#
#   make                build the engine (bin/k3)
#   make test           run every test that needs no model weights
#   make bench          kernel microbenchmarks
#   make portable       build without -march/-mcpu=native (for distribution)
#   make debug          -O0 -g with assertions
#   make asan / ubsan   sanitizer builds
#   make format         clang-format the tree
#   make clean
#
# Nothing here requires a checkpoint. `make test` is the gate that must stay green.
#
# PLATFORMS. Linux/x86-64 is the reference. macOS/arm64 builds with plain `make` too,
# but needs Homebrew's libomp for OpenMP (`brew install libomp`) because Apple Clang
# ships no OpenMP runtime; the platform block below detects and wires it up. Windows
# builds under MSYS2's MinGW64 environment (`pacman -S mingw-w64-x86_64-gcc`) -- open
# the "MSYS2 MinGW x64" shell specifically, not the plain MSYS2 shell, so `cc`/`make`
# resolve to the native-Windows-target toolchain rather than the POSIX-emulation one;
# then plain `make` works, no flags to remember. See src/io/k3_portable_io.h for what
# O_DIRECT, pread, posix_memalign and getrusage become there, and why -static and
# -lpsapi (below) are load-bearing rather than stylistic on this platform.

# ---------------------------------------------------------------------------- config --
CC       ?= cc
PYTHON   ?= python3
BUILD    ?= build
BIN      ?= bin
PREFIX   ?= /usr/local

# ---------------------------------------------------------------------- platform --
# Two things differ on macOS/arm64 and both are build failures, not warnings:
#
#   -march=native   is x86 spelling. -mcpu= is the arm64 one, and it is what sets
#                   tuning as well as architecture; -march= on aarch64 selects an
#                   architecture level and leaves tuning alone even where it is
#                   accepted, and it is rejected outright by Apple Clang before
#                   roughly Xcode 13. `native` rather than a named core, so the same
#                   line works on M1 through M5 and beyond.
#   -fopenmp        Apple Clang ships no OpenMP runtime. Homebrew's libomp supplies it,
#                   but the flag must be passed through the preprocessor
#                   (-Xpreprocessor -fopenmp) and the library linked explicitly.
#
# UNAME_S/UNAME_M are the detection; everything below keys off them. Any of these can
# still be overridden on the command line, which is how `make portable` and the
# sanitizer targets work.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    ARCH ?= -mcpu=native
  else
    ARCH ?= -march=native
  endif
  # Locate libomp without hardcoding a prefix: Homebrew is /opt/homebrew on Apple
  # Silicon and /usr/local on Intel, and MacPorts is elsewhere again. Fall back to the
  # Apple Silicon default if brew is not on PATH, so the error message names a real
  # path rather than an empty one.
  # `:=`, not `?=`. CFLAGS and LDFLAGS are recursive, so a recursive OMP_PREFIX re-runs
  # `brew --prefix` once per expansion -- about twenty forks for `make test`, each a
  # process spawn. A command-line `make OMP_PREFIX=...` still wins either way, because
  # command-line assignments override makefile assignments regardless of flavour.
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp
  # Say what is wrong and how to fix it, rather than letting the compiler report a
  # missing omp.h fifty lines into the build.
  ifeq ($(wildcard $(OMP_PREFIX)/include/omp.h),)
    $(warning libomp not found at $(OMP_PREFIX). Install it with `brew install libomp`,)
    $(warning or point the build at another copy with `make OMP_PREFIX=/path/to/libomp`.)
  endif
else ifneq ($(findstring MINGW,$(UNAME_S)),)
  # MSYS2's uname reports e.g. MINGW64_NT-10.0-26200 -- match the MINGW substring
  # rather than the whole string, since the trailing Windows build number varies.
  #
  # -march=native and -fopenmp behave exactly as on Linux; MinGW-w64's GCC needs no
  # Apple-Clang-style preprocessor dance for OpenMP.
  #
  # -static: the MinGW runtime is POSIX-threads-model (winpthreads), which pulls in
  # libwinpthread-1.dll dynamically by default. A binary launched from outside this
  # shell's PATH then fails immediately with STATUS_DLL_NOT_FOUND -- confirmed
  # directly, not a theoretical risk. Static linking removes the dependency instead
  # of asking every caller to keep MSYS2's bin directory on PATH.
  #
  # -lpsapi: peak_rss_bytes() (k3_run.c) calls GetProcessMemoryInfo for Windows' RSS
  # equivalent, which lives in psapi.dll; nothing else in this build needs it, so it
  # is not pulled in by default the way -lm is.
  ARCH ?= -march=native
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp -static -lpsapi
  # The MinGW-w64 GCC package ships no libasan/libubsan runtime at all (confirmed:
  # `-lasan`/`-lubsan` fail to link, and no such file exists anywhere under /mingw64),
  # unlike its Linux and macOS counterparts. MSYS2's separate Clang/LLVM environment
  # (`pacman -S mingw-w64-clang-x86_64-clang mingw-w64-clang-x86_64-compiler-rt`) does
  # carry a real one, so the sanitizer targets switch compilers on this platform only.
  SANITIZER_CC ?= clang
  SANITIZER_LDFLAGS ?= -lpsapi -lwinpthread
  # These are dynamic DLLs, not statically linked, and Windows resolves them by
  # searching the executable's OWN directory before anything on PATH -- confirmed
  # directly: a PATH containing the exact directory these ship in still failed with
  # STATUS_DLL_NOT_FOUND, and copying them next to bin/k3.exe fixed it immediately.
  # libc++.dll is not an obvious dependency of a C99 project; it is pulled in
  # transitively because libclang_rt.asan_dynamic itself links against it.
  SANITIZER_DLLS ?= $(wildcard /clang64/bin/libclang_rt.asan_dynamic-x86_64.dll) \
                     $(wildcard /clang64/bin/libwinpthread-1.dll) \
                     $(wildcard /clang64/bin/libc++.dll)
else
  # GCC uses -mcpu=native for AArch64 tuning. Keep -march=native on the x86 reference;
  # `make portable` drops both so the produced binary is not tied to the build host.
  ifeq ($(UNAME_M),aarch64)
    ARCH ?= -mcpu=native
  else
    ARCH ?= -march=native
  endif
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif

# Linux and macOS: the platform's own CC already has a working sanitizer runtime, so
# the asan/ubsan targets need no compiler override and no extra link libraries or DLL
# copying. Windows sets all three itself, above.
SANITIZER_CC ?= $(CC)
SANITIZER_LDFLAGS ?=
SANITIZER_DLLS ?=

# -Wpointer-arith is not cosmetic: weight pointers are `const void *`, and arithmetic on
# a void pointer is a silent GNU extension that strides by ONE BYTE. Without this flag
# that mistake compiles clean under -Wall -Wextra and returns the wrong tensor.
#
# -ffp-contract=off keeps floating-point results reproducible across compilers by
# disabling automatic FMA contraction. The test-suite compares against a reference to a
# fixed tolerance; letting the compiler fuse changes results by more than that.
WARN     := -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter
# -pthread is for the trunk's asynchronous reader, which is a plain pthread rather than
# an OpenMP construct. It composes with the platform OpenMP flags above rather than
# replacing them: Apple Clang needs -Xpreprocessor -fopenmp AND -pthread.
CFLAGS   ?= -O3 -std=gnu99 $(WARN) $(ARCH) $(OMP_CFLAGS) -pthread -ffp-contract=off
LDFLAGS  ?= -lm $(OMP_LDFLAGS) -pthread

# Flat include search across the module dirs: sources use "k3.h", "k3_cache.h" etc
# rather than path-qualified includes, which keeps them relocatable.
INCLUDES := -Iinclude -Iinclude/k3 -Ithird_party \
            -Isrc/core -Isrc/io -Isrc/cache -Isrc/model -Isrc/tokenizer

# ----------------------------------------------------------------------------- files --
ENGINE_SRC := src/core/k3_ops.c \
              src/io/k3_st.c src/io/k3_load.c src/io/k3_trunk.c \
              src/cache/k3_cache.c \
              src/model/k3_bind.c
ENGINE_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(ENGINE_SRC))

CLI_SRC    := src/cli/k3_run.c
CLI_BIN    := $(BIN)/k3

# Tests that need no checkpoint. These run in CI on every push.
UNIT_TESTS := test_ops test_cache test_st test_model_stream test_cfg test_tok scale_test k3_model test_trunk
# Tests that need real shards. Built and run by `make test-all` with SHARD_DIR set;
# see the weights-test target below.
WEIGHT_TESTS := test_expert test_real_layer

TEST_BINS  := $(addprefix $(BIN)/,$(UNIT_TESTS))
WEIGHT_BINS := $(addprefix $(BIN)/,$(WEIGHT_TESTS))

FIXTURES   ?= tests/fixtures
TOK_FILES  ?= $(HOME)/k3model

# The safetensors test rebuilds an index and writes it to $(BUILD) rather than /tmp, so
# two concurrent `make test` runs cannot race on one filename and `make clean` removes it.

# ---------------------------------------------------------------------------- targets --
.PHONY: all test test-all bench portable debug asan ubsan format clean install help \
        tok cfg ops cache st oracle weights-test

all: $(CLI_BIN)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CLI_BIN): $(CLI_SRC) $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(CLI_SRC) $(ENGINE_OBJ) -o $@ $(LDFLAGS)

$(BIN):
	@mkdir -p $(BIN)

# Each test links only what it needs, so a failure points at one subsystem.
$(BIN)/test_ops: tests/unit/test_ops.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_cache: tests/unit/test_cache.c $(BUILD)/src/cache/k3_cache.o \
                   $(BUILD)/src/io/k3_load.o $(BUILD)/src/io/k3_st.o \
                   $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_st: tests/unit/test_st.c $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_model_stream: tests/unit/test_model_stream.c $(BUILD)/src/model/k3_bind.o \
                          $(BUILD)/src/io/k3_st.o $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# The tokenizer and config reader are portable C99 with no OpenMP and no platform calls,
# so they build and are verifiable on any machine, including one with no checkpoint.
$(BIN)/test_tok: tests/unit/test_tok.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $< -o $@

$(BIN)/test_cfg: tests/unit/test_cfg.c src/core/k3_ops.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $^ -o $@ -lm

# Allocates at REAL model widths (a ~1.8 GB KDA layer), so it needs the optimised build
# rather than the portable C99 one the tokenizer and config tests use.
$(BIN)/scale_test: tests/unit/scale_test.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/k3_model: tests/unit/k3_model.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_trunk: tests/unit/test_trunk.c $(BUILD)/src/io/k3_trunk.o \
                   $(BUILD)/src/io/k3_st.o \
                   $(BUILD)/src/model/k3_bind.o \
                   $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/bench_kernels: benchmarks/bench_kernels.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

## test: everything that needs no model weights
test: $(CLI_BIN) $(TEST_BINS)
	@echo "== ultra CLI contract =="; \
	  if ./$(CLI_BIN) fake --ids 1 --preset ultra >/dev/null 2>&1; then \
	      echo "ultra mode accepted a missing --trunk"; exit 1; \
	  else rc=$$?; test $$rc -eq 2 || exit 1; fi; \
	  if ./$(CLI_BIN) fake --ids 1 --preset ultra --trunk fake --spec 2 >/dev/null 2>&1; then \
	      echo "ultra mode accepted --spec"; exit 1; \
	  else rc=$$?; test $$rc -eq 2 || exit 1; fi
	@echo "== op kernels ==";        ./$(BIN)/test_ops $(FIXTURES)/ops
	@echo "== streaming cache ==";   ./$(BIN)/test_cache $(FIXTURES)/cache
	@echo "== safetensors ==";       ./$(BIN)/test_st $(FIXTURES)/st $(BUILD)/st_index.json \
	    plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32
	@echo "== model streaming ==";   ./$(BIN)/test_model_stream $(FIXTURES)/st
	@echo "== config reader ==";     ./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
	@echo "== config refusals =="; \
	  for f in no_layermap bad_layer_index bad_topk; do \
	      ./$(BIN)/test_cfg reject $(FIXTURES)/cfg/$$f.json || exit 1; \
	  done
	@echo "== tokenizer =="; \
	  if [ -f "$(TOK_FILES)/tiktoken.model" ]; then \
	      ./$(BIN)/test_tok "$(TOK_FILES)" roundtrip src/core/k3_ops.c; \
	  else \
	      echo "  NOT RUN: no tiktoken.model at $(TOK_FILES)"; \
	      echo "           the vocabulary ships with the checkpoint, not with this"; \
	      echo "           repository. Run: make tok TOK_FILES=/path/to/k3model"; \
	  fi
	@echo "== real dimensions ==";   ./$(BIN)/scale_test
	@echo "== trunk streaming ==";   ./$(BIN)/test_trunk
	@echo "== full-model oracle =="; ./$(BIN)/k3_model $(FIXTURES)
	@echo
	@if [ ! -f "$(TOK_FILES)/tiktoken.model" ]; then \
	     echo "NOTE: tokenizer parity did NOT run; see above. Everything else did."; \
	 fi
	@echo "ALL WEIGHTLESS TESTS PASSED"

## test-all: adds tests that need a real checkpoint (set SHARD_DIR)
test-all: test
	@test -n "$(SHARD_DIR)" || { echo "set SHARD_DIR=/path/to/shards"; exit 2; }
	$(MAKE) weights-test SHARD_DIR="$(SHARD_DIR)"

# SHARD_DIR is quoted: unquoted, a path containing a space (routine on Windows, e.g.
# an "AI LOCAL MODELS" folder) silently splits into extra argv entries instead of
# failing loudly, and the program reads whatever the truncated first token happens to
# resolve to rather than refusing outright.
weights-test: $(WEIGHT_BINS)
	./$(BIN)/test_expert "$(SHARD_DIR)" 1 64
	./$(BIN)/test_real_layer "$(SHARD_DIR)" 1 4 8

$(BIN)/test_expert: tests/unit/test_expert.c $(BUILD)/src/io/k3_load.o \
                    $(BUILD)/src/io/k3_st.o $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_real_layer: tests/unit/test_real_layer.c $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

## tok: tokenizer parity against the reference implementation
tok: $(BIN)/test_tok
	@$(PYTHON) tools/tok_parity.py ./$(BIN)/test_tok

## cfg: config reader against both supported config layouts
cfg: $(BIN)/test_cfg
	@./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
	@test -f "$(TOK_FILES)/config.json" && ./$(BIN)/test_cfg real "$(TOK_FILES)/config.json" \
	    || echo "  (skipped real config: none at $(TOK_FILES))"

## bench: kernel microbenchmarks, no weights required
bench: $(BIN)/bench_kernels
	./$(BIN)/bench_kernels

## portable: drop the -march/-mcpu=native tuning, for a distributable binary
# On x86-64 that means a generic AVX2 + FMA baseline. On arm64 there is no equivalent
# sub-baseline worth naming -- the ISA is the baseline -- so tuning is simply omitted.
portable:
ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
	$(MAKE) ARCH= all
else
	$(MAKE) ARCH="-mavx2 -mfma" all
endif

## debug: -O0 -g, assertions on
debug:
	$(MAKE) CFLAGS="-O0 -g3 -std=gnu99 $(WARN) $(OMP_CFLAGS) -ffp-contract=off" all

# The sanitizer builds drop OpenMP deliberately: ASan's interceptors and the OpenMP
# runtime's thread pool produce false positives together, and a serial build is the
# point of a sanitizer run. OMP_CFLAGS is still omitted rather than replaced, so the
# #pragma omp lines compile to nothing on every platform alike.
asan:
	$(MAKE) CC=$(SANITIZER_CC) CFLAGS="-O1 -g -std=gnu99 $(WARN) -fsanitize=address,undefined -fno-omit-frame-pointer" \
	        LDFLAGS="-lm -fsanitize=address,undefined $(SANITIZER_LDFLAGS)" ARCH= all
ifneq ($(strip $(SANITIZER_DLLS)),)
	cp -f $(SANITIZER_DLLS) $(BIN)/
endif

ubsan:
	$(MAKE) CC=$(SANITIZER_CC) CFLAGS="-O1 -g -std=gnu99 $(WARN) -fsanitize=undefined" \
	        LDFLAGS="-lm -fsanitize=undefined $(SANITIZER_LDFLAGS)" ARCH= all
ifneq ($(strip $(SANITIZER_DLLS)),)
	cp -f $(SANITIZER_DLLS) $(BIN)/
endif

format:
	@command -v clang-format >/dev/null || { echo "clang-format not installed"; exit 1; }
	clang-format -i $(shell find src include tests benchmarks -name '*.c' -o -name '*.h' 2>/dev/null)

# Installs the binary AND the public headers, matching CMake's install rules exactly
# the two build systems are documented as interchangeable, so they must stay so.
# third_party/json.h goes with them: k3_cfg.h includes it and exposes jval in its
# signatures, so an installed k3_cfg.h without it does not compile.
install: $(CLI_BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(CLI_BIN) $(DESTDIR)$(PREFIX)/bin/k3
	install -d $(DESTDIR)$(PREFIX)/include/k3
	install -m 644 include/k3/*.h $(DESTDIR)$(PREFIX)/include/k3/
	install -m 644 third_party/json.h $(DESTDIR)$(PREFIX)/include/k3/

clean:
	rm -rf $(BUILD) $(BIN)

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
