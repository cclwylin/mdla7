# MDLA7 SystemC v0 build
#
# Usage:   make
#          make SYSTEMC_HOME=/custom/path     (override auto-detect)
#
# Tested:  macOS Intel (x86_64) + Apple Silicon (arm64) via Homebrew systemc;
#          SystemC 2.3.4 / 3.0.x; clang++ (Apple) and g++ (Linux).

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# --- Auto-detect SystemC home if not set externally ----------------------
ifndef SYSTEMC_HOME
  ifeq ($(UNAME_S),Darwin)
    # Prefer "brew --prefix systemc"; fall back to per-arch default.
    BREW_SYSTEMC := $(shell brew --prefix systemc 2>/dev/null)
    ifneq ($(BREW_SYSTEMC),)
      SYSTEMC_HOME := $(BREW_SYSTEMC)
    else ifeq ($(UNAME_M),arm64)
      SYSTEMC_HOME := /opt/homebrew/opt/systemc
    else
      SYSTEMC_HOME := /usr/local/opt/systemc
    endif
  else
    SYSTEMC_HOME ?= /usr/local
  endif
endif

SYSTEMC_INC := $(SYSTEMC_HOME)/include
SYSTEMC_LIB := $(SYSTEMC_HOME)/lib
ifeq ($(wildcard $(SYSTEMC_LIB)/libsystemc.*),)
  ifneq ($(wildcard $(SYSTEMC_HOME)/lib-macosx64/libsystemc.*),)
    SYSTEMC_LIB := $(SYSTEMC_HOME)/lib-macosx64
  else ifneq ($(wildcard $(SYSTEMC_HOME)/lib-linux64/libsystemc.*),)
    SYSTEMC_LIB := $(SYSTEMC_HOME)/lib-linux64
  endif
endif

# Sanity check: bail loudly if SystemC wasn't found.
SYSTEMC_LIB_CHECK := $(wildcard $(SYSTEMC_LIB)/libsystemc.*)
ifeq ($(SYSTEMC_LIB_CHECK),)
  $(warning SystemC not found at SYSTEMC_HOME=$(SYSTEMC_HOME))
  $(warning  - Intel macOS: brew install systemc → /usr/local/opt/systemc)
  $(warning  - Apple Silicon: brew install systemc → /opt/homebrew/opt/systemc)
  $(warning  - Or set SYSTEMC_HOME=... explicitly)
  $(error  cannot proceed without SystemC)
endif

CXX      ?= clang++

# macOS: Apple CLT's partial c++/v1 directory shadows the SDK's complete one.
# Force the SDK's libc++ headers in to avoid "cassert: file not found".
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SDKROOT ?= $(shell xcrun --show-sdk-path)
  MACOS_LIBCXX_FIX := -isystem $(SDKROOT)/usr/include/c++/v1
endif

CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
            -DSC_ALLOW_DEPRECATED_IEEE_API \
            -I include -I $(SYSTEMC_INC) $(MACOS_LIBCXX_FIX)
LDFLAGS  := -L $(SYSTEMC_LIB) -lsystemc -Wl,-rpath,$(SYSTEMC_LIB)

BUILD   := build
HEADERS := $(wildcard include/mdla7/*.h)
HOST_STAMP := $(BUILD)/.host-$(UNAME_S)-$(UNAME_M)

# Each .cpp under src/ becomes its own binary.
BINS := $(patsubst src/%.cpp,$(BUILD)/%,$(wildcard src/*.cpp))

all: $(BINS)

$(BUILD)/%: src/%.cpp $(HEADERS) $(HOST_STAMP) | $(BUILD)
	@echo "[build] $@   ($(UNAME_S)/$(UNAME_M))"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD):
	mkdir -p $@

$(HOST_STAMP): | $(BUILD)
	@rm -f $(BUILD)/.host-*
	@touch $@

run: $(BUILD)/main
	$(BUILD)/main

test: $(BUILD)/test_conv_synth
	$(BUILD)/test_conv_synth

# Extract first CONV op from a .tflite into build/conv_layer.bin, then run test.
# Models live at <repo>/model/, one level up from systemc/.
TFLITE ?= ../model/INT8/efficientnet_lite0_int8.tflite

test-tflite: $(BUILD)/test_tflite_conv $(BUILD)/conv_layer.bin
	$(BUILD)/test_tflite_conv $(BUILD)/conv_layer.bin

$(BUILD)/conv_layer.bin: scripts/extract_conv.py $(TFLITE) | $(BUILD)
	python3 scripts/extract_conv.py $(TFLITE) $@

clean:
	rm -rf $(BUILD)

.PHONY: all run test test-tflite clean
