# Makefile for HW1 Fortran project
# Converted from compile function in hw.py

# Compiler and flags
FC := gfortran
FFLAGS := -O3 -std=f2018 
# FFLAGS += -Wall -Wextra
LDFLAGS :=
LIBS :=

# Optional parallel support
ifdef PARALLEL
    FFLAGS += -fopenmp
    LDFLAGS += -fopenmp
endif

# Directories
SRCDIR := src
APPDIR := app
BUILDDIR := build
BINDIR := bin

# Target binary
TARGET := $(BINDIR)/hw1

# Source files - use wildcard for robustness
MODULES := input_arrays.f90 output_array.f90 mem_util.f90 path_util.f90
MODULE_SRCS := $(addprefix $(SRCDIR)/, $(MODULES))
MODULE_OBJS := $(patsubst $(SRCDIR)/%.f90, $(BUILDDIR)/%.o, $(MODULE_SRCS))

MAIN_SRC := $(APPDIR)/main.f90
MAIN_OBJ := $(BUILDDIR)/main.o

OBJS := $(MAIN_OBJ) $(MODULE_OBJS)

# Default target
.DEFAULT_GOAL := all

all: $(TARGET)

# Create directories as order-only prerequisites
$(BUILDDIR):
	@mkdir -p $@

$(BINDIR):
	@mkdir -p $@

# Pattern rule for compiling modules
$(BUILDDIR)/%.o: $(SRCDIR)/%.f90 | $(BUILDDIR)
	@echo "Compiling module: $<"
	$(FC) $(FFLAGS) -c $< -J$(BUILDDIR) -o $@

# Compile main application (depends on module objects for proper linking)
$(MAIN_OBJ): $(MAIN_SRC) $(MODULE_OBJS) | $(BUILDDIR)
	@echo "Compiling main: $<"
	$(FC) $(FFLAGS) -c $< -I$(BUILDDIR) -o $@

# Link final binary
$(TARGET): $(OBJS) | $(BINDIR)
	@echo "Linking: $@"
	$(FC) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo "Build complete: $@"

# Clean targets
clean:
	@echo "Cleaning all build artifacts..."
	@rm -rf $(BUILDDIR)
	@rm -f $(SRCDIR)/*.mod $(SRCDIR)/*.o
	@rm -f $(TARGET)
	@echo "Clean complete"

clean-build:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR)

clean-src:
	@echo "Cleaning source artifacts..."
	@rm -f $(SRCDIR)/*.mod $(SRCDIR)/*.o

clean-bin:
	@echo "Cleaning binary..."
	@rm -f $(TARGET)

# Force rebuild
rebuild: clean all

# Install target (if needed)
install: $(TARGET)
	@echo "No install target defined"

# Show build information
info:
	@echo "Compiler:    $(FC)"
	@echo "Flags:       $(FFLAGS)"
	@echo "Target:      $(TARGET)"
	@echo "Sources:     $(MODULE_SRCS) $(MAIN_SRC)"
	@echo "Objects:     $(OBJS)"

# Phony targets
.PHONY: all clean clean-build clean-src clean-bin rebuild install info help

# Print help
help:
	@echo "Makefile for HW1 Fortran project"
	@echo ""
	@echo "Usage:"
	@echo "  make [target]              - Build target"
	@echo "  make PARALLEL=1 [target]   - Build with OpenMP support"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  clean      - Clean all build artifacts"
	@echo "  clean-*    - Clean specific artifacts (build/src/bin)"
	@echo "  rebuild    - Clean and rebuild"
	@echo "  info       - Show build configuration"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                - Build normally"
	@echo "  make PARALLEL=1     - Build with OpenMP"
	@echo "  make clean rebuild  - Clean rebuild"