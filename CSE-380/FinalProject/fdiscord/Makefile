# Fortran Discrete Ordinates Transport Solver Makefile
# Compiles source files from src/ and app/ directories
# Places binary in bin/ folder

# Compiler and flags
FC = gfortran-13
FC_MPI = mpifort
FFLAGS = -O2 -Wall -Wextra -fcheck=all -fbacktrace -fopenmp
FFLAGS_MPI = -O2 -Wall -Wextra -fopenmp
LDFLAGS = -fopenmp

# Directories
SRC_DIR = src
APP_DIR = app
BUILD_DIR = build
BUILD_DIR_MPI = build_mpi
BUILD_DIR_SERIAL = build_serial
BIN_DIR = bin

# Target binary names
TARGET = $(BIN_DIR)/fdiscord
TARGET_MPI = $(BIN_DIR)/fdiscord_mpi
TARGET_SERIAL = $(BIN_DIR)/fdiscord_serial

# Source files
SRC_FILES = $(SRC_DIR)/material.f90 \
            $(SRC_DIR)/settings.f90 \
            $(SRC_DIR)/flux.f90 \
            $(SRC_DIR)/json_reader.f90 \
            $(SRC_DIR)/json_writer.f90 \
            $(SRC_DIR)/solver.f90
APP_FILES = $(APP_DIR)/main.f90

# Object files (placed in build directory)
SRC_OBJS = $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(SRC_FILES))
APP_OBJS = $(patsubst $(APP_DIR)/%.f90,$(BUILD_DIR)/%.o,$(APP_FILES))
ALL_OBJS = $(SRC_OBJS) $(APP_OBJS)

# Module files
MOD_DIR = $(BUILD_DIR)

# Default target
all: $(TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link the final binary
$(TARGET): $(ALL_OBJS) | $(BIN_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -o $@ $(ALL_OBJS) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile source files with module dependencies
# Independent modules (no dependencies)
$(BUILD_DIR)/material.o: $(SRC_DIR)/material.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -I$(MOD_DIR) -c $< -o $@

$(BUILD_DIR)/settings.o: $(SRC_DIR)/settings.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -I$(MOD_DIR) -c $< -o $@

$(BUILD_DIR)/flux.o: $(SRC_DIR)/flux.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -I$(MOD_DIR) -c $< -o $@

# JSON reader depends on material and settings modules
$(BUILD_DIR)/json_reader.o: $(SRC_DIR)/json_reader.f90 $(BUILD_DIR)/material.o $(BUILD_DIR)/settings.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -I$(MOD_DIR) -c $< -o $@

# JSON writer depends on material and settings modules
$(BUILD_DIR)/json_writer.o: $(SRC_DIR)/json_writer.f90 $(BUILD_DIR)/material.o $(BUILD_DIR)/settings.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -I$(MOD_DIR) -c $< -o $@

# Solver depends on material, settings, and flux modules
$(BUILD_DIR)/solver.o: $(SRC_DIR)/solver.f90 $(BUILD_DIR)/material.o $(BUILD_DIR)/settings.o $(BUILD_DIR)/flux.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -I$(MOD_DIR) -c $< -o $@

# Main depends on material, settings, solver, json_reader, and json_writer modules
$(BUILD_DIR)/main.o: $(APP_DIR)/main.f90 $(BUILD_DIR)/material.o $(BUILD_DIR)/settings.o $(BUILD_DIR)/solver.o $(BUILD_DIR)/json_reader.o $(BUILD_DIR)/json_writer.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(MOD_DIR) -I$(MOD_DIR) -c $< -o $@

# ============================================================================
# MPI VERSION
# ============================================================================

# MPI object files
SRC_OBJS_MPI = $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR_MPI)/%.o,$(SRC_FILES))
APP_OBJS_MPI = $(patsubst $(APP_DIR)/%.f90,$(BUILD_DIR_MPI)/%.o,$(APP_FILES))
ALL_OBJS_MPI = $(SRC_OBJS_MPI) $(APP_OBJS_MPI)
MOD_DIR_MPI = $(BUILD_DIR_MPI)

# Build MPI version
mpi: $(TARGET_MPI)

# Create MPI build directory
$(BUILD_DIR_MPI):
	mkdir -p $(BUILD_DIR_MPI)

# Link MPI binary
$(TARGET_MPI): $(ALL_OBJS_MPI) | $(BIN_DIR)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -o $@ $(ALL_OBJS_MPI) $(LDFLAGS)
	@echo "MPI build complete: $(TARGET_MPI)"

# MPI compile rules
$(BUILD_DIR_MPI)/material.o: $(SRC_DIR)/material.f90 | $(BUILD_DIR_MPI)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -I$(MOD_DIR_MPI) -c $< -o $@

$(BUILD_DIR_MPI)/settings.o: $(SRC_DIR)/settings.f90 | $(BUILD_DIR_MPI)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -I$(MOD_DIR_MPI) -c $< -o $@

$(BUILD_DIR_MPI)/flux.o: $(SRC_DIR)/flux.f90 | $(BUILD_DIR_MPI)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -I$(MOD_DIR_MPI) -c $< -o $@

$(BUILD_DIR_MPI)/json_reader.o: $(SRC_DIR)/json_reader.f90 $(BUILD_DIR_MPI)/material.o $(BUILD_DIR_MPI)/settings.o | $(BUILD_DIR_MPI)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -I$(MOD_DIR_MPI) -c $< -o $@

$(BUILD_DIR_MPI)/json_writer.o: $(SRC_DIR)/json_writer.f90 $(BUILD_DIR_MPI)/material.o $(BUILD_DIR_MPI)/settings.o | $(BUILD_DIR_MPI)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -I$(MOD_DIR_MPI) -c $< -o $@

$(BUILD_DIR_MPI)/solver.o: $(SRC_DIR)/solver.f90 $(BUILD_DIR_MPI)/material.o $(BUILD_DIR_MPI)/settings.o $(BUILD_DIR_MPI)/flux.o | $(BUILD_DIR_MPI)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -I$(MOD_DIR_MPI) -c $< -o $@

$(BUILD_DIR_MPI)/main.o: $(APP_DIR)/main.f90 $(BUILD_DIR_MPI)/material.o $(BUILD_DIR_MPI)/settings.o $(BUILD_DIR_MPI)/solver.o $(BUILD_DIR_MPI)/json_reader.o $(BUILD_DIR_MPI)/json_writer.o | $(BUILD_DIR_MPI)
	$(FC_MPI) $(FFLAGS_MPI) -J$(MOD_DIR_MPI) -I$(MOD_DIR_MPI) -c $< -o $@

# ============================================================================
# SERIAL VERSION (no OpenMP)
# ============================================================================

# Serial flags without OpenMP
FFLAGS_SERIAL = -O2 -Wall -Wextra -fcheck=all -fbacktrace
LDFLAGS_SERIAL =

# Serial object files
SRC_OBJS_SERIAL = $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR_SERIAL)/%.o,$(SRC_FILES))
APP_OBJS_SERIAL = $(patsubst $(APP_DIR)/%.f90,$(BUILD_DIR_SERIAL)/%.o,$(APP_FILES))
ALL_OBJS_SERIAL = $(SRC_OBJS_SERIAL) $(APP_OBJS_SERIAL)
MOD_DIR_SERIAL = $(BUILD_DIR_SERIAL)

# Build serial version
serial: $(TARGET_SERIAL)

# Create serial build directory
$(BUILD_DIR_SERIAL):
	mkdir -p $(BUILD_DIR_SERIAL)

# Link serial binary
$(TARGET_SERIAL): $(ALL_OBJS_SERIAL) | $(BIN_DIR)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -o $@ $(ALL_OBJS_SERIAL) $(LDFLAGS_SERIAL)
	@echo "Serial build complete: $(TARGET_SERIAL)"

# Serial compile rules
$(BUILD_DIR_SERIAL)/material.o: $(SRC_DIR)/material.f90 | $(BUILD_DIR_SERIAL)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -I$(MOD_DIR_SERIAL) -c $< -o $@

$(BUILD_DIR_SERIAL)/settings.o: $(SRC_DIR)/settings.f90 | $(BUILD_DIR_SERIAL)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -I$(MOD_DIR_SERIAL) -c $< -o $@

$(BUILD_DIR_SERIAL)/flux.o: $(SRC_DIR)/flux.f90 | $(BUILD_DIR_SERIAL)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -I$(MOD_DIR_SERIAL) -c $< -o $@

$(BUILD_DIR_SERIAL)/json_reader.o: $(SRC_DIR)/json_reader.f90 $(BUILD_DIR_SERIAL)/material.o $(BUILD_DIR_SERIAL)/settings.o | $(BUILD_DIR_SERIAL)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -I$(MOD_DIR_SERIAL) -c $< -o $@

$(BUILD_DIR_SERIAL)/json_writer.o: $(SRC_DIR)/json_writer.f90 $(BUILD_DIR_SERIAL)/material.o $(BUILD_DIR_SERIAL)/settings.o | $(BUILD_DIR_SERIAL)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -I$(MOD_DIR_SERIAL) -c $< -o $@

$(BUILD_DIR_SERIAL)/solver.o: $(SRC_DIR)/solver.f90 $(BUILD_DIR_SERIAL)/material.o $(BUILD_DIR_SERIAL)/settings.o $(BUILD_DIR_SERIAL)/flux.o | $(BUILD_DIR_SERIAL)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -I$(MOD_DIR_SERIAL) -c $< -o $@

$(BUILD_DIR_SERIAL)/main.o: $(APP_DIR)/main.f90 $(BUILD_DIR_SERIAL)/material.o $(BUILD_DIR_SERIAL)/settings.o $(BUILD_DIR_SERIAL)/solver.o $(BUILD_DIR_SERIAL)/json_reader.o $(BUILD_DIR_SERIAL)/json_writer.o | $(BUILD_DIR_SERIAL)
	$(FC) $(FFLAGS_SERIAL) -J$(MOD_DIR_SERIAL) -I$(MOD_DIR_SERIAL) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BUILD_DIR_MPI) $(BUILD_DIR_SERIAL)
	rm -rf $(BIN_DIR)
	@echo "Clean complete"

# Clean and rebuild
rebuild: clean all

# Clean and rebuild MPI
rebuild_mpi: clean mpi

# Run the program
run: $(TARGET)
	$(TARGET)

# Print variables for debugging
print:
	@echo "FC:        $(FC)"
	@echo "FFLAGS:    $(FFLAGS)"
	@echo "SRC_DIR:   $(SRC_DIR)"
	@echo "APP_DIR:   $(APP_DIR)"
	@echo "BUILD_DIR: $(BUILD_DIR)"
	@echo "BIN_DIR:   $(BIN_DIR)"
	@echo "TARGET:    $(TARGET)"
	@echo "SRC_FILES: $(SRC_FILES)"
	@echo "APP_FILES: $(APP_FILES)"
	@echo "ALL_OBJS:  $(ALL_OBJS)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the project (default, with OpenMP)"
	@echo "  mpi              - Build MPI version"
	@echo "  serial           - Build serial version (no OpenMP)"
	@echo "  clean            - Remove build artifacts"
	@echo "  rebuild          - Clean and rebuild"
	@echo "  rebuild_mpi      - Clean and rebuild MPI version"
	@echo "  run              - Build and run the program"
	@echo "  print            - Print makefile variables"
	@echo "  help             - Show this help message"

.PHONY: all mpi serial clean rebuild rebuild_mpi run print help