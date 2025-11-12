# Fortran Discrete Ordinates Transport Solver Makefile
# Compiles source files from src/ and app/ directories
# Places binary in bin/ folder

# Compiler and flags
FC = gfortran-13
FFLAGS = -O2 -Wall -Wextra -fcheck=all -fbacktrace
LDFLAGS =

# Directories
SRC_DIR = src
APP_DIR = app
BUILD_DIR = build
BIN_DIR = bin

# Target binary name
TARGET = $(BIN_DIR)/fdiscord

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

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	@echo "Clean complete"

# Clean and rebuild
rebuild: clean all

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
	@echo "  all              - Build the project (default)"
	@echo "  clean            - Remove build artifacts"
	@echo "  rebuild          - Clean and rebuild"
	@echo "  run              - Build and run the program"
	@echo "  print            - Print makefile variables"
	@echo "  help             - Show this help message"

.PHONY: all clean rebuild run print help