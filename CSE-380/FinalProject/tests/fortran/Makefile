# Makefile for FDiscord unit tests

FC = gfortran
FFLAGS = -O2 -Wall -Wextra -fcheck=all -fbacktrace
FFLAGS_COV = -O0 -Wall -Wextra -fcheck=all -fbacktrace --coverage -fprofile-arcs -ftest-coverage
LDFLAGS =
LDFLAGS_COV = --coverage

# Source directories
SRC_DIR = ../../fdiscord/src
TEST_DIR = .
BUILD_DIR = build

# Source files from main project
SRC_FILES = $(SRC_DIR)/material.f90 \
            $(SRC_DIR)/settings.f90 \
            $(SRC_DIR)/flux.f90 \
            $(SRC_DIR)/solver.f90

# Test file
TEST_FILE = test_fdiscord.f90

# Object files
SRC_OBJS = $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(SRC_FILES))
TEST_OBJ = $(BUILD_DIR)/test_fdiscord.o

# Target
TARGET = test_fdiscord

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source modules
$(BUILD_DIR)/material.o: $(SRC_DIR)/material.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/settings.o: $(SRC_DIR)/settings.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/flux.o: $(SRC_DIR)/flux.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/solver.o: $(SRC_DIR)/solver.f90 $(BUILD_DIR)/material.o $(BUILD_DIR)/settings.o $(BUILD_DIR)/flux.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

# Compile test file
$(TEST_OBJ): $(TEST_FILE) $(SRC_OBJS) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

# Link
$(TARGET): $(SRC_OBJS) $(TEST_OBJ)
	$(FC) $(FFLAGS) -o $@ $(SRC_OBJS) $(TEST_OBJ) $(LDFLAGS)
	@echo "Test executable built: $(TARGET)"

run: $(TARGET)
	./$(TARGET)

# Coverage target - build with coverage instrumentation
coverage:
	$(MAKE) clean
	$(MAKE) all FFLAGS="$(FFLAGS_COV)" LDFLAGS="$(LDFLAGS_COV)"
	@echo "Coverage test build complete"

# Generate coverage report (run after executing tests)
coverage-report:
	@echo "Generating Fortran test coverage reports..."
	@mkdir -p coverage
	@for f in $(SRC_DIR)/*.f90; do \
		base=$$(basename $$f .f90); \
		if [ -f $(BUILD_DIR)/$$base.gcda ]; then \
			gcov -o $(BUILD_DIR) $$f > coverage/$$base.gcov.txt 2>&1; \
			mv $$base.f90.gcov coverage/ 2>/dev/null || true; \
		fi; \
	done
	@echo "Coverage reports generated in coverage/"
	@echo ""
	@echo "=== Coverage Summary ==="
	@for f in coverage/*.gcov; do \
		if [ -f "$$f" ]; then \
			echo ""; \
			head -n 5 "$$f" | grep -E "(File|Lines|Branches)" || true; \
		fi; \
	done

# Clean coverage data
coverage-clean:
	rm -f $(BUILD_DIR)/*.gcda $(BUILD_DIR)/*.gcno
	rm -rf coverage *.gcov
	@echo "Coverage data cleaned"

clean:
	rm -rf $(BUILD_DIR) $(TARGET) *.gcov
	@echo "Clean complete"

.PHONY: all run clean coverage coverage-report coverage-clean