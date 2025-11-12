# Makefile for FDiscord unit tests with FRUIT

FC = gfortran
FFLAGS = -O2 -Wall -Wextra -fcheck=all -fbacktrace
FFLAGS_FRUIT = -O2 -std=legacy  # More permissive flags for FRUIT (older Fortran code)
FFLAGS_COV = -O0 -Wall -Wextra -fcheck=all -fbacktrace --coverage -fprofile-arcs -ftest-coverage
FFLAGS_FRUIT_COV = -O0 -std=legacy --coverage -fprofile-arcs -ftest-coverage
LDFLAGS =
LDFLAGS_COV = --coverage

# Source directories
SRC_DIR = ../../fdiscord/src
TEST_DIR = .
BUILD_DIR = build
FRUIT_DIR = fruit/src

# Source files from main project
SRC_FILES = $(SRC_DIR)/material.f90 \
            $(SRC_DIR)/settings.f90 \
            $(SRC_DIR)/flux.f90 \
            $(SRC_DIR)/solver.f90

# FRUIT framework file
FRUIT_FILE = $(FRUIT_DIR)/fruit.f90

# Test files
TEST_MODULE = test_fdiscord_fruit.f90
TEST_DRIVER = fruit_driver.f90

# Object files
SRC_OBJS = $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(SRC_FILES))
FRUIT_OBJ = $(BUILD_DIR)/fruit.o
TEST_MODULE_OBJ = $(BUILD_DIR)/test_fdiscord_fruit.o
TEST_DRIVER_OBJ = $(BUILD_DIR)/fruit_driver.o

# Targets
TARGET = test_fdiscord_fruit
TARGET_OLD = test_fdiscord

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

# Compile FRUIT framework (with permissive flags for older code)
$(FRUIT_OBJ): $(FRUIT_FILE) | $(BUILD_DIR)
	$(FC) $(FFLAGS_FRUIT) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

# Compile test module (depends on FRUIT and solver modules)
$(TEST_MODULE_OBJ): $(TEST_MODULE) $(FRUIT_OBJ) $(SRC_OBJS) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

# Compile test driver (depends on FRUIT and test module)
$(TEST_DRIVER_OBJ): $(TEST_DRIVER) $(FRUIT_OBJ) $(TEST_MODULE_OBJ) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

# Link FRUIT test executable
$(TARGET): $(SRC_OBJS) $(FRUIT_OBJ) $(TEST_MODULE_OBJ) $(TEST_DRIVER_OBJ)
	$(FC) $(FFLAGS) -o $@ $(SRC_OBJS) $(FRUIT_OBJ) $(TEST_MODULE_OBJ) $(TEST_DRIVER_OBJ) $(LDFLAGS)
	@echo "FRUIT test executable built: $(TARGET)"

# Build old test (without FRUIT) for comparison
$(TARGET_OLD): $(SRC_OBJS) $(BUILD_DIR)/test_fdiscord.o
	$(FC) $(FFLAGS) -o $@ $(SRC_OBJS) $(BUILD_DIR)/test_fdiscord.o $(LDFLAGS)
	@echo "Old test executable built: $(TARGET_OLD)"

$(BUILD_DIR)/test_fdiscord.o: test_fdiscord.f90 $(SRC_OBJS) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c $< -o $@

old: $(TARGET_OLD)

run: $(TARGET)
	./$(TARGET)

# Coverage target - build with coverage instrumentation
# Note: FRUIT framework is not instrumented for coverage (we only care about source code coverage)
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
	@for f in coverage/*.gcov.txt; do \
		if [ -f "$$f" ]; then \
			grep "Lines executed" "$$f" 2>/dev/null | head -1 || true; \
		fi; \
	done
	@echo ""
	@echo "Detailed coverage reports: coverage/*.gcov"
	@echo "For HTML reports, run: make coverage-html"

# Generate HTML coverage report using lcov (recommended) or gcovr
coverage-html:
	@echo "Generating HTML coverage report..."
	@if command -v lcov >/dev/null 2>&1; then \
		echo "Using lcov..."; \
		mkdir -p coverage/html; \
		lcov --capture --directory $(BUILD_DIR) --output-file coverage/coverage.info 2>/dev/null; \
		lcov --remove coverage/coverage.info '*/fruit/*' --output-file coverage/coverage.info 2>/dev/null; \
		genhtml coverage/coverage.info --output-directory coverage/html 2>/dev/null; \
		echo ""; \
		echo "HTML coverage report generated: coverage/html/index.html"; \
		echo "Open with: firefox coverage/html/index.html"; \
	elif command -v gcovr >/dev/null 2>&1; then \
		echo "Using gcovr..."; \
		gcovr --root . --filter '$(SRC_DIR)/.*' --html --html-details \
			--gcov-ignore-errors=no_working_dir_found \
			-o coverage/coverage.html; \
		echo ""; \
		echo "HTML coverage report generated: coverage/coverage.html"; \
		echo "Open with: firefox coverage/coverage.html"; \
	else \
		echo ""; \
		echo "ERROR: Neither lcov nor gcovr is installed."; \
		echo ""; \
		echo "Install one of:"; \
		echo "  - lcov:  sudo apt-get install lcov"; \
		echo "  - gcovr: pip3 install gcovr"; \
		echo ""; \
		exit 1; \
	fi

# Clean coverage data
coverage-clean:
	rm -f $(BUILD_DIR)/*.gcda $(BUILD_DIR)/*.gcno
	rm -rf coverage *.gcov
	@echo "Coverage data cleaned"

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TARGET_OLD) *.gcov
	@echo "Clean complete"

.PHONY: all old run clean coverage coverage-report coverage-html coverage-clean