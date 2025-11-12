# Simple Makefile for pFUnit tests

# Compiler
FC = gfortran-13

# pFUnit paths - set PFUNIT_DIR environment variable to your installation
# Example: export PFUNIT_DIR=/path/to/pFUnit/install
ifndef PFUNIT_DIR
    $(error PFUNIT_DIR is not set. Please set it to your pFUnit installation directory)
endif

PFUNIT_BIN = $(PFUNIT_DIR)/PFUNIT-4.14/bin
PFUNIT_INCLUDE = $(PFUNIT_DIR)/PFUNIT-4.14/include
PFUNIT_LIB = $(PFUNIT_DIR)/PFUNIT-4.14/lib

# Compiler flags
FFLAGS = -I$(PFUNIT_INCLUDE)
FFLAGS += -I$(PFUNIT_DIR)/GFTL-1.16/include/v1
FFLAGS += -I$(PFUNIT_DIR)/GFTL_SHARED-1.11/include/v1
FFLAGS += -I$(PFUNIT_DIR)/FARGPARSE-1.10/include
FFLAGS += -I../../fdiscord/src
FFLAGS += -J$(BUILD_DIR)  # Output module files to build directory
FFLAGS += -fopenmp

# Libraries (use only funit for serial tests, not pfunit which requires MPI)
# Order matters: dependents before dependencies
LIBS = -L$(PFUNIT_LIB) -lfunit
LIBS += -L$(PFUNIT_DIR)/FARGPARSE-1.10/lib -lfargparse
LIBS += -L$(PFUNIT_DIR)/GFTL_SHARED-1.11/lib -lgftl-shared-v2 -lgftl-shared-v2-as-default -lgftl-shared-v1
LIBS += -fopenmp

# Build directory for compiled artifacts
BUILD_DIR = build
COVERAGE_DIR = coverage

# Test files
TEST_PF = test_gauss_legendre.pf test_avg_flux.pf test_flux_solvers.pf test_material.pf test_settings.pf test_solve_flux.pf test_flux_copy.pf
TEST_F90 = $(addprefix $(BUILD_DIR)/,$(TEST_PF:.pf=.F90))
TEST_OBJ = $(addprefix $(BUILD_DIR)/,$(TEST_PF:.pf=.o))
DRIVER_OBJ = $(BUILD_DIR)/driver.o
TEST_EXE = test_runner

# Source code to test (from fdiscord)
SRC_DIR = ../../fdiscord/src
SOLVER_OBJ = $(addprefix $(BUILD_DIR)/,solver.o material.o flux.o settings.o)
SRC_FILES = $(SRC_DIR)/solver.f90 $(SRC_DIR)/material.f90 $(SRC_DIR)/flux.f90 $(SRC_DIR)/settings.f90

# Coverage flags
COVERAGE_FLAGS = --coverage

.PHONY: all clean test coverage html-coverage

all: $(TEST_EXE)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Preprocess .pf to .F90 in build directory
$(BUILD_DIR)/%.F90: %.pf | $(BUILD_DIR)
	$(PFUNIT_BIN)/funitproc $< $@

# Compile test .F90 to .o
$(BUILD_DIR)/%.o: $(BUILD_DIR)/%.F90
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -c $< -o $@

# Compile source files from fdiscord
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

# Compile the pFUnit driver
$(DRIVER_OBJ): $(TEST_F90) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -I. -c -D_TEST_SUITES='"testSuites.inc"' $(PFUNIT_INCLUDE)/driver.F90 -o $@

# Dependencies
$(BUILD_DIR)/solver.o: $(BUILD_DIR)/settings.o $(BUILD_DIR)/material.o $(BUILD_DIR)/flux.o
$(BUILD_DIR)/test_gauss_legendre.o: $(BUILD_DIR)/solver.o
$(BUILD_DIR)/test_avg_flux.o: $(BUILD_DIR)/solver.o
$(BUILD_DIR)/test_flux_solvers.o: $(BUILD_DIR)/flux.o
$(BUILD_DIR)/test_material.o: $(BUILD_DIR)/material.o
$(BUILD_DIR)/test_settings.o: $(BUILD_DIR)/settings.o
$(BUILD_DIR)/test_solve_flux.o: $(BUILD_DIR)/solver.o $(BUILD_DIR)/material.o $(BUILD_DIR)/settings.o
$(BUILD_DIR)/test_flux_copy.o: $(BUILD_DIR)/flux.o

# Link test executable
$(TEST_EXE): $(TEST_OBJ) $(SOLVER_OBJ) $(DRIVER_OBJ)
	$(FC) -o $@ $^ $(LIBS)

# Run tests
test: $(TEST_EXE)
	./$(TEST_EXE) -x > test_results.xml

# Coverage target - rebuild with coverage and generate reports
coverage:
	@echo "Building with coverage instrumentation..."
	$(MAKE) clean
	$(MAKE) FFLAGS="$(FFLAGS) $(COVERAGE_FLAGS)" LIBS="$(LIBS) $(COVERAGE_FLAGS)" all
	@echo "Running tests..."
	./$(TEST_EXE)
	@echo "Generating coverage reports..."
	@mkdir -p $(COVERAGE_DIR)
	@echo ""
	@echo "Coverage Summary:"
	@echo "================"
	@for src in $(SRC_FILES); do \
		base=$$(basename $$src .f90); \
		output=$$(gcov -o $(BUILD_DIR) $$src 2>&1); \
		coverage=$$(echo "$$output" | grep "Lines executed" | tail -1); \
		printf "  %-15s %s\n" "$$base.f90:" "$$coverage"; \
		if [ -f $$base.f90.gcov ]; then \
			mv $$base.f90.gcov $(COVERAGE_DIR)/; \
		fi; \
	done
	@echo ""
	@echo "Detailed reports available in $(COVERAGE_DIR)/"

# HTML coverage target - generate browsable HTML coverage reports
html-coverage:
	@echo "Building with coverage instrumentation..."
	$(MAKE) clean
	$(MAKE) FFLAGS="$(FFLAGS) $(COVERAGE_FLAGS)" LIBS="$(LIBS) $(COVERAGE_FLAGS)" all
	@echo "Running tests..."
	./$(TEST_EXE)
	@echo "Generating HTML coverage reports..."
	@mkdir -p $(COVERAGE_DIR)/html
	@lcov --capture --directory $(BUILD_DIR) --output-file $(COVERAGE_DIR)/coverage-all.info \
		--gcov-tool gcov-13 --ignore-errors source --quiet
	@lcov --extract $(COVERAGE_DIR)/coverage-all.info '*/fdiscord/src/*' \
		--output-file $(COVERAGE_DIR)/coverage.info --quiet
	@genhtml $(COVERAGE_DIR)/coverage.info --output-directory $(COVERAGE_DIR)/html \
		--ignore-errors unmapped,source --quiet
	@echo "Cleaning up temporary gcov files..."
	@find ../.. -name "*.gcov" ! -path "*/tests/fortran/coverage/*" -delete 2>/dev/null || true
	@rm -f $(COVERAGE_DIR)/coverage-all.info
	@echo ""
	@echo "HTML coverage report generated in $(COVERAGE_DIR)/html/"
	@echo "Open $(COVERAGE_DIR)/html/index.html in a browser to view"

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TEST_EXE) $(COVERAGE_DIR) *.gcda *.gcno *.gcov test_results.xml