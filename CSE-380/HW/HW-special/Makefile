F90 := gfortran
FFLAGS := -O3 -std=f2018
ifdef PARALLEL
    FFLAGS += -fopenmp
endif

BIN_DIR = bin
BUILD_DIR = build

TARGET 		:= $(BIN_DIR)/hw_special
MODULES 	:= $(wildcard src/*.f90)
APPS 		:= $(wildcard app/*.f90)
MODULE_OBJS := $(patsubst src/%.f90,$(BUILD_DIR)/%.o,$(MODULES))
APP_OBJS 	:= $(patsubst app/%.f90,$(BUILD_DIR)/%.o,$(APPS))
OBJS 		:= $(MODULE_OBJS) $(APP_OBJS)

# Link all object files to create the final executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(F90) $(FFLAGS) $^ -o $@

# Ensure modules are compiled before applications that use them
$(APP_OBJS): $(MODULE_OBJS)

# Compile source modules into object files in build directory
$(BUILD_DIR)/%.o: src/%.f90 | $(BUILD_DIR)
	$(F90) $(FFLAGS) -J$(BUILD_DIR) -c $< -o $@

# Compile application files into object files in build directory
$(BUILD_DIR)/%.o: app/%.f90 | $(BUILD_DIR)
	$(F90) $(FFLAGS) -J$(BUILD_DIR) -c $< -o $@

# Create bin and build directories if they don't exist
$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: clean