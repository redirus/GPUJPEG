# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yiyuan/rocs-dev/GPUJPEG

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yiyuan/rocs-dev/GPUJPEG/build

# Include any dependencies generated for this target.
include CMakeFiles/tester.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tester.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tester.dir/flags.make

CMakeFiles/tester.dir/src/main.c.o: CMakeFiles/tester.dir/flags.make
CMakeFiles/tester.dir/src/main.c.o: ../src/main.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yiyuan/rocs-dev/GPUJPEG/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/tester.dir/src/main.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/tester.dir/src/main.c.o   -c /home/yiyuan/rocs-dev/GPUJPEG/src/main.c

CMakeFiles/tester.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/tester.dir/src/main.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yiyuan/rocs-dev/GPUJPEG/src/main.c > CMakeFiles/tester.dir/src/main.c.i

CMakeFiles/tester.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/tester.dir/src/main.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yiyuan/rocs-dev/GPUJPEG/src/main.c -o CMakeFiles/tester.dir/src/main.c.s

CMakeFiles/tester.dir/src/main.c.o.requires:
.PHONY : CMakeFiles/tester.dir/src/main.c.o.requires

CMakeFiles/tester.dir/src/main.c.o.provides: CMakeFiles/tester.dir/src/main.c.o.requires
	$(MAKE) -f CMakeFiles/tester.dir/build.make CMakeFiles/tester.dir/src/main.c.o.provides.build
.PHONY : CMakeFiles/tester.dir/src/main.c.o.provides

CMakeFiles/tester.dir/src/main.c.o.provides.build: CMakeFiles/tester.dir/src/main.c.o

# Object files for target tester
tester_OBJECTS = \
"CMakeFiles/tester.dir/src/main.c.o"

# External object files for target tester
tester_EXTERNAL_OBJECTS =

tester: CMakeFiles/tester.dir/src/main.c.o
tester: CMakeFiles/tester.dir/build.make
tester: /usr/local/cuda/lib64/libcudart.so
tester: libgpujpeg.so
tester: /usr/local/cuda/lib64/libcudart.so
tester: CMakeFiles/tester.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable tester"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tester.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tester.dir/build: tester
.PHONY : CMakeFiles/tester.dir/build

CMakeFiles/tester.dir/requires: CMakeFiles/tester.dir/src/main.c.o.requires
.PHONY : CMakeFiles/tester.dir/requires

CMakeFiles/tester.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tester.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tester.dir/clean

CMakeFiles/tester.dir/depend:
	cd /home/yiyuan/rocs-dev/GPUJPEG/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yiyuan/rocs-dev/GPUJPEG /home/yiyuan/rocs-dev/GPUJPEG /home/yiyuan/rocs-dev/GPUJPEG/build /home/yiyuan/rocs-dev/GPUJPEG/build /home/yiyuan/rocs-dev/GPUJPEG/build/CMakeFiles/tester.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tester.dir/depend

