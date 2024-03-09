#=========================================================================
# file: $(BLD_DIR)/config.mk
#
# This makefile fragment is the main configuration file for the pscfpp
# build system.  A copy of this file is included by all other makefiles. 
# This file is created and installed by the "setup" script. One copy of 
# this file is installed in the root directory of the src/ directory
# tree, which is used for in-source compilation. Another copy is 
# installed in the root directory of the bld/ directory tree, which is 
# used for out-of-source compilation.
#
# This file contains user-modifiable definitions of several types of 
# makefile variables:
# 
#  - Variables ROOT_DIR, SRC_DIR, BLD_DIR, BIN_DIR and DAT_DIR that 
#    contain absolute paths for the simpatico root directory and some 
#    of its subdirectories.
#
#  - A variable UTIL_DEBUG that if defined, enables compilation of
#    a debugging version fo the code with additional sanity checks.
#
#  - Variables that control the command name by which the compiler 
#    is invoked and command line options passed to the compiler 
# 
#  - A variable MAKEDEP that enables automatic dependency generation.
#
#=========================================================================
# Variables that define absolute directory paths
#
# In the config.mk file installed in each build directory, correct
# values of these variables should have been set by the setup script,
# and should not need to modified by the user.

# Absolute path to the root simpatico directory
ROOT_DIR=/home/juntong/CSU/pscfplus

# Path to the build directory (location for intermediate generated files)
# This should also be the directory that contains this script.
BLD_DIR=$(ROOT_DIR)/bld

# Path to the source directory (contains C++ source code files)
SRC_DIR=$(ROOT_DIR)/src

# Installation directory for binary executable program files
BIN_DIR=$(ROOT_DIR)/bin

# Directory for shared permanent (read-only) data used by programs.
DAT_DIR=$(ROOT_DIR)/data

#======================================================================
# Conditional compilation of debugging.

# Defining UTIL_DEBUG enables a variety of extra sanity checks, at some
# cost in speed. Debugging is disabled (commented out) by default.
#UTIL_DEBUG=1

# Comment: After setup but before compilation, the above definitions of 
# UTIL_DEBUG may be uncommented or commented out from the command line 
# by using the "configure" script. To do so, invoke the configure script 
# with the -g option prior to compilation from the directory that contains 
# this main configuration file (e.g., from src/ or bld/). Specifically, 
# invoke "./configure -g1" to enable debugging or "./configure -g0" to
# disable debugging. 
#
#======================================================================
# Compiler configuration variables.
#
# The following block of variable definitions is initialized by 
# the setup script by copying a compiler configuration file in the 
# make/compiler directory.  If the setup script is invoked with no 
# argument, the file "make/compiler/default" is used by default.  
# Users may add files to the make/compiler directory to store 
# settings required for a particular environment. The name of a 
# desired compiler configuration file may be specfied by invoking
# the setup script with the base name of the desired file as an 
# argument (e.g., "> ./setup local"). 
#
# Variables defined in this block define the names of the commands 
# used to invoke the compiler when compiling and linking files, some
# of the command line options passed to the compiler to control,
# and search paths for header files and libraries files for required
# external libraries.  See the section of this file entitled "Makefile 
# Patterns and Recipes" for a discussion of how these variables are 
# used.
#
#=========================================================================
# file: $(BLD_DIR)/config.mk
#
# This makefile fragment is the main configuration file for the pscfpp
# build system.  A copy of this file is included by all other makefiles. 
# This file is created and installed by the "setup" script. One copy of 
# this file is installed in the root directory of the src/ directory
# tree, which is used for in-source compilation. Another copy is 
# installed in the root directory of the bld/ directory tree, which is 
# used for out-of-source compilation.
#
# This file contains user-modifiable definitions of several types of 
# makefile variables:
# 
#  - Variables ROOT_DIR, SRC_DIR, BLD_DIR, BIN_DIR and DAT_DIR that 
#    contain absolute paths for the simpatico root directory and some 
#    of its subdirectories.
#
#  - A variable UTIL_DEBUG that if defined, enables compilation of
#    a debugging version fo the code with additional sanity checks.
#
#  - Variables that control the command name by which the compiler 
#    is invoked and command line options passed to the compiler 
# 
#  - A variable MAKEDEP that enables automatic dependency generation.
#
#=========================================================================
# Variables that define absolute directory paths
#
# In the config.mk file installed in each build directory, correct
# values of these variables should have been set by the setup script,
# and should not need to modified by the user.

# Absolute path to the root simpatico directory
ROOT_DIR=/home/juntong/CSU/pscfplus

# Path to the build directory (location for intermediate generated files)
# This should also be the directory that contains this script.
BLD_DIR=$(ROOT_DIR)/bld

# Path to the source directory (contains C++ source code files)
SRC_DIR=$(ROOT_DIR)/src

# Installation directory for binary executable program files
BIN_DIR=$(ROOT_DIR)/bin

# Directory for shared permanent (read-only) data used by programs.
DAT_DIR=$(ROOT_DIR)/data

#======================================================================
# Conditional compilation of debugging.

# Defining UTIL_DEBUG enables a variety of extra sanity checks, at some
# cost in speed. Debugging is disabled (commented out) by default.
#UTIL_DEBUG=1

# Comment: After setup but before compilation, the above definitions of 
# UTIL_DEBUG may be uncommented or commented out from the command line 
# by using the "configure" script. To do so, invoke the configure script 
# with the -g option prior to compilation from the directory that contains 
# this main configuration file (e.g., from src/ or bld/). Specifically, 
# invoke "./configure -g1" to enable debugging or "./configure -g0" to
# disable debugging. 
#
#======================================================================
# Compiler configuration variables.
#
# The following block of variable definitions is initialized by 
# the setup script by copying a compiler configuration file in the 
# make/compiler directory.  If the setup script is invoked with no 
# argument, the file "make/compiler/default" is used by default.  
# Users may add files to the make/compiler directory to store 
# settings required for a particular environment. The name of a 
# desired compiler configuration file may be specfied by invoking
# the setup script with the base name of the desired file as an 
# argument (e.g., "> ./setup local"). 
#
# Variables defined in this block define the names of the commands 
# used to invoke the compiler when compiling and linking files, some
# of the command line options passed to the compiler to control,
# and search paths for header files and libraries files for required
# external libraries.  See the section of this file entitled "Makefile 
# Patterns and Recipes" for a discussion of how these variables are 
# used.
#
# ---------------------------------------------------------------
# Default compiler configuration file (gcc and nvcc)
#
# The definitions given below work for systems in which:
#
#   - The command g++ is used to invoke the C++ compiler.
#   - The command nvcc is used to invoke the nvidia cuda compiler
#   - Header files and libraries for the CUDA and FFTW libraries
#     are in standard locations.
#   - The gsl-config command can be used to find the correct header
#     and library paths for the Gnu Scientific Library (GSL)
#
# These definitions work in most linux environments and in a
# Mac OSX environment with XCode installed, for which g++ invokes 
# the clang compiler.
#
# ---------------------------------------------------------------
# C++ compiler and options (*.cpp files)

# C++ compiler Command name
CXX=g++

# Compiler option to specify ANSI C++ 2011 standard (required)
CXX_STD = --std=c++11

# Flags always passed to compiler when debugging is enabled
CXXFLAGS_DEBUG= -Wall $(CXX_STD)

# Flags always passed to compiler when debugging is disabled (fast)
CXXFLAGS_FAST= -Wall $(CXX_STD) -O3 -ffast-math -Winline 

# Compiler flags used in unit tests
TESTFLAGS= -Wall $(CXX_STD)

# ---------------------------------------------------------------
# Cuda compiler and options (*.cu files)

 NVXXFLAGS= -O3 -arch=sm_75 -DREPS=1 -DCHN=0 -DDFT=0 -DNBP=0 -DCMP=0


# Cuda compiler command
NVXX=nvcc


# ---------------------------------------------------------------
# Linker / Loader 

# Flags passed to compiler for linking and loading
LDFLAGS=

# ---------------------------------------------------------------
# Archiver

# Library archiver command (for creating static libraries)
AR=ar

# Flags (command line options) passed to archiver
ARFLAGS=rcs

#-----------------------------------------------------------------------
# Paths to header and library files for required external libraries

# Gnu Scientific Library
# Note: Paths may be automatically generated using gsl-config
GSL_INC=-I /usr/include
GSL_LIB=-L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm

# FFTW Fast Fourier transform library
FFTW_INC=
FFTW_LIB=-lfftw3

#JSONCPP library
JSONCPP_INC=
JSONCPP_LIB=-ljsoncpp

# CUDA libraries
# PSSP_CUFFT_PREFIX=/usr/local/cuda
# CUFFT_INC=-I$(PSSP_CUFFT_PREFIX)/include
# CUFFT_LIB=-L$(PSSP_CUFFT_PREFIX)/lib -lcufft -lcudart
# CUFFT_LIB=-lcufft -lcudart -lcuda -lcurand
CUFFT_INC=
CUFFT_LIB= -lcufft -lcudart

# ======================================================================
# General definitions for all systems (Do not modify)

# Assign value of CXX_FLAGS, depending on whether debugging is enabled
ifdef UTIL_DEBUG
   # Flags for serial programs with debugging
   CXXFLAGS=$(CXXFLAGS_DEBUG)
else
   # Flags for serial programs with no debugging
   CXXFLAGS=$(CXXFLAGS_FAST)
endif

# Initialize INCLUDE path for header files (must include SRC_DIR)
# This initial value is added to in the patterns.mk file in each 
# namespace level subdirectory of the src/ directory.
INCLUDES= -I$(SRC_DIR)

# Variable UTIL_CXX11 must be defined to enable use of features of
# the C++ 2011 language standard. 
UTIL_CXX11=1

# ======================================================================
# Makefile Patterns and Recipes
#
# The makefile variables defined above are used in the makefile pattern 
# rules that control compilation of C++ files, creation of libraries, 
# and linking to create executables. The following sections briefly 
# explain these rules, to provide a context for the meaning of the 
# variables defined above.
#
#-----------------------------------------------------------------------
# Compiler Pattern Rules:
#
# The pattern rule for compiling and linking C++ files in a particular 
# top-level subdirectory of the src/ directory is defined in a file
# named patterns.mk in the relevant subdirectory.  For example, the rule 
# for compiling C++ files in the src/fd1d/ directory tree, which contains 
# all entities defined in the Pscf::Fd1d C++ namespace, is given in the 
# file src/fd1d/patterns.mk. The rules for different subdirectories
# of src/ are similar except for differences in which preprocessor 
# variable definitions are passed to the compiler. For each subdirectory
# of src/ namespace, the basic compiler pattern rule for C++ files is of 
# the form:
# 
# $(BLD_DIR)%.o:$(SRC_DIR)/%.cpp
#      $(CXX) $(INCLUDES) $(DEFINES) $(CXXFLAGS) -c -o $@ $<
#
# This pattern compiles a *.cpp file in a subdirectory of the source 
# directory $(SRC_DIR) and creates a *.o object file in a corresponding
# subdirectory of the build directory, $(BLD_DIR). The variables used
# in this pattern are:
#
# CXX         - C++ compiler executable name 
# INCLUDES    - Directories to search for included header files
# DEFINES     - compiler options that define C preprocessor macros
# CXXFLAGS    - compiler options used during compilation
#
# Comments:
# 
# 1) The variable INCLUDES is a string that must include the path 
# $(SRC_DIR) to the simpatico/src directory, in order to allow the 
# compiler to find header files that are part of the package.
#
# 2) The variable DEFINES in the above pattern is a stand-in for a 
# variable that specifies a list of C preprocessor macro definitions. 
# This variable is not defined in this main configuration file, and
# is assigned different values for code in different top-level 
# subdirectories of src/. A value of DEFINES for each such directory
# is assigned in the corresponding pattern.mk file.  The value of 
# $(DEFINES) for each namespace contains a string of compiler 
# options that use the compiler "-D" option to define the set of
# preprocessor macro definitions used to control conditional
# compilation of optional features that are relevant in a particular 
# namespace. Each of these preprocessor variable macros has the
# same name as a corresponding makefile variable that must be defined
# to enable the feature. Thus for, example, when the build system has 
# been configured to enable debugging, the DEFINES string will include
# a substring "-D UTIL_DEBUG" to define the UTIL_DEBUG preprocessor 
# macro and thereby enable conditional compilation of blocks of code
# that contain optional sanity checks.  
#
# 4) The variable $(CXXFLAGS) should specify all flags that are used by 
# the compiler, rather than only the preprocessor, and that are used in
# all namespaces. This string normally contains the $(CXX_STD) string 
# as a substring, as well as options that specify the optimization 
# level (e.g., -O3) and any desired compiler warnings (e.g., "-Wall").
#
#-----------------------------------------------------------------------
# Archiver Recipes:
#
# The simpatico build system creates a static library in each namespace
# level subdirectory of the build directory in which code is compiled.
# The recipe used to compile this library is defined in the sources.mk
# file in the appropriate namespace-level directory. The rule for the
# McMd namespace, as an example, is of the form
#
# $(AR) rcs $(mcMd_LIB) $(mcMd_OBJS)
#
# where $(AR) is the name of archiver command used to create a library,
# $(mcMD_LIB) is an absolute path for the resulting library file and 
# $(mcMd_OBJS) is a string that contains absolute paths for all of the
# *.o object files created by compiling source files in the directory
# src/mcMd. Recipes for other namespaces are analogous.
#
#-----------------------------------------------------------------------
# Linker recipes:
# 
# Executable files are created by linking the compiled main program to
# the required set of static libraries. For example, recipe for creating 
# the mdSim executable is of the form
#
#	$(CXX) -o $(mdSim_EXE) $(mdSim).o $(LIBS) $(LDFLAGS)
#
# Here $(mdSim_EXE) is the path to the executable, which is installed 
# in the bin/ directory by default, $(mdSim).o is the path to the 
# object file created by compiling the src/mcMd/mdSim.cpp source 
# file, $(LIBS) is a list of all required state libraries files, and
# $(LDFLAGS) is a list of flags passed to the linker. 
#
# The variable $(LDFLAGS) is empty by default, but can, if necessary, 
# be used to specify a non-standard path to a directory containing 
# the MPI library when compiling parallel programs. This should not 
# be necessary if the compiler is invoked using the name of a wrapper
# script that sets this path automatically.
 
#=======================================================================
# Automatic dependency generation.
 
# Scripts invoked to compute dependencies among header files.
MAKEDEP=$(BIN_DIR)/makeDepCpp
MAKEDEP_CUDA=$(BIN_DIR)/makeDepCuda

# The file $(BIN_DIR)/makeDepCpp and $(BIN_DIR)/makeDepCuda are executable 
# python scripts that are installed in the binary directory specified by 
# the setup script, and that is used during compilation to analyze 
# dependencies among C++ files. Both script import a python module named 
# pscfpp.makeDepend that is located in the $(ROOT_DIR)/lib/python/pscfpp
# directory. For the python interpreter to find this $(ROOT_DIR)/lib/python
# must be in your $PYTHON_PATH environment variable.# ======================================================================
# General definitions for all systems (Do not modify)

# Assign value of CXX_FLAGS, depending on whether debugging is enabled
ifdef UTIL_DEBUG
   # Flags for serial programs with debugging
   CXXFLAGS=$(CXXFLAGS_DEBUG)
else
   # Flags for serial programs with no debugging
   CXXFLAGS=$(CXXFLAGS_FAST)
endif

# Initialize INCLUDE path for header files (must include SRC_DIR)
# This initial value is added to in the patterns.mk file in each 
# namespace level subdirectory of the src/ directory.
INCLUDES= -I$(SRC_DIR)

# Variable UTIL_CXX11 must be defined to enable use of features of
# the C++ 2011 language standard. 
UTIL_CXX11=1

# ======================================================================
# Makefile Patterns and Recipes
#
# The makefile variables defined above are used in the makefile pattern 
# rules that control compilation of C++ files, creation of libraries, 
# and linking to create executables. The following sections briefly 
# explain these rules, to provide a context for the meaning of the 
# variables defined above.
#
#-----------------------------------------------------------------------
# Compiler Pattern Rules:
#
# The pattern rule for compiling and linking C++ files in a particular 
# top-level subdirectory of the src/ directory is defined in a file
# named patterns.mk in the relevant subdirectory.  For example, the rule 
# for compiling C++ files in the src/fd1d/ directory tree, which contains 
# all entities defined in the Pscf::Fd1d C++ namespace, is given in the 
# file src/fd1d/patterns.mk. The rules for different subdirectories
# of src/ are similar except for differences in which preprocessor 
# variable definitions are passed to the compiler. For each subdirectory
# of src/ namespace, the basic compiler pattern rule for C++ files is of 
# the form:
# 
# $(BLD_DIR)%.o:$(SRC_DIR)/%.cpp
#      $(CXX) $(INCLUDES) $(DEFINES) $(CXXFLAGS) -c -o $@ $<
#
# This pattern compiles a *.cpp file in a subdirectory of the source 
# directory $(SRC_DIR) and creates a *.o object file in a corresponding
# subdirectory of the build directory, $(BLD_DIR). The variables used
# in this pattern are:
#
# CXX         - C++ compiler executable name 
# INCLUDES    - Directories to search for included header files
# DEFINES     - compiler options that define C preprocessor macros
# CXXFLAGS    - compiler options used during compilation
#
# Comments:
# 
# 1) The variable INCLUDES is a string that must include the path 
# $(SRC_DIR) to the simpatico/src directory, in order to allow the 
# compiler to find header files that are part of the package.
#
# 2) The variable DEFINES in the above pattern is a stand-in for a 
# variable that specifies a list of C preprocessor macro definitions. 
# This variable is not defined in this main configuration file, and
# is assigned different values for code in different top-level 
# subdirectories of src/. A value of DEFINES for each such directory
# is assigned in the corresponding pattern.mk file.  The value of 
# $(DEFINES) for each namespace contains a string of compiler 
# options that use the compiler "-D" option to define the set of
# preprocessor macro definitions used to control conditional
# compilation of optional features that are relevant in a particular 
# namespace. Each of these preprocessor variable macros has the
# same name as a corresponding makefile variable that must be defined
# to enable the feature. Thus for, example, when the build system has 
# been configured to enable debugging, the DEFINES string will include
# a substring "-D UTIL_DEBUG" to define the UTIL_DEBUG preprocessor 
# macro and thereby enable conditional compilation of blocks of code
# that contain optional sanity checks.  
#
# 4) The variable $(CXXFLAGS) should specify all flags that are used by 
# the compiler, rather than only the preprocessor, and that are used in
# all namespaces. This string normally contains the $(CXX_STD) string 
# as a substring, as well as options that specify the optimization 
# level (e.g., -O3) and any desired compiler warnings (e.g., "-Wall").
#
#-----------------------------------------------------------------------
# Archiver Recipes:
#
# The simpatico build system creates a static library in each namespace
# level subdirectory of the build directory in which code is compiled.
# The recipe used to compile this library is defined in the sources.mk
# file in the appropriate namespace-level directory. The rule for the
# McMd namespace, as an example, is of the form
#
# $(AR) rcs $(mcMd_LIB) $(mcMd_OBJS)
#
# where $(AR) is the name of archiver command used to create a library,
# $(mcMD_LIB) is an absolute path for the resulting library file and 
# $(mcMd_OBJS) is a string that contains absolute paths for all of the
# *.o object files created by compiling source files in the directory
# src/mcMd. Recipes for other namespaces are analogous.
#
#-----------------------------------------------------------------------
# Linker recipes:
# 
# Executable files are created by linking the compiled main program to
# the required set of static libraries. For example, recipe for creating 
# the mdSim executable is of the form
#
#	$(CXX) -o $(mdSim_EXE) $(mdSim).o $(LIBS) $(LDFLAGS)
#
# Here $(mdSim_EXE) is the path to the executable, which is installed 
# in the bin/ directory by default, $(mdSim).o is the path to the 
# object file created by compiling the src/mcMd/mdSim.cpp source 
# file, $(LIBS) is a list of all required state libraries files, and
# $(LDFLAGS) is a list of flags passed to the linker. 
#
# The variable $(LDFLAGS) is empty by default, but can, if necessary, 
# be used to specify a non-standard path to a directory containing 
# the MPI library when compiling parallel programs. This should not 
# be necessary if the compiler is invoked using the name of a wrapper
# script that sets this path automatically.
 
#=======================================================================
# Automatic dependency generation.
 
# Scripts invoked to compute dependencies among header files.
MAKEDEP=$(BIN_DIR)/makeDepCpp
MAKEDEP_CUDA=$(BIN_DIR)/makeDepCuda

# The file $(BIN_DIR)/makeDepCpp and $(BIN_DIR)/makeDepCuda are executable 
# python scripts that are installed in the binary directory specified by 
# the setup script, and that is used during compilation to analyze 
# dependencies among C++ files. Both script import a python module named 
# pscfpp.makeDepend that is located in the $(ROOT_DIR)/lib/python/pscfpp
# directory. For the python interpreter to find this $(ROOT_DIR)/lib/python
# must be in your $PYTHON_PATH environment variable.
