#-----------------------------------------------------------------------
# This makefile fragment defines:
#
#   - A variable $(PSPG_DEFS) that is passed to the processor to define 
#     preprocessor flags that effect the code in the pspg/ directory. 
#
#   - Variable $(PSPG_SUFFIX) that can be added to the name of the
#     pspg library. 
#
#   - Variable $(pspg_LIB) giving the pspg library file path.
#
# This file must be included by every makefile in the pspg directory. 
#-----------------------------------------------------------------------
# Flag to define preprocessor macros.

# Comments:
#
# The variable PSPG_DEFS uses the "-D" compiler option to pass C/C++
# preprocessor definitions to the compiler. If not empty, it must consist
# of a list of zero or more preprocessor macro names, each preceded by the 
# compiler flag "-D".  
#
# The variable PSPG_SUFFIX is appended to the base name pspg.a of 
# the static library $(PSPG_LIB). It is empty by default.
 
# Initialize macros to empty strings
PGC_DEFS=
PGC_SUFFIX:=

#-----------------------------------------------------------------------
# Path to the pssp library 
# Note: BLD_DIR is defined in config.mk

pgc_LIBNAME=pgc$(PGC_SUFFIX)$(UTIL_SUFFIX)
pgc_LIB=$(BLD_DIR)/pgc/lib$(pgc_LIBNAME).a
#-----------------------------------------------------------------------
# Path to executable file

PGC1D_EXE=$(BIN_DIR)/pg1d$(PGC_SUFFIX)$(UTIL_SUFFIX)
PGC2D_EXE=$(BIN_DIR)/pg2d$(PGC_SUFFIX)$(UTIL_SUFFIX)
PGC3D_EXE=$(BIN_DIR)/pg3d$(PGC_SUFFIX)$(UTIL_SUFFIX)
PGCD_EXE=$(BIN_DIR)/pg$(PGC_SUFFIX)$(UTIL_SUFFIX)
#-----------------------------------------------------------------------
