#-----------------------------------------------------------------------
# This makefile fragment defines:
#
#   - A variable $(PSCF_DEFS) that is passed to the processor to define 
#     preprocessor flags that effect the code in the pscf/ directory. 
#
#   - A variable $(PSCF_SUFFIX) that indicates what other features are
#     enabled, which is also added after PSCF_MPI_SUFFIX to the file
#     name of pscf library. 
#
#   - A variable $(PSCF_LIB) that the absolute path to the pscf library 
#     file.
#
# This file must be included by every makefile in the pscf directory. 
#-----------------------------------------------------------------------
# Comments:
#
# The variable PSCF_DEFS uses the "-D" compiler option to used to pass 
# C/C++ preprocessor definitions to the compiler. If not empty, it must 
# consist of a list of zero or more preprocessor macro names, each 
# preceded by the compiler flag "-D". 
#
# The variable PSCF_DEFS is a recursive (normal) makefile variable, and
# may be extended using the += operator, e.g., PSCF_DEFS+=-DPSCF_THING.
# PSCF_SUFFIX is instead a non-recursive makefile variable, which may 
# be extended using the := operator, as PSCF_SUFFIX:=$(PSCF_SUFFIX)_g. 
# They are defined differently because the += operator for recursive
# variables adds a white space before an added string, which is 
# appropriate for PSCF_DEFS, but not for PSCF_SUFFIX. 
 
# Initialize macros to empty strings
PSCF_DEFS=
PSCF_SUFFIX:=

#-----------------------------------------------------------------------
# Path to the pscf library 
# Note: BLD_DIR is defined in config.mk in root of bld directory

pscf_LIBNAME=pscf$(PSCF_SUFFIX)$(UTIL_SUFFIX)
pscf_LIB=$(BLD_DIR)/pscf/lib$(pscf_LIBNAME).a
#-----------------------------------------------------------------------
