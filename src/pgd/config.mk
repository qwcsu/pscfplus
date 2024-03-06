# Initialize macros to empty strings
PGD_DEFS=
PGD_SUFFIX:=

#-----------------------------------------------------------------------
# Path to the dpdpg library 
# Note: BLD_DIR is defined in config.mk

pgd_LIBNAME=pgd$(PGD_SUFFIX)$(UTIL_SUFFIX)
pgd_LIB=$(BLD_DIR)/pgd/lib$(pgd_LIBNAME).a
#-----------------------------------------------------------------------
# Path to executable file

PGD_1D_EXE=$(BIN_DIR)/pg1d$(PGD_SUFFIX)$(UTIL_SUFFIX)
PGD_2D_EXE=$(BIN_DIR)/pg2d$(PGD_SUFFIX)$(UTIL_SUFFIX)
PGD_3D_EXE=$(BIN_DIR)/pg3d$(PGD_SUFFIX)$(UTIL_SUFFIX)
PGDD_EXE=$(BIN_DIR)/pg$(PGD_SUFFIX)$(UTIL_SUFFIX)
#-----------------------------------------------------------------------
