
# Initialize macros to empty strings
PSPG_DEFS=
PSPG_SUFFIX:=

#-----------------------------------------------------------------------
# Path to the pscf library 
# Note: BLD_DIR is defined in config.mk in root of bld directory

pspg_LIBNAME=pspg$(PSCF_SUFFIX)$(UTIL_SUFFIX)
pspg_LIB=$(BLD_DIR)/pspg/lib$(pscf_LIBNAME).a
#-----------------------------------------------------------------------
