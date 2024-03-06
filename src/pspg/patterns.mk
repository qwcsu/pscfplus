# Local pscf-specific libraries needed in src/pspg
PSPG_LIBS=$(pspg_LIB) $(pscf_LIB) $(util_LIB)

# All libraries needed in executables built in src/pssp
LIBS=$(PSPG_LIBS)

# Add paths to Gnu scientific library (GSL)
INCLUDES+=$(GSL_INC)
LIBS+=$(GSL_LIB) 

# Add paths to Jsoncpp (GSL)
INCLUDES+=$(JSONCPP_INC)
LIBS+=$(JSONCPP_LIB) 

# Add paths to Cuda FFT library
PSPG_DEFS+=-DPGC_FFTW -DGPU_OUTER
INCLUDES+=$(CUFFT_INC)
LIBS+=$(CUFFT_LIB)

# Preprocessor macro definitions needed in src/pspg
DEFINES=$(UTIL_DEFS) $(PSCF_DEFS) $(PSPG_DEFS) 

# Dependencies on build configuration files
MAKE_DEPS=  -A$(BLD_DIR)/config.mk
MAKE_DEPS+= -A$(BLD_DIR)/util/config.mk
MAKE_DEPS+= -A$(BLD_DIR)/pscf/config.mk
MAKE_DEPS+= -A$(BLD_DIR)/pspg/config.mk


# Pattern rule to compile *.cu class source files in src/pssp
$(BLD_DIR)/%.o:$(SRC_DIR)/%.cu
	$(NVXX) $(CPPFLAGS) $(NVXXFLAGS) $(INCLUDES) $(DEFINES) -c -o $@ $<
ifdef MAKEDEP_CUDA
	$(MAKEDEP_CUDA) $(INCLUDES) $(DEFINES) $(MAKE_DEPS) -S$(SRC_DIR) -B$(BLD_DIR) $<
endif

