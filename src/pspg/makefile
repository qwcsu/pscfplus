#-----------------------------------------------------------------------
# The copy of this namespace-level makefile in the src/ directory is 
# copied to the bld/serial and bld/parallel directories by the setup
# script to create the copies in those directories. Only the copy in
# the src/ directory is stored in the repository.
#-----------------------------------------------------------------------
# Include makefiles

SRC_DIR_REL =..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/pspg/include.mk

#-----------------------------------------------------------------------
# Main targets 

all: $(pspg_OBJS) $(pspg_LIB)

clean:
	rm -f $(pscf_OBJS) $(pscf_OBJS:.o=.d) $(pscf_LIB)
	cd tests; $(MAKE) clean

veryclean:
	$(MAKE) clean
	-rm -f *.o */*.o
	-rm -f *.d */*.d
	-rm -f lib*.a

#-----------------------------------------------------------------------
# Include dependency files

-include $(pspg_OBJS:.o=.d)
