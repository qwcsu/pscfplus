# Include makefiles

SRC_DIR_REL =../..
include $(SRC_DIR_REL)/config.mk
include $(BLD_DIR)/util/config.mk
include $(BLD_DIR)/pscf/config.mk
include $(BLD_DIR)/pspg/config.mk
include $(SRC_DIR)/pspg/patterns.mk
include $(SRC_DIR)/util/sources.mk
include $(SRC_DIR)/pscf/sources.mk
include $(SRC_DIR)/pspg/sources.mk
include $(BLD_DIR)/pgd/config.mk
include $(SRC_DIR)/pgd/patterns.mk
include $(SRC_DIR)/pgd/sources.mk

#-----------------------------------------------------------------------
# Main targets 

all: $(pgd_solvers_OBJS) 

clean:
	rm -f $(pgd_solvers_OBJS) $(pgd_solvers_OBJS:.o=.d)

veryclean:
	$(MAKE) clean
	-rm -f *.o 
	-rm -f *.d 

#-----------------------------------------------------------------------
# Include dependency files

-include $(pgd_OBJS:.o=.d)
-include $(pgc_OBJS:.o=.d)
-include $(pspg_OBJS:.o=.d)
-include $(pscf_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
-include $(PSCF_FD).d 
