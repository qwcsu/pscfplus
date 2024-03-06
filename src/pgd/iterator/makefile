#--------------------------------------------------------------------
# Include makefiles

SRC_DIR_REL =../..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/pspg/include.mk
include $(SRC_DIR)/pgc/include.mk
include $(SRC_DIR)/pgd/include.mk

#--------------------------------------------------------------------
# Main targets 

all: $(pgd_iterator_OBJS) 

includes:
	echo $(INCLUDES)

clean:
	rm -f $(pgd_iterator_OBJS) $(pgd_iterator_OBJS:.o=.d) 

#--------------------------------------------------------------------
# Include dependency files

-include $(pgd_OBJS:.o=.d)
