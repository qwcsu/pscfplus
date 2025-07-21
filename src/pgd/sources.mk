include $(SRC_DIR)/pspg/field/sources.mk
include $(SRC_DIR)/pspg/math/sources.mk
include $(SRC_DIR)/pgd/solvers/sources.mk
include $(SRC_DIR)/pgd/iterator/sources.mk

pgd_=$(pspg_field_) \
     $(pgc_solvers_) \
     $(pgd_solvers_) \
     $(pgd_iterator_) \
     pgd/System.cu  

pgd_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pgd_))
pgd_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pgd_:.cu=.o))

$(pgd_LIB): $(pgd_OBJS)
	$(AR) rcs $(pgd_LIB) $(pgd_OBJS)