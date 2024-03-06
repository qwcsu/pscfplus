include $(SRC_DIR)/pspg/field/sources.mk
include $(SRC_DIR)/pspg/math/sources.mk
include $(SRC_DIR)/pgc/iterator/sources.mk
include $(SRC_DIR)/pgc/solvers/sources.mk

pgc_= \
  $(pspg_field_) \
  $(pgc_solvers_) \
  $(pgc_iterator_) \
  $(pspg_math_) \
  pgc/System.cu

pgc_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pgc_))
pgc_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pgc_:.cu=.o))

$(pgc_LIB): $(pgc_OBJS)
	$(AR) rcs $(pgc_LIB) $(pgc_OBJS)

