include $(SRC_DIR)/pspg/math/sources.mk
include $(SRC_DIR)/pspg/field/sources.mk

pspg_= $(pspg_field_) $(pspg_math_) 

pspg_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pspg_))
pspg_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pspg_:.cu=.o))

$(pspg_LIB): $(pspg_OBJS)
	$(AR) rcs $(pspg_LIB) $(pspg_OBJS)

