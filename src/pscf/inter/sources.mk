pscf_inter_= \
  pscf/inter/Interaction.cpp \
  pscf/inter/ChiInteraction.cpp 

pscf_inter_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pscf_inter_))
pscf_inter_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pscf_inter_:.cpp=.o))

