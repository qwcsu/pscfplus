pscf_mesh_= \
  pscf/mesh/Mesh.cpp \
  pscf/mesh/MeshIterator.cpp 

pscf_mesh_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pscf_mesh_))
pscf_mesh_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pscf_mesh_:.cpp=.o))

