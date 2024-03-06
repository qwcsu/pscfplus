pgc_solvers_= \
     pgc/solvers/WaveList.cu \
     pgc/solvers/Propagator.cu \
     pgc/solvers/Block.cu \
     pgc/solvers/Polymer.cu \
     pgc/solvers/Mixture.cu \
     pgc/solvers/Joint.cu 

pgc_solvers_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pgc_solvers_))
pgc_solvers_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pgc_solvers_:.cu=.o))

