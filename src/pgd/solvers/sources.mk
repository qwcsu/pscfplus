pgd_solvers_= \
     pgd/solvers/DPolymer.cu \
     pgd/solvers/Bond.cu \
     pgd/solvers/DPropagator.cu\
     pgd/solvers/DMixture.cu\
     pgc/solvers/WaveList.cu\

pgd_solvers_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pgd_solvers_))
pgd_solvers_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pgd_solvers_:.cu=.o))