pscf_tests_inter_=pscf/tests/inter/Test.cc

pscf_tests_inter_SRCS=\
     $(addprefix $(SRC_DIR)/, $(pscf_tests_inter_))
pscf_tests_inter_OBJS=\
     $(addprefix $(BLD_DIR)/, $(pscf_tests_inter_:.cc=.o))

