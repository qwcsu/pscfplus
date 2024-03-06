pgc_iterator_= \
  pgc/iterator/AmIterator.cu 

  

pgc_iterator_SRCS=\
	  $(addprefix $(SRC_DIR)/, $(pgc_iterator_))
pgc_iterator_OBJS=\
	  $(addprefix $(BLD_DIR)/, $(pgc_iterator_:.cu=.o))

