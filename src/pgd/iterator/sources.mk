pgd_iterator_= \
  pgd/iterator/AmIterator.cu \
  pgc/iterator/Iterator.cu 

  

pgd_iterator_SRCS=\
	  $(addprefix $(SRC_DIR)/, $(pgd_iterator_))
pgd_iterator_OBJS=\
	  $(addprefix $(BLD_DIR)/, $(pgd_iterator_:.cu=.o))
