# include makefiles

SRC_DIR_REL =..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/pgc/include.mk
include $(SRC_DIR)/pgd/include.mk

#-----------------------------------------------------------------------
# Variable definition

PGD_1D=$(BLD_DIR)/pgd/pg1d
PGD_2D=$(BLD_DIR)/pgd/pg2d
PGD_3D=$(BLD_DIR)/pgd/pg3d
PGDD=$(BLD_DIR)/pgd/pg

PGD_EXE=$(PGDD_EXE) $(PGD_3D_EXE) $(PGD_2D_EXE) $(PGD_1D_EXE) 

#-----------------------------------------------------------------------
# Main targets 

all: $(pgd_OBJS) $(pgd_LIB) $(PGD_EXE)

clean:
	rm -f $(pgd_OBJS) $(pgd_OBJS:.o=.d)
	rm -f $(PGD_1D).o $(PGD_1D).d
	rm -f $(PGD_2D).o $(PGD_2D).d
	rm -f $(PGD_3D).o $(PGD_3D).d
	rm -f $(PGDD).o $(PGDDD).d
	rm -f $(pgd_LIB)
	rm -f *.o *.d */*.o */*.d
	#cd tests; $(MAKE) clean

veryclean:
	$(MAKE) clean
	-rm -f lib*.a

# Executable target

$(PGD_1D_EXE): $(PGD_1D).o $(PGD_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGD_1D_EXE) $(PGD_1D).o $(LIBS)

$(PGD_2D_EXE): $(PGD_2D).o $(PGD_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGD_2D_EXE) $(PGD_2D).o $(LIBS)

$(PGD_3D_EXE): $(PGD_3D).o $(PGD_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGD_3D_EXE) $(PGD_3D).o $(LIBS)

$(PGDD_EXE): $(PGDD).o $(PGD_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGDD_EXE) $(PGDD).o $(LIBS)


# Short name for executable target (for convenience)
pgd1d:
	$(MAKE) $(PGD_1D_EXE)

pgd2d:
	$(MAKE) $(PGD_2D_EXE)

pgd3d:
	$(MAKE) $(PGD_3D_EXE)

pgd:
	$(MAKE) $(PGDD_EXE)


#-----------------------------------------------------------------------
# Include dependency files

-include $(pgd_OBJS:.o=.d)
-include $(pspg_OBJS:.o=.d)
-include $(pscf_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
-include $(PGD_1D).d 
-include $(PGD_2D).d 
-include $(PGD_3D).d 
-include $(PGDD).d 
