#-----------------------------------------------------------------------
# The copy of this namespace-level makefile in the src/ directory is 
# copied to the bld/serial and bld/parallel directories by the setup
# script to create the copies in those directories. Only the copy in
# the src/ directory is stored in the repository.
#-----------------------------------------------------------------------
# Include makefiles

SRC_DIR_REL =..
include $(SRC_DIR_REL)/config.mk
include $(SRC_DIR)/pgc/include.mk

#-----------------------------------------------------------------------
# Variable definition

PGC1D=$(BLD_DIR)/pgc/pg1d
PGC2D=$(BLD_DIR)/pgc/pg2d
PGC3D=$(BLD_DIR)/pgc/pg3d
PGCD=$(BLD_DIR)/pgc/pg

PGC_EXE=$(PGCD_EXE) $(PGC3D_EXE) $(PGC2D_EXE) $(PGC1D_EXE) 
#-----------------------------------------------------------------------
# Main targets 

all: $(pgc_OBJS) $(pgc_LIB) $(PGC_EXE)


clean:
	rm -f $(pgc_OBJS) $(pgc_OBJS:.o=.d)
	rm -f $(PGC1D).o $(PGC1D).d
	rm -f $(PGC2D).o $(PGC2D).d
	rm -f $(PGC3D).o $(PGC3D).d
	rm -f $(PGCD).o $(PGCD).d

veryclean:
	$(MAKE) clean
	-rm -f lib*.a

# Executable target

$(PGC1D_EXE): $(PGC1D).o $(PGC_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGC1D_EXE) $(PGC1D).o $(LIBS)

$(PGC2D_EXE): $(PGC2D).o $(PGC_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGC2D_EXE) $(PGC2D).o $(LIBS)

$(PGC3D_EXE): $(PGC3D).o $(PGC_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGC3D_EXE) $(PGC3D).o $(LIBS)

$(PGCD_EXE): $(PGCD).o $(PGC_LIBS)
	$(CXX) $(LDFLAGS) -o $(PGCD_EXE) $(PGCD).o $(LIBS)


# Short name for executable target (for convenience)
pgc1d:
	$(MAKE) $(PGC1D_EXE)

pgc2d:
	$(MAKE) $(PGC2D_EXE)

pgc3d:
	$(MAKE) $(PGC3D_EXE)

pgc:
	$(MAKE) $(PGCD_EXE)



#-----------------------------------------------------------------------
# Include dependency files

-include $(pgc_OBJS:.o=.d)
-include $(pscf_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
-include $(PGC1D).d 
-include $(PGC3D).d 
-include $(PGC3D).d 
-include $(PGCD).d 
