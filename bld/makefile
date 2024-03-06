include config.mk

.PHONY: util pscf pg_c pg_d clean veryclean
# ======================================================================
# Main build targets

# Build all CPU and GPU programs
all:
	$(MAKE) pg_d
	$(MAKE) pg_c

# Build code in Util names (general scientific utilities)
util: 
	cd util; $(MAKE) all

# Build common code for PSCF in Util and Pscf namespaces
pscf: 
	cd util; $(MAKE) all
	cd pscf; $(MAKE) all

# Build pscf_pcNd GPU code for periodic structures (install in BIN_DIR)
pg_c:
	cd util; $(MAKE) all
	cd pscf; $(MAKE) all
	cd pgc; $(MAKE) all

pg_d:
	cd util; $(MAKE) all
	cd pscf; $(MAKE) all
	cd pgd; $(MAKE) all

# ======================================================================
# Clean targets

# Remove object (*.o), dependency (*.d) and library (*.a) files
clean:
	cd util; $(MAKE) clean
	cd pscf; $(MAKE) clean
	cd pgc; $(MAKE) clean
	cd pgd; $(MAKE) clean


# Remove all automatically generated files, recreate initial state
veryclean:
	cd util; $(MAKE) veryclean
	cd pscf; $(MAKE) veryclean
	cd pgc; $(MAKE) veryclean
	cd pgd; $(MAKE) veryclean
	rm -f util/config.mk
	rm -f pscf/config.mk
	rm -f pg_c/config.mk
	rm -f pg_d/config.mk
ifneq ($(BLD_DIR),$(SRC_DIR))
	rm -f util/makefile
	rm -f pscf/makefile
	rm -f pgc/makefile
	rm -f pgd/makefile
	rm -f configure
endif
	rm -f config.mk

# ======================================================================
