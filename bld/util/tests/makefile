BLD_DIR_REL =../..
include $(BLD_DIR_REL)/config.mk
include $(BLD_DIR)/util/config.mk
include $(SRC_DIR)/util/patterns.mk
include $(SRC_DIR)/util/sources.mk
include $(SRC_DIR)/util/tests/sources.mk

TEST=$(BLD_DIR)/util/tests/Test

.SECONDARY:$(util_tests_OBJS)

all: $(util_tests_EXES)

run: $(TEST)
	@echo "BLD_DIR is $(BLD_DIR)"
	@echo "SRC_DIR is $(SRC_DIR)"
	$(TEST) $(SRC_DIR)/util/tests/ 
	@echo "Finished running $(TEST)"
	@cd $(SRC_DIR)/util/tests; $(MAKE) -s clean-outputs
	
quiet: $(TEST)
	@$(TEST) $(SRC_DIR)/util/tests/ > log
	@echo `grep failed log` ", "\
              `grep successful log` "in util/tests/log" > count
	@cat count
	@cd $(SRC_DIR)/util/tests; $(MAKE) -s clean-outputs
	

clean:
	rm -f $(util_tests_EXES) 
	rm -f $(util_tests_OBJS) 
	rm -f $(util_tests_OBJS:.o=.d)
	rm -f log count
	cd $(SRC_DIR)/util/tests; $(MAKE) clean-outputs
ifeq ($(BLD_DIR),$(SRC_DIR))
	cd accumulators/unit; $(MAKE) clean
	cd archives; $(MAKE) clean
	cd containers; $(MAKE) clean
	cd format; $(MAKE) clean
	cd misc; $(MAKE) clean
	cd param; $(MAKE) clean
	cd random; $(MAKE) clean
	cd signal; $(MAKE) clean
	cd space; $(MAKE) clean
	rm -f */*.o */*/*.o 
	rm -f */*.d */*/*.d
endif

clean-outputs:
	@cd accumulators/unit; $(MAKE) clean-outputs
	@cd archives; $(MAKE) clean-outputs
	@cd containers; $(MAKE) clean-outputs
	@cd param/serial; $(MAKE) clean-outputs
	@cd random; $(MAKE) clean-outputs

-include $(util_tests_OBJS:.o=.d)
-include $(util_OBJS:.o=.d)
