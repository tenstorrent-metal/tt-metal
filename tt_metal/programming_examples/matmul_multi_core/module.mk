MATMUL_MULTI_CORE_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multi_core/matmul_multi_core.cpp

MATMUL_MULTI_CORE_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multi_core.d

-include $(MATMUL_MULTI_CORE_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multi_core
$(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multi_core: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multi_core.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multi_core.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multi_core.o: $(MATMUL_MULTI_CORE_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
