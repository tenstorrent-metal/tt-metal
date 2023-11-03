MATMUL_MULTI_CORE_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse/matmul_multicore_reuse.cpp

MATMUL_MULTI_CORE_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse.d

-include $(MATMUL_MULTI_CORE_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse
$(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse.o
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse.o: $(MATMUL_MULTI_CORE_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
