LOOPBACK_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/loopback/loopback.cpp

LOOPBACK_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/loopback.d

-include $(LOOPBACK_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/loopback
$(PROGRAMMING_EXAMPLES_TESTDIR)/loopback: $(PROGRAMMING_EXAMPLES_OBJDIR)/loopback.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/loopback.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/loopback.o: $(LOOPBACK_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
