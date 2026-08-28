# GNUmakefile — link the mini-app against an installed libamrex.
#
#   make AMREX_LIBRARY_HOME=/path/to/amrex-install
#
# Compile and link flags are taken straight from the installed amrex.pc, so
# the application inherits exactly what the library was built with. Build
# libamrex first — see README.md.
#
# Link with the same mpicxx/icpx that built the library; SYCL device-code
# linking fails confusingly across a compiler mismatch. CXX is set below
# rather than left to make's g++ default, and can still be overridden on the
# command line (make CXX=...).
#
# GPU-aware MPI is an environment/launch setting, not a build flag (e.g.
# MPICH_GPU_SUPPORT_ENABLED=1); record it alongside the results for each run.

AMREX_LIBRARY_HOME ?= $(AMREX_INSTALL)

ifeq ($(strip $(AMREX_LIBRARY_HOME)),)
  $(error Set AMREX_LIBRARY_HOME (or AMREX_INSTALL) to the libamrex install prefix)
endif

LIBDIR := $(AMREX_LIBRARY_HOME)/lib
INCDIR := $(AMREX_LIBRARY_HOME)/include
PKGDIR := $(LIBDIR)/pkgconfig

COMPILE_CPP_FLAGS := $(shell pkg-config --cflags $(PKGDIR)/amrex.pc)
COMPILE_LIB_FLAGS := $(shell pkg-config --libs   $(PKGDIR)/amrex.pc)

CXX      := mpicxx
CXXFLAGS := -I$(INCDIR) $(COMPILE_CPP_FLAGS)
LFLAGS   := -L$(LIBDIR) $(COMPILE_LIB_FLAGS)

main3d.ex: main.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(LFLAGS)

clean:
	$(RM) main3d.ex
.PHONY: clean
