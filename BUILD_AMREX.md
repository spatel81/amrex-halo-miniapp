# AMReX/MPI halo-exchange mini-app

A minimal AMReX + MPI application for comparing single-node halo-exchange and
kernel-launch performance between two systems. It builds one `MultiFab` over a
fixed domain decomposition and runs a timed loop of

```
trivial GPU kernel  →  FillBoundary()  (ghost-cell exchange)
```

reporting min/max/avg/stddev per phase, reduced across ranks. No AMR, no
linear solvers, no I/O inside the timed region.

Written for Aurora and Sunspot at ALCF (Intel Data Center GPU Max / PVC,
oneAPI, MPICH, PBS Pro).

**The point of the exercise is an A/B comparison, so the two installs must be
identical:** same AMReX git tag, same build flags, same oneAPI module. Any
difference between them becomes a confound in every number you collect. §7
lists the provenance to capture so that "identical" is something you can prove
afterward rather than assume.

> ⚠ marks items to confirm on-system. Module names, queue names, and wrapper
> script paths change; §8 has commands to check each.

## Contents

| File | Purpose |
|---|---|
| `main.cpp` | the mini-app |
| `GNUmakefile` | links the app against an installed libamrex (§4) |
| `inputs` | default parameters |

---

## 1. Get the source and pin a version

Clone separately on each system, at the same tag:

```bash
export AMREX_TAG=25.08          # same tag on both systems
git clone --branch $AMREX_TAG --depth 1 https://github.com/AMReX-Codes/amrex.git amrex-$AMREX_TAG
cd amrex-$AMREX_TAG && git rev-parse HEAD
```

Pin a release tag, not `development` — a moving branch lets the two systems
diverge silently between builds.

Paths used below:

```bash
export AMREX_SRC=$HOME/src/amrex-$AMREX_TAG
export AMREX_INSTALL=$HOME/opt/amrex-$AMREX_TAG-sycl
```

## 2. Environment

Both systems use the **same oneAPI module**:

```bash
module use /soft/modulefiles
module load oneapi/release/2025.3.1
module list
```

This is not necessarily the default on either system, so load it explicitly
rather than relying on whatever the login environment provides. Loading a
different version on one machine reintroduces the compiler as an uncontrolled
variable — and older/newer `icpx` releases differ in which `-fsycl-*` flags
they accept, so a mismatch can also simply fail to build.

Use the same module at **run time** as at build time.

The C++ compiler is `icpx`, reached through the MPICH wrapper `mpicxx`. AMReX's
SYCL configuration finds these itself; don't override `CXX` unless a build
failure forces it.

## 3. Build and install libamrex

```bash
cd $AMREX_SRC
./configure -h                 # ⚠ confirm flag spellings for this tag

./configure \
  --prefix=$AMREX_INSTALL \
  --dim=3 \
  --comp=intel-llvm \
  --with-mpi=yes \
  --with-omp=no \
  --with-sycl=yes \
  --enable-tiny-profile=yes \
  --debug=no

make -j16
make install
```

`configure` writes a `GNUmakefile` in the AMReX top directory, which you can
edit afterwards. These are the underlying make variables it sets — useful if
you'd rather drive the build directly:

| Make variable | Value | Notes |
|---|---|---|
| `PREFIX` | `$AMREX_INSTALL` | install root |
| `DIM` | `3` | |
| `COMP` | `intel-llvm` | oneAPI `icx`/`icpx` |
| `USE_MPI` | `TRUE` | |
| `USE_OMP` | `FALSE` | pure MPI, one rank per GPU tile |
| `USE_SYCL` | `TRUE` | Intel PVC backend |
| `PRECISION` | `DOUBLE` | |
| `DEBUG` | `FALSE` | |
| `TINY_PROFILE` | `TRUE` | cross-check on the `MPI_Wtime` instrumentation |
| `SYCL_AOT` | `TRUE` | see below |
| `AMREX_INTEL_ARCH` | `pvc` | required when `SYCL_AOT=TRUE` |

**On `SYCL_AOT`:** ahead-of-time compilation for PVC makes the build slower
but removes SPIR-V JIT from the first kernel launch. Worth it for a timing
study — JIT cost otherwise lands inside the run and you're relying on
`n_warmup` to hide it. If you leave it off, leave it off on *both* systems:
JIT cost depends on the Level Zero driver version, so an asymmetric choice
becomes another uncontrolled difference.

Installs to:

- `$AMREX_INSTALL/include/` — headers
- `$AMREX_INSTALL/lib/libamrex.a`
- `$AMREX_INSTALL/lib/pkgconfig/amrex.pc` — the flags AMReX was built with

Check before moving on:

```bash
ls $AMREX_INSTALL/lib/libamrex.a
cat $AMREX_INSTALL/lib/pkgconfig/amrex.pc     # Cflags:/Libs: should show -fsycl and the MPI bits
```

## 4. Build the mini-app

```bash
make AMREX_LIBRARY_HOME=$AMREX_INSTALL
```

The `GNUmakefile` pulls compile and link flags straight out of the installed
`amrex.pc`, so the application inherits exactly what the library was built
with — which is what makes "both machines link an identical AMReX" verifiable
rather than assumed. Produces `main3d.ex`.

Link with the same `mpicxx`/`icpx` that built the library. SYCL device-code
linking fails in confusing ways across a compiler mismatch. ⚠ If the build
complains that the compiler differs from the one recorded in the library,
`configure` has an `--allow-different-compiler` option — prefer fixing the
environment over setting it.

## 5. Running on Aurora

### Interactive, for the first run

```bash
qsub -I -l select=1 -l walltime=60:00 -l filesystems=home:flare -q debug -A <ProjectName>
```

⚠ Confirm the queue and filesystem names for your allocation.

### Rank-to-tile mapping

An Aurora node is:

```
2 × Xeon CPU Max              = 104 physical cores
6 × Data Center GPU Max 1550  = 12 tiles (2 per GPU)
```

**12 ranks per node, one rank per tile.** 104 ÷ 12 ≈ 8 cores per rank, hence
`--depth=8`.

```bash
mpiexec -n 12 -ppn 12 --depth=8 --cpu-bind depth \
  gpu_tile_compact.sh \
  ./main3d.ex n_iter=50 n_warmup=5
```

### Sample Run Batch script

```bash
#!/bin/bash

module use /soft/modulefiles
module load oneapi/release/2025.3.1

WRAP=gpu_tile_compact.sh
LAUNCH="mpiexec -n 12 -ppn 12 --depth=8 --cpu-bind depth $WRAP"

{
  echo "== host =="; hostname
  echo "== modules =="; module list 2>&1
  echo "== env =="; env | grep -E 'MPICH_GPU|ZE_|MPIR_CVAR|FI_'
  echo "== binding =="
  $LAUNCH bash -c 'echo "rank $PMI_RANK mask=$ZE_AFFINITY_MASK"' | sort -V
} > provenance.$PBS_JOBID.txt 2>&1

for rep in 1 2 3; do
  echo "=== repetition $rep ==="
  $LAUNCH ./main3d.ex n_iter=50 n_warmup=5
done > timings.$PBS_JOBID.out 2>&1
```

**Multiple (Three) repetitions .** Establish the run-to-run spread on each
machine first.

## 6. Running on Sunspot

Identical, except the PBS header:

```bash
#PBS -l filesystems=home:tegu
#PBS -q workq
```

⚠ Verify both. Change nothing else — same AMReX tag, same flags, same oneAPI
module, same parameters, same three repetitions.

To isolate GPU-aware MPI as a variable, add one job with
`MPICH_GPU_SUPPORT_ENABLED=0` and nothing else touched.

## 7. Provenance to record per system

Timings without these aren't comparable:

```bash
{
  echo "== host ==";      hostname
  echo "== amrex ==";     git -C $AMREX_SRC rev-parse HEAD; git -C $AMREX_SRC describe --tags
  echo "== modules ==";   module list 2>&1
  echo "== compiler =="; mpicxx -show; icpx --version
  echo "== amrex.pc =="; cat $AMREX_INSTALL/lib/pkgconfig/amrex.pc
  echo "== runtime env =="; env | grep -E 'MPICH_GPU|ZE_|MPIR_CVAR|FI_'
} > build_provenance.$(hostname -s).txt
```

## 8. Verify-on-system checklist

```bash
cd $AMREX_SRC
./configure -h                                   # exact flag names for this tag
sed -n '1,60p' Tools/GNUMake/comps/sycl.mak      # SYCL_AOT / AMREX_INTEL_ARCH / SYCL_AOT_GRF_MODE
module avail oneapi 2>&1 | head -40              # confirm 2025.3.1 is present
ls /soft/tools/mpi_wrapper_utils/                # tile-binding wrapper
```

⚠ On AMReX releases older than ~22.x the SYCL variables were `USE_DPCPP` /
`COMP=dpcpp`. Not an issue on a current tag.

## 9. Reading the output

Check in this order:

1. **`FillBoundary verification: PASSED`** — the first line. The app fills each
   cell with a position-dependent value, exchanges, and asserts every ghost
   cell holds its owner's data (wrapping in the periodic directions) before any
   timing runs. A failure aborts, because a halo exchange that moves nothing
   would otherwise produce fast, plausible, meaningless numbers.
2. **`num boxes`** — should be 64 with the defaults. If it says 1, the domain
   wasn't decomposed: 11 idle ranks and no exchange at all.
3. **`NProcs = 12`**.
4. **`[FillBoundary] avg`** — the headline number for the comparison.
5. **`[FillBoundary] max` vs `avg`** — a large gap is load imbalance, not
   necessarily a fabric effect. 64 boxes over 12 ranks means four ranks hold 6
   boxes and eight hold 5; the lighter ranks wait inside `FillBoundary` for the
   heavier ones, and that wait is charged to the exchange.
6. **The TinyProfile table** — AMReX's own `FillBoundary` timer, an independent
   measurement of the same phase. Material disagreement with the `MPI_Wtime`
   number points at the instrumentation rather than the hardware.

### What the mini-app does and doesn't measure

With the defaults (64³, `max_grid_size=16`, `ngrow=1`, `ncomp=1`), each halo
message is a few KB — squarely in the **latency**-dominated regime — and the
kernel launches 4,096 work-items per box, which barely occupies a PVC tile.

So it measures **MPI message latency, kernel-launch overhead, the GPU-aware
MPI path, and synchronization**. It does *not* measure bandwidth or GPU compute
throughput.
