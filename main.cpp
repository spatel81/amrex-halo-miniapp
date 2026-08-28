// main.cpp — Minimal AMReX/MPI mini-app for comparing single-node
// halo-exchange and kernel-launch performance between two systems.
//
// The smallest AMReX skeleton that still exercises the operations most
// likely to be sensitive to fabric and runtime differences: a fixed domain
// decomposition, a trivial GPU kernel, and FillBoundary() (halo exchange)
// in a tight, timed loop.
//
// NOT included on purpose: AMR regridding, linear solvers, application
// physics, particles, plotfile I/O in the timed region. Those are separate
// concerns and would only muddy the comparison.
//
// See BUILD_AMREX.md for build and run instructions.

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <mpi.h>

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>

using namespace amrex;

namespace {

// Simple running-stats helper (min/max/avg/stddev) over a rank-local
// sample of per-iteration timings, reduced across ranks.
struct TimingStats {
    double min_val = 0.0, max_val = 0.0, avg_val = 0.0, stddev_val = 0.0;
};

// Globally unique value for a cell, used only by the FillBoundary
// correctness check. Encoded as an exact integer in double (safe well past
// any n_cell we would run here), so the post-exchange comparison can be a
// bit-exact equality test — FillBoundary copies bytes, it does no
// arithmetic, so there is no rounding to tolerate.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real pattern_value (int i, int j, int k, int n, int ncell) noexcept
{
    const amrex::Long L = static_cast<amrex::Long>(ncell);
    const amrex::Long idx = static_cast<amrex::Long>(i)
        + L * (static_cast<amrex::Long>(j)
        + L * (static_cast<amrex::Long>(k)
        + L *  static_cast<amrex::Long>(n)));
    return static_cast<amrex::Real>(1 + idx);
}

TimingStats reduce_timings(const std::vector<double>& local_times)
{
    // Local avg/stddev first, over this rank's own iterations.
    //
    // The MPI_Reduce calls below are collective and must therefore run on
    // every rank unconditionally — an early return on an empty sample would
    // deadlock. Guard the local arithmetic instead.
    double local_min = 0.0, local_max = 0.0, local_avg = 0.0, local_stddev = 0.0;

    if (!local_times.empty()) {
        const double n_samples = static_cast<double>(local_times.size());

        double local_sum = std::accumulate(local_times.begin(), local_times.end(), 0.0);
        local_avg = local_sum / n_samples;

        double local_sq_sum = 0.0;
        for (double t : local_times) {
            local_sq_sum += (t - local_avg) * (t - local_avg);
        }
        local_stddev = std::sqrt(local_sq_sum / n_samples);

        local_min = *std::min_element(local_times.begin(), local_times.end());
        local_max = *std::max_element(local_times.begin(), local_times.end());
    }

    // Reduce across ranks: min-of-mins, max-of-maxes, avg-of-avgs,
    // avg-of-stddevs (a rough but adequate summary for scoping purposes —
    // refine later with a proper histogram if needed).
    //
    // MPI_Reduce fills the receive buffer on the root rank only, so these
    // must be initialized — otherwise the reads below are indeterminate on
    // every other rank.
    double global_min = 0.0, global_max = 0.0;
    double global_avg_sum = 0.0, global_stddev_sum = 0.0;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_avg, &global_avg_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stddev, &global_stddev_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int nranks = ParallelDescriptor::NProcs();
    TimingStats out;
    out.min_val = global_min;
    out.max_val = global_max;
    out.avg_val = global_avg_sum / nranks;
    out.stddev_val = global_stddev_sum / nranks;
    return out;
}

void print_stats(const std::string& label, const TimingStats& s)
{
    if (ParallelDescriptor::IOProcessor()) {
        amrex::Print() << label
                        << "  min=" << s.min_val << " s"
                        << "  max=" << s.max_val << " s"
                        << "  avg=" << s.avg_val << " s"
                        << "  stddev(avg-of-local)=" << s.stddev_val << " s"
                        << "\n";
    }
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // ---- Parameters (override via inputs file or command line) ----
        // Defaults: 64^3 domain, periodic in y/z (dim 1,2), non-periodic in
        // x (dim 0), sized so a 12-rank run gets a box count and rank
        // distribution representative of a production-scale case.
        int n_cell = 64;
        int max_grid_size = 16;       // Deliberately small: on a 64^3 domain this
                                      // yields 4^3 = 64 boxes, so a 12-rank run
                                      // actually exercises inter-rank halo
                                      // exchange. At 64 there would be one box,
                                      // 11 idle ranks, and no exchange at all.
        int ngrow = 1;                 // ghost cells; typical stencil width
        int n_iter = 50;                // timed iterations
        int n_warmup = 5;               // untimed warm-up iterations
        int ncomp = 1;                  // number of components in the MultiFab

        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
            pp.query("ngrow", ngrow);
            pp.query("n_iter", n_iter);
            pp.query("n_warmup", n_warmup);
            pp.query("ncomp", ncomp);
        }

        // ---- Domain / BoxArray / DistributionMapping ----
        Box domain_box(IntVect(0,0,0), IntVect(n_cell-1, n_cell-1, n_cell-1));
        BoxArray ba(domain_box);
        ba.maxSize(max_grid_size);

        DistributionMapping dm{ba};

        // Periodicity: x non-periodic (inflow/outflow), y/z periodic.
        Array<int,AMREX_SPACEDIM> is_periodic{0, 1, 1};
        RealBox real_box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
        Geometry geom(domain_box, &real_box, CoordSys::cartesian, is_periodic.data());

        if (ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "=== AMReX/MPI minimal reproducer ===\n"
                            << "NProcs        = " << ParallelDescriptor::NProcs() << "\n"
                            << "n_cell        = " << n_cell << "\n"
                            << "max_grid_size = " << max_grid_size << "\n"
                            << "num boxes     = " << ba.size() << "\n"
                            << "ngrow         = " << ngrow << "\n"
                            << "ncomp         = " << ncomp << "\n"
                            << "n_warmup      = " << n_warmup << "\n"
                            << "n_iter        = " << n_iter << "\n";
        }

        // ---- MultiFab (GPU-resident by default via The_Arena()) ----
        MultiFab mf(ba, dm, ncomp, ngrow);

        // ---- Correctness check (untimed) ----
        //
        // Prove FillBoundary is actually moving data before trusting any
        // timing taken from it. A uniform fill cannot do this: if every cell
        // holds the same value, a halo exchange that silently transferred
        // nothing would leave the MultiFab bit-identical to one that worked.
        //
        // So: give every valid cell a globally unique, position-dependent
        // value and leave the ghosts at a sentinel. After the exchange, each
        // ghost cell that has an owner must hold exactly that owner's value
        // (wrapping in y/z, which are periodic), and each ghost cell with no
        // owner — outside the domain in the non-periodic x direction — must
        // still hold the sentinel. Valid cells are checked too, to catch an
        // exchange that overwrites data it should not touch.
        {
            constexpr Real sentinel = -1.0;

            mf.setVal(sentinel);
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                Array4<Real> const& a = mf.array(mfi);
                amrex::ParallelFor(bx, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
                    a(i,j,k,n) = pattern_value(i, j, k, n, n_cell);
                });
            }
            Gpu::synchronize();

            mf.FillBoundary(geom.periodicity());
            Gpu::synchronize();

            // Flag mismatches per cell, then take a global max — nonzero
            // means at least one cell somewhere is wrong.
            MultiFab err(ba, dm, 1, ngrow);
            err.setVal(0.0);
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const Box& gbx = mfi.fabbox();          // valid + ghosts
                Array4<Real const> const& a = mf.const_array(mfi);
                Array4<Real> const& e = err.array(mfi);
                amrex::ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    // y and z wrap (periodic); x does not.
                    const int jj = ((j % n_cell) + n_cell) % n_cell;
                    const int kk = ((k % n_cell) + n_cell) % n_cell;
                    const bool has_owner = (i >= 0 && i < n_cell);

                    Real bad = 0.0;
                    for (int n = 0; n < ncomp; ++n) {
                        const Real expected = has_owner
                            ? pattern_value(i, jj, kk, n, n_cell)
                            : sentinel;
                        if (a(i,j,k,n) != expected) { bad = 1.0; }
                    }
                    e(i,j,k,0) = bad;
                });
            }
            Gpu::synchronize();

            if (err.norm0(0, ngrow) > 0.0) {
                amrex::Abort("FillBoundary verification FAILED: the halo "
                             "exchange is not delivering the expected data. "
                             "Timings from this build would be meaningless.");
            }
            amrex::Print() << "FillBoundary verification: PASSED\n";
        }

        mf.setVal(1.0);

        std::vector<double> kernel_times;
        std::vector<double> fillboundary_times;
        std::vector<double> iter_times;
        kernel_times.reserve(n_iter);
        fillboundary_times.reserve(n_iter);
        iter_times.reserve(n_iter);

        auto run_kernel = [&](MultiFab& fab) {
            // Trivial GPU kernel: cheap arithmetic op on every valid cell.
            // Intentionally NOT representative of real application physics
            // cost — just enough work to simulate "do work, then
            // communicate" without the kernel dominating the timing.
            for (MFIter mfi(fab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                Array4<Real> const& a = fab.array(mfi);
                amrex::ParallelFor(bx, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept {
                    a(i,j,k,n) = a(i,j,k,n) * 1.0000001 + 1.0e-12;
                });
            }
            Gpu::synchronize();
        };

        // ---- Warm-up (untimed): first-touch, allocation, JIT, etc. ----
        for (int w = 0; w < n_warmup; ++w) {
            run_kernel(mf);
            mf.FillBoundary(geom.periodicity());
        }
        ParallelDescriptor::Barrier();

        // ---- Timed loop ----
        for (int it = 0; it < n_iter; ++it) {
            ParallelDescriptor::Barrier();
            double t_iter0 = MPI_Wtime();

            double t_k0 = MPI_Wtime();
            run_kernel(mf);
            double t_k1 = MPI_Wtime();

            double t_fb0 = MPI_Wtime();
            mf.FillBoundary(geom.periodicity());
            // FillBoundary's pack/unpack are GPU kernels and the call can
            // return with device work still in flight. Without this sync the
            // phase measures "time to queue the exchange", and the remainder
            // spills into the next iteration's kernel timing. No-op if AMReX
            // already synchronized internally; cheap insurance either way.
            Gpu::synchronize();
            double t_fb1 = MPI_Wtime();

            double t_iter1 = MPI_Wtime();

            kernel_times.push_back(t_k1 - t_k0);
            fillboundary_times.push_back(t_fb1 - t_fb0);
            iter_times.push_back(t_iter1 - t_iter0);
        }

        // ---- Report ----
        TimingStats kernel_stats = reduce_timings(kernel_times);
        TimingStats fb_stats = reduce_timings(fillboundary_times);
        TimingStats iter_stats = reduce_timings(iter_times);

        print_stats("[kernel]        ", kernel_stats);
        print_stats("[FillBoundary]  ", fb_stats);
        print_stats("[full iteration]", iter_stats);
    }
    amrex::Finalize();
    return 0;
}
