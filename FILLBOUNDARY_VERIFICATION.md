# The FillBoundary verification check

An explanation of the correctness check at `main.cpp:176-240`, which runs once
before the timed loop and aborts the run if the halo exchange is not delivering
the data it should.

## The problem it solves

The timed loop calls `FillBoundary` on a MultiFab that has been uniformly
filled (`mf.setVal(1.0)`). That is fine for timing — bytes are bytes — but it
makes the exchange **unfalsifiable**. If a build, a rank mapping, or a
GPU-aware MPI path silently transferred nothing, every cell would still read
`1.0` afterward and the MultiFab would be bit-identical to one where the
exchange worked perfectly.

The result would be fast, plausible, and completely meaningless numbers — and
the fast ones would look like the good result.

So the check runs once, before timing, with data that can actually distinguish
the two cases.

## How it works

### 1. Make every cell distinguishable

`mf.setVal(sentinel)` (`-1.0`) paints everything including ghost cells, then a
kernel overwrites only `validbox()` — the owned cells — with
`pattern_value(i,j,k,n)`.

That helper flattens the 4-D index into

```
1 + i + N*(j + N*(k + N*n))
```

so every cell in the global domain gets a unique number. Ghost cells are left
at `-1.0`; pattern values start at 1, so the two can never collide.

### 2. Exchange

```cpp
mf.FillBoundary(geom.periodicity());
```

### 3. Check every cell, valid and ghost

Iterating over `fabbox()` (valid + ghosts), each cell computes what it *should*
hold:

```cpp
jj = ((j % n_cell) + n_cell) % n_cell     // y wraps
kk = ((k % n_cell) + n_cell) % n_cell     // z wraps
has_owner = (0 <= i < n_cell)             // x does not wrap
```

The double-modulo is there because C's `%` keeps the sign of the dividend: a
ghost at `j = -1` gives `-1 % 64 == -1`, and `((-1)+64)%64 == 63` is the
periodic image you want. A ghost at `j = 64` maps to `0`.

Then:

- **`has_owner` true** — the cell must equal `pattern_value(i, jj, kk, n)`, the
  value its owner wrote. This covers ghosts filled over MPI, ghosts filled by a
  same-rank local copy, and the valid cells themselves.
- **`has_owner` false** — the cell is off the end of the domain in the
  non-periodic x direction. Nothing owns it, so it must still be `-1.0`.

That second branch is the half that usually gets skipped, and it is what
catches an exchange writing into regions it has no business touching. Including
the valid cells in the sweep catches the same class of bug from the other side.

Comparison is exact `!=`, not a tolerance. `FillBoundary` copies bytes and does
no arithmetic, so there is nothing to round; and `1 + idx` maxes out at 262,144
at the default `n_cell = 64`, exactly representable in a double.

### 4. Reduce and abort

Rather than aborting from inside a device kernel, each cell writes `0.0` or
`1.0` into a scratch MultiFab, and `err.norm0(0, ngrow)` takes a max over all
cells *and* all ranks — `norm0` reduces globally. Nonzero means at least one
cell somewhere is wrong, and the run aborts instead of printing timings.

On success the app prints:

```
FillBoundary verification: PASSED
```

## What it does and does not cover

It proves the exchange delivered correct data **once**, in this configuration,
on this build. That is the thing worth proving: it is a build/config sanity
gate, and it runs before timing so its own cost is not measured.

It does *not* verify the timed loop — after the check, `setVal(1.0)` restores a
uniform field. Iteration 37 could do something strange and this would not
notice.

It also says nothing about whether the kernel computes anything sensible, which
is deliberate: the kernel is a placeholder standing in for launch overhead, not
a physics calculation.
