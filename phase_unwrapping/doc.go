// Package phase_unwrapping implements two-dimensional phase unwrapping on top
// of the root cv package, mirroring a useful subset of OpenCV's
// phase_unwrapping contrib module (whose principal class is
// HistogramPhaseUnwrapping).
//
// Phase unwrapping is the problem of recovering a continuous ("absolute") phase
// surface phi from a measured phase map that is only known modulo 2*pi, i.e.
// wrapped into the principal interval (-pi, pi]. Wrapping is the operation
//
//	psi(x, y) = atan2(sin(phi), cos(phi)) in (-pi, pi]
//
// and unwrapping inverts it by adding, per pixel, the integer multiple of 2*pi
// that makes the surface continuous. This is only possible up to a single
// global additive constant that is a multiple of 2*pi, because wrapping
// discards the absolute offset. All functions therefore return a result whose
// values match the true surface up to such a constant.
//
// # What it provides
//
//   - [HistogramPhaseUnwrapping] — a quality-guided, path-following unwrapper
//     following Herraez, Burton, Lalor and Gdeisat, "Fast two-dimensional
//     phase-unwrapping algorithm based on sorting by reliability following a
//     noncontinuous path" (Applied Optics, 2002). Every pixel is assigned a
//     reliability derived from the second differences of the wrapped phase in
//     its 3x3 neighbourhood. Edges between neighbouring pixels are ordered by
//     the combined reliability of their endpoints using a two-level histogram
//     (the technique that gives the class its name) and processed most-reliable
//     first, merging pixels into groups whose relative 2*pi offsets are fixed
//     as the groups join. The resolve order steers the unwrapping path around
//     unreliable (noisy) regions, so localized noise does not corrupt the
//     whole map.
//   - [Params] and [DefaultParams] — configuration mirroring OpenCV's
//     Params (Width, Height, HistogramThresh, NrOfSmallBins, NrOfLargeBins).
//   - [HistogramPhaseUnwrapping.GetInverseReliabilityMap] — the per-pixel
//     inverse-reliability image (high values = less reliable), matching
//     OpenCV's getInverseReliabilityMap.
//   - [Wrap] / [WrapPhaseMap] — the wrapping operator and a helper that wraps a
//     whole continuous surface, useful for constructing test inputs.
//   - [ReliabilityMap] — a standalone quality/reliability helper exposing the
//     same second-difference measure used to guide the path.
//   - [Residues], [CountResidues] and [ResidueList] — a straightforward
//     Goldstein-style residue (branch-cut charge) detector, as a dense grid, a
//     count and an explicit list. Residue-free maps are guaranteed to unwrap
//     exactly along any spanning path; residues signal genuinely ambiguous
//     (under-sampled or noisy) regions.
//
// # Additional unwrapping methods
//
// Beyond the histogram core this package provides a family of standard 2-D
// unwrappers, all deterministic and all recovering a residue-free surface exactly
// (up to a global 2*pi constant):
//
//   - [GoldsteinBranchCut] — residue detection, opposite-polarity branch-cut
//     placement (with border connection for unbalanced residues) and flood-fill
//     integration that never crosses a cut. The result is always congruent to the
//     input. [BranchCuts] exposes the cut mask on its own.
//   - [LeastSquaresUnwrap] / [LeastSquaresUnwrapMat] — unweighted minimum-norm
//     unwrapping solving the Neumann-boundary Poisson equation exactly with the
//     2-D discrete cosine transform (Ghiglia and Pritt).
//   - [WeightedLeastSquaresUnwrap] — weighted least squares via preconditioned
//     conjugate gradients using the DCT Poisson solver as preconditioner, letting
//     unreliable pixels be down-weighted or masked out.
//   - [FlynnMinimumDiscontinuity] — minimisation of the total weighted 2*pi
//     discontinuity of the result by local integer moves over a path-integrated
//     start, in the spirit of Flynn's algorithm; [TotalDiscontinuity] scores the
//     objective.
//   - [QualityGuidedUnwrap] and [MaskedUnwrap] — priority-queue quality-guided
//     path following over a reliability map, optionally restricted to a boolean
//     mask (masked-out pixels become NaN).
//   - [TemporalUnwrap] — temporal (multi-wavelength / multi-frequency) unwrapping
//     that resolves each pixel independently from a coarse unambiguous map to a
//     fine one, with no spatial error propagation.
//   - [Unwrapper] — a common interface unifying every method, with adapters
//     [HistogramUnwrapper], [LeastSquaresUnwrapper], [WeightedLeastSquaresUnwrapper],
//     [GoldsteinUnwrapper], [FlynnUnwrapper] and [QualityGuidedUnwrapper].
//
// # Quality maps and operators
//
//   - [PhaseDerivativeVariance], [MaximumPhaseGradient] and [PseudoCorrelation] —
//     the standard Ghiglia and Pritt quality maps (the first two are cost maps
//     where lower is better; the last is a reliability map where higher is
//     better) that drive the quality-guided and weighted methods.
//   - [WrapToPi] — the canonical scalar wrapping operator; [Rewrap] re-wraps a
//     whole unwrapped surface; [Congruence] snaps a non-congruent estimate (from
//     least squares, say) onto the surface closest to it that re-wraps to the
//     input.
//
// # Input and output conventions
//
// Coordinates follow the image convention used throughout the root package: the
// first index is the row (Y, vertical) and the second is the column (X,
// horizontal). Wrapped phase is accepted either as a single-channel
// [github.com/malcolmston/opencv.FloatMat] (via
// [HistogramPhaseUnwrapping.UnwrapPhaseMap]) or as a [][]float64 grid indexed
// [row][col] (via [HistogramPhaseUnwrapping.UnwrapPhaseMapGrid]). Wrapped
// values are expected in (-pi, pi]; inputs outside that range are wrapped
// defensively before processing. The unwrapped result has the same shape as the
// input.
//
// # Determinism
//
// Every function is deterministic: identical inputs yield identical outputs.
// There is no reliance on randomness, map iteration order or wall-clock time.
// Edges are generated in a fixed row-major order and the histogram buckets
// preserve insertion order, so the exact resolve sequence — and therefore the
// exact result — is reproducible. For a residue-free surface the unwrapped
// result is independent of the resolve order and is recovered exactly (up to
// the unavoidable global 2*pi constant).
//
// # Deferred / out of scope
//
// The following related techniques are intentionally not implemented:
//
//   - Exact network-flow / minimum-cost-flow global L0/L1 optimisation. Discrete
//     discontinuities are minimised by the local search in
//     [FlynnMinimumDiscontinuity] rather than a global min-cost-flow solver.
//   - 3-D volumetric and phase-shifting-specific unwrapping.
//   - OpenCV's exact fixed-point histogram bin layout and its mask/shadow
//     handling; this port uses float64 throughout and, except where a mask is
//     passed explicitly to [MaskedUnwrap], treats every pixel as valid.
//
// The quality-guided histogram method is the required, fully working core; the
// additional branch-cut, least-squares, weighted least-squares, minimum-
// discontinuity, quality-guided, masked and temporal methods round out the
// classical 2-D unwrapping toolbox, and the items above are the remaining gaps
// relative to the wider literature.
package phase_unwrapping
