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
//   - [Residues] and [CountResidues] — a straightforward Goldstein-style residue
//     (branch-cut charge) detector. Residue-free maps are guaranteed to unwrap
//     exactly along any spanning path; residues signal genuinely ambiguous
//     (under-sampled or noisy) regions.
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
//   - Branch-cut / Goldstein path-following that places cuts between residues
//     of opposite sign. [Residues] detects residues but no cut placement or
//     integration around cuts is performed.
//   - Minimum L-p norm (least-squares, weighted least-squares, L0/L1)
//     unwrapping via FFT/DCT Poisson solvers or iterative solvers.
//   - Network-flow / minimum-cost-flow global optimisation.
//   - Temporal (multi-frequency) and 3-D volumetric unwrapping.
//   - OpenCV's exact fixed-point histogram bin layout and its mask/shadow
//     handling; this port uses float64 throughout and treats every pixel as
//     valid.
//
// The quality-guided histogram method is the required, fully working core; the
// items above are the documented gaps relative to the wider literature.
package phase_unwrapping
