// Package inpaint is a standard-library-only image inpainting, restoration and
// gradient-domain editing toolkit built on top of the root cv package
// (github.com/malcolmston/opencv). It reconstructs missing or corrupted image
// regions from their surroundings and blends imagery seamlessly.
//
// The package operates on the root package's [cv.Mat], a dense row-major matrix
// of 8-bit unsigned samples. Single-channel (grayscale) and three-channel (RGB)
// images are the common cases. Three-channel data is treated as RGB, matching
// Go's image package and the root cv conventions, not OpenCV's native BGR.
// Masks — which select the region to fill — are represented by the package's
// own [Mask] type (a dense binary grid) that interoperates with [cv.Mat] via
// [MaskFromMat] and [Mask.ToMat]. Gradient fields use the [FloatImage] helper.
//
// # Dependencies
//
// inpaint imports only the root cv package and the Go standard library. It uses
// no cgo and no third-party code, so it builds and runs anywhere the Go
// toolchain does. It is deterministic: every routine, including the randomised
// PatchMatch search, uses a fixed internal seed so identical inputs yield
// identical outputs.
//
// # Algorithms
//
// Region filling:
//
//   - [InpaintTelea] implements Telea's (2004) Fast Marching Method inpainting:
//     unknown pixels are visited in increasing distance from the boundary (a
//     true eikonal solve, see [DistanceTransform] / [FastMarcher]) and each is
//     set to a gradient-corrected, direction/geometry/level weighted average of
//     its known neighbours, so smooth gradients are propagated inward.
//   - [InpaintNavierStokes] fills by harmonic initialisation followed by a
//     Bertalmio-style transport that propagates the image Laplacian along
//     isophotes (level lines), continuing edges into the hole.
//   - [InpaintDiffusion] solves the Laplace equation on the hole with the
//     surrounding pixels as Dirichlet boundary (a harmonic fill).
//   - [InpaintCriminisi] is Criminisi's (2004) exemplar-based completion:
//     priorities combine a confidence term and an isophote data term, and the
//     highest-priority boundary patch is filled by copying the best-matching
//     known patch. This propagates both structure and texture.
//   - [InpaintPatchMatch] completes the hole by iterated PatchMatch: a
//     randomised approximate nearest-neighbour field ([PatchMatchNNF] / [NNF])
//     maps hole patches to known patches, whose votes are averaged.
//   - [Inpaint] dispatches to any of the above by [Method].
//
// Restoration mask detection:
//
//   - [DetectScratches] finds thin bright or dark line defects (film scratches,
//     pen strokes) via a directional morphological top-hat.
//   - [DetectText] finds text/caption overlays via morphological gradient and
//     horizontal grouping.
//   - [DetectBlotches] finds small dust/blotch specks by size-limited extrema.
//   - [MaskFromColor] / [MaskFromColorRange] select pixels near a target colour
//     (e.g. a coloured logo or subtitle to erase).
//
// Seamless cloning and gradient-domain editing (Poisson image editing):
//
//   - [SeamlessClone] imports a masked source region into a destination by
//     solving the Poisson equation with the source (or mixed) gradients as
//     guidance and the destination pinned at the seam ([CloneMode]).
//   - [PoissonBlend] is the lower-level guided-gradient solver.
//   - [ColorChange], [IlluminationChange] and [TextureFlattening] reintegrate a
//     modified source-gradient field inside a masked region.
//   - [GradientX], [GradientY], [Laplacian], [Divergence] and [SolvePoisson]
//     expose the underlying gradient-domain primitives on [FloatImage].
//
// # Fidelity
//
// The implementations favour clarity and correctness over raw speed and are
// faithful in spirit rather than bit-exact with OpenCV. The iterative Poisson
// and Navier-Stokes solvers use Gauss-Seidel / explicit stepping, so they are
// intended for modest image sizes. All routines validate their arguments and
// panic (with a "inpaint:" prefix) on structural misuse such as size or channel
// mismatches.
package inpaint
