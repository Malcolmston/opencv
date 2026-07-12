// Package photo is a standard-library-only computational-photography toolkit
// built on top of the root cv package (github.com/malcolmston/opencv). It ports
// a useful subset of OpenCV's photo module: edge-aware denoising, region
// inpainting, edge-preserving smoothing, detail enhancement, stylization,
// contrast-preserving decolorization and Poisson seamless cloning.
//
// The package operates on the root package's [cv.Mat], a dense row-major matrix
// of 8-bit unsigned samples. Single-channel (grayscale) and three-channel (RGB)
// images are the common cases; where a function is channel-agnostic it says so.
// Three-channel data is treated as RGB, matching Go's image package and the
// root cv conventions, not OpenCV's native BGR.
//
// # Dependencies
//
// photo imports only the root cv package and the Go standard library (math,
// image). It uses no cgo and no third-party code, so it builds and runs
// anywhere the Go toolchain does. Small numeric helpers (edge-replicated
// sampling, rounding-and-clamping to [0,255], luma) are reimplemented locally
// rather than reaching into cv's unexported internals.
//
// # Algorithms and fidelity
//
// The implementations favour clarity and correctness over raw speed, and are
// deliberately faithful in spirit rather than bit-exact with OpenCV:
//
//   - [FastNlMeansDenoising] / [FastNlMeansDenoisingColored] implement the
//     non-local means estimator directly (a per-pixel weighted average of
//     pixels whose surrounding patches resemble the target patch). This is the
//     exact O(N·S²·T²) formulation, not OpenCV's integral-image approximation,
//     so keep the search and template windows small.
//   - [Inpaint] offers a distance-ordered boundary fill ([InpaintTelea], a
//     simplified fast-marching scheme) and a Laplace-diffusion fill
//     ([InpaintNS]). Both propagate surrounding colour into the masked region.
//   - [EdgePreservingFilter] is implemented on top of [cv.BilateralFilter];
//     [DetailEnhance] and [Stylization] build on it.
//   - [Decolor] chooses the non-negative channel mixture that maximises
//     grayscale variance, a simple contrast-preserving decolorization, and
//     returns a saturation-boosted colour image alongside it.
//   - [SeamlessClone] solves the Poisson equation with guidance gradients via
//     Gauss–Seidel iteration; use it on small images.
//
// # Deferred
//
// The following OpenCV photo features are intentionally out of scope: HDR tone
// mapping and exposure fusion (createTonemap*, MergeMertens, MergeDebevec),
// camera-response calibration, the primal-dual TVL1 denoiser
// (denoise_TVL1), multi-frame denoising overloads, OpenCV's integral-image NLM
// acceleration, and the exact Cewu-Lu decolorization optimisation.
package photo
