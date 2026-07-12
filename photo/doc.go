// Package photo is a standard-library-only computational-photography toolkit
// built on top of the root cv package (github.com/malcolmston/opencv). It ports
// a useful subset of OpenCV's photo module: edge-aware denoising (single- and
// multi-frame, plus total-variation), region inpainting, edge-preserving
// smoothing, detail enhancement, stylization, pencil-sketch and cartoon
// rendering, oil painting, contrast-preserving decolorization, tone and
// white-balance correction, and Poisson image editing (seamless cloning,
// colour change, illumination change and texture flattening).
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
//     [DetailEnhance] and [Stylization] build on it. [DomainTransformFilter]
//     additionally provides the Gastal–Oliveira domain-transform edge-preserving
//     filter with both recursive (RF) and normalized-convolution (NC) variants,
//     which [PencilSketch] and [Cartoonify] use.
//   - [FastNlMeansDenoisingMulti] and [FastNlMeansDenoisingColoredMulti] extend
//     non-local means over a temporal window of frames. [DenoiseTVL1] solves the
//     total-variation L1 model with a Chambolle–Pock primal–dual iteration.
//   - [ColorChange], [IlluminationChange] and [TextureFlattening] are Poisson
//     edits: they reintegrate a modified source-gradient field inside a masked
//     region against the surrounding pixels, via Gauss–Seidel.
//   - [OilPainting] buckets each neighbourhood by intensity and returns its
//     dominant tone; [GammaCorrection], [UnsharpMask], [HistogramStretch],
//     [GrayWorldWhiteBalance] and [SimpleWhiteBalance] are per-pixel tone and
//     white-balance operators.
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
// camera-response calibration, OpenCV's integral-image NLM acceleration, and the
// exact Cewu-Lu decolorization optimisation.
package photo
