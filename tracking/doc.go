// Package tracking implements classical, CPU-only computer-vision tracking and
// motion-estimation algorithms on top of the root cv package
// (github.com/malcolmston/opencv, imported as cv).
//
// The package is written entirely against the Go standard library and the root
// module. It has no third-party dependencies, no cgo and no GPU code, and every
// routine is deterministic: given the same inputs it always produces the same
// output. Images are represented by the library's central *cv.Mat type; the
// package converts multi-channel input to grayscale internally where an
// algorithm operates on intensity. A few small geometric helper types (Rect,
// Point2f, RotatedRect, TermCriteria) and a dense Matrix type for the linear
// filters are provided locally, mirroring the convention of the sibling
// cv/video and cv/optflow packages, which also define their own small value
// types rather than reaching across package boundaries.
//
// # Motion estimation
//
//   - [CalcOpticalFlowLK] / [CalcOpticalFlowPyrLK] — sparse Lucas-Kanade optical
//     flow, single level and pyramidal (coarse-to-fine) respectively.
//   - [CalcOpticalFlowHornSchunck] — dense variational Horn-Schunck flow.
//   - [CalcOpticalFlowFarneback] — dense flow via block matching over a Gaussian
//     pyramid, a compact stand-in for Farneback polynomial expansion.
//   - [FlowField] — a dense (u, v) displacement field with sampling and
//     statistics helpers.
//   - [BuildOpticalFlowPyramid] — the Gaussian pyramid shared by the sparse
//     solvers.
//
// # Region tracking
//
//   - [MeanShift] / [CamShift] — density-mode seeking on a back-projection image,
//     the second additionally estimating scale and orientation.
//   - [CalcBackProjection] and [Histogram1D] — the histogram back-projection
//     pipeline that feeds mean-shift.
//   - [KCFTracker] — a template correlation tracker (KCF-lite) using a
//     cosine-windowed normalized-cross-correlation response.
//   - [MatchTemplateNCC] — normalized cross-correlation template matching.
//
// # Recursive filters
//
//   - [KalmanFilter] — a linear Kalman filter with the standard predict/correct
//     cycle, plus [NewConstantVelocityKalman2D] for point tracking.
//   - [ParticleFilter] — a bootstrap (sequential importance resampling) particle
//     filter with deterministic, seedable sampling.
//   - [Matrix] — the small dense linear-algebra type the filters are built on.
//
// # Detection association
//
//   - [CentroidTracker] — nearest-centroid identity assignment across frames.
//   - [IoUTracker] — intersection-over-union based detection-to-track
//     association.
package tracking
