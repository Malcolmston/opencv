// Package videoproc implements temporal (video) computer-vision primitives on
// top of the root cv package (github.com/malcolmston/opencv), mirroring a
// useful subset of OpenCV's video, bgsegm and optflow modules.
//
// Every routine operates on the library's existing image type, cv.Mat, and
// produces cv.Mat masks or cv.FloatMat accumulators; no competing image type
// is introduced. The only small helper type added is [PointF], a floating-point
// image coordinate used by the tracking and trajectory routines, and the
// container types [FlowField] and [Trajectory].
//
// The package is written entirely against the Go standard library (only the
// math and sort packages) and the root cv package. It has no third-party
// dependencies, no cgo, and is deterministic: given the same frames it always
// returns the same result.
//
// # What it provides
//
//   - Frame differencing: [AbsDiff], [FrameDifference], [ThreeFrameDifference],
//     accumulation ([Accumulate], [AccumulateWeighted], [AccumulateSquare]) and
//     motion metrics ([MotionEnergy], [CountMotionPixels]).
//   - Background subtraction: [RunningAverageSubtractor],
//     [MedianBackgroundSubtractor] and [MOGBackgroundSubtractor], all satisfying
//     the [BackgroundSubtractor] interface.
//   - Motion history images (Bobick & Davis): [MotionHistory],
//     [UpdateMotionHistory], [MotionGradient] and [GlobalMotionOrientation].
//   - Temporal filtering across a frame stack: [TemporalAverage],
//     [TemporalMedian], [TemporalMinimum], [TemporalMaximum], [TemporalGaussian],
//     plus the online filters [ExponentialMovingAverage] and [MovingWindowFilter].
//   - Shot-boundary (cut) detection: [HistogramL1Difference],
//     [HistogramChiSquare], [PixelDifferenceRatio], [EdgeChangeRatio],
//     [MeanIntensityDelta], the online [ShotBoundaryDetector] and the batch
//     [DetectShotBoundaries].
//   - Frame interpolation: [BlendFrames], [CrossFade], [WarpByFlow] and
//     motion-compensated [InterpolateFlow] over a [FlowField].
//   - Feature-based stabilization: [EstimateGlobalTranslation], [WarpTranslate],
//     [SmoothTrajectory] and the online [Stabilizer].
//   - Dense trajectory sampling: [SampleDenseGrid], [TrackPoints], [Trajectory]
//     and [DenseTrajectorySampler].
//
// Unless stated otherwise, a frame may be single- or multi-channel; multi-channel
// frames are converted to grayscale intensity (Rec. 601 luma) before analysis,
// exactly as the root cv package does.
package videoproc
