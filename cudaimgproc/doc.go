// Package cudaimgproc is a pure-Go, standard-library-only mirror of OpenCV's
// cudaimgproc module. It presents the same API shape as the CUDA image
// processing routines — a [GpuMat] device-matrix type, no-op [Stream] objects
// and the familiar create*/Detect/Compute/Apply/Match algorithm objects — but
// every operation runs on the CPU.
//
// # Honest scope
//
// There is NO GPU here. Despite the "cuda" name and the GpuMat/Stream
// vocabulary, this package contains no cgo, no CUDA kernels and no device
// memory. A [GpuMat] is a thin wrapper around a host-resident [cv.Mat]; every
// function computes its result on the CPU by delegating to the root package
// github.com/malcolmston/opencv (imported here as cv). The device-flavoured
// surface exists so that code written against OpenCV's cuda modules — which
// upload a Mat, run a sequence of GpuMat operations, then download the result —
// can be ported to pure Go with minimal edits. Because the maths is identical
// to the root package, results are deterministic and byte-for-byte comparable
// with the equivalent cv call; the [Stream] type is a placeholder that
// schedules nothing.
//
// If you need real GPU acceleration, use OpenCV's native cudaimgproc through
// cgo bindings instead. This package trades throughput for portability: it
// builds and runs anywhere Go does, with no drivers or toolkits.
//
// # Design
//
// The package imports only the Go standard library and the root cv package; it
// pulls in none of the sibling cv/* subpackages. Data crosses the host/"device"
// boundary through [GpuMat.Upload] and [GpuMat.Download], mirroring the OpenCV
// workflow:
//
//	var d cudaimgproc.GpuMat
//	d.Upload(img)                       // host -> "device"
//	gray := cudaimgproc.CvtColor(d, cv.ColorRGB2Gray)
//	out := gray.Download()              // "device" -> host
//
// Every algorithm is offered both as a free function (matching the
// cuda::<name> free functions) and, where OpenCV exposes an algorithm object,
// as a create*/method pair (for example [CreateCLAHE] returning a [CLAHE] with
// an Apply method). All functions accept a trailing, ignored [Stream] argument
// so call sites that pass a stream compile unchanged.
//
// # Contents
//
//   - Colour: [CvtColor], [CvtColorBayer], [DemosaicBayer], [SwapChannels],
//     [GammaCorrection], [AlphaComp].
//   - Histogram: [CalcHist], [EqualizeHist], [CreateCLAHE], [HistEven],
//     [HistRange], [CalcBackProject].
//   - Edges and shapes: [CreateCannyEdgeDetector], [CreateHoughLinesDetector],
//     [CreateHoughSegmentDetector], [CreateHoughCirclesDetector].
//   - Corners: [CreateHarrisCorner], [CreateMinEigenValCorner].
//   - Matching: [CreateTemplateMatching].
//   - Mean shift: [MeanShiftFiltering], [MeanShiftProc],
//     [MeanShiftSegmentation].
//   - Filtering and blending: [BilateralFilter], [Blend], [BlendLinear].
package cudaimgproc
