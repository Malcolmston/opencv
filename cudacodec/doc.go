// Package cudacodec is a CPU-backed, API-compatible mirror of OpenCV's
// cudacodec module for the cv image toolkit
// ([github.com/malcolmston/opencv]). It reproduces the shape of OpenCV's
// hardware video-codec API — [VideoReader], [VideoWriter], the [Codec],
// [ColorFormat] and [SurfaceFormat] enumerations, [EncoderParams] and
// [FormatInfo] — but contains no cgo, no CUDA and no GPU code whatsoever. It
// builds and runs anywhere the Go toolchain does.
//
// # What "cudacodec" means here
//
// In upstream OpenCV, cudacodec wraps NVIDIA's NVDEC/NVENC hardware engines to
// decode and encode compressed video (H.264, HEVC, VP9, AV1, …) directly on the
// GPU, exchanging frames as device-resident cv::cuda::GpuMat objects. None of
// that is reachable from the pure-Go standard library: the hardware codecs
// require proprietary drivers and native SDKs behind cgo.
//
// This package is therefore an honest substitution, not an emulation. It keeps
// the familiar names and call shapes so code ported from OpenCV reads
// naturally, while delegating all real work to the sibling
// [github.com/malcolmston/opencv/videoio] package, which implements
// standard-library-only containers:
//
//   - Motion-JPEG AVI (image/jpeg inside a real RIFF/AVI container) stands in
//     for the intra-coded compressed codecs ([CodecJPEG], and — by
//     substitution — [CodecH264], [CodecHEVC], and friends).
//   - Animated PNG / APNG (image/png) stands in for lossless streams.
//   - Animated GIF (image/gif) stands in for palette-limited streams.
//   - Numbered image sequences round out the set.
//
// The concrete container is selected from the output file's extension
// (".avi", ".png"/".apng", ".gif"); the [Codec] value you request is recorded
// in [FormatInfo] and honoured as an intent, but the bytes on disk are always
// one of the four standard-library formats above. No NVDEC/NVENC bitstream is
// ever produced or consumed. See [CreateVideoWriter] and [CreateVideoReader].
//
// # GpuMat and Stream
//
// [GpuMat] is a thin wrapper around a host-resident [cv.Mat]; there is no device
// memory, so [GpuMat.Upload] and [GpuMat.Download] are ordinary copies and
// [Stream] is a no-op scheduling handle whose [Stream.WaitForCompletion] returns
// immediately. They exist so that method signatures match OpenCV's
// GpuMat/Stream-based API.
//
// # Typical use
//
// Encoding a synthetic clip and reading it back:
//
//	w, _ := cudacodec.CreateVideoWriter("clip.avi", image.Pt(64, 48),
//		cudacodec.CodecH264, 25, cudacodec.ColorFormatBGR)
//	for _, m := range frames {
//		_ = w.Write(cudacodec.NewGpuMatFromMat(m))
//	}
//	_ = w.Release()
//
//	r, _ := cudacodec.CreateVideoReader("clip.avi")
//	dst := cudacodec.NewGpuMat()
//	for r.NextFrame(dst) {
//		use(dst.Download())
//	}
//	_ = r.Release()
//
// Because the underlying codecs are deterministic, a written clip reads back
// with the same frame count and — for the lossless APNG/GIF-free paths — high
// fidelity; the MJPEG path is visually faithful but lossy, exactly as a real
// intra-coded codec would be.
package cudacodec
