package cudafeatures2d

import (
	"github.com/malcolmston/opencv/features2d"
)

// ORB is a CPU-backed mirror of cv::cuda::ORB. It detects oriented FAST
// keypoints and computes rotated BRIEF descriptors by delegating to
// features2d.ORB. Descriptors are returned as a device [GpuMat] whose rows are
// the bit-packed descriptors, matching the GPU class's device-side output.
//
// Construct one with [CreateORB].
type ORB struct {
	impl *features2d.ORB
}

// CreateORB returns an ORB detector retaining up to nFeatures keypoints
// (nFeatures <= 0 keeps all detected corners). It mirrors cv::cuda::ORB::create;
// GPU-specific parameters such as scaleFactor and nlevels have no effect in this
// single-scale CPU port and are omitted.
func CreateORB(nFeatures int) *ORB {
	return &ORB{impl: features2d.NewORB(nFeatures)}
}

// applyMask removes keypoints whose location falls on a zero pixel of mask. A
// nil or empty mask keeps every keypoint. The mask is taken single-channel (its
// first channel is consulted).
func applyMask(kps []KeyPoint, mask *GpuMat) []KeyPoint {
	if mask.Empty() {
		return kps
	}
	m := mask.host()
	out := kps[:0:0]
	for _, kp := range kps {
		x, y := kp.Pt.X, kp.Pt.Y
		if x < 0 || y < 0 || x >= m.Cols || y >= m.Rows {
			continue
		}
		if m.Data[(y*m.Cols+x)*m.Channels] == 0 {
			continue
		}
		out = append(out, kp)
	}
	return out
}

// Detect finds oriented FAST keypoints in the device image, optionally filtered
// by mask (pass a nil or empty GpuMat for no mask). It mirrors
// cv::cuda::ORB::detect. It panics if image is empty.
func (o *ORB) Detect(image, mask *GpuMat) []KeyPoint {
	if image.Empty() {
		panic("cudafeatures2d: ORB.Detect on empty image")
	}
	kps := o.impl.Detect(image.host())
	return applyMask(kps, mask)
}

// Compute computes rotated BRIEF descriptors for the supplied keypoints on the
// device image, returning the (possibly reduced) keypoints and a device
// descriptor [GpuMat]. It mirrors cv::cuda::ORB::compute. Keypoints are described
// with the same steered BRIEF extractor features2d.ORB uses. It panics if image
// is empty.
func (o *ORB) Compute(image *GpuMat, kps []KeyPoint) ([]KeyPoint, *GpuMat) {
	if image.Empty() {
		panic("cudafeatures2d: ORB.Compute on empty image")
	}
	if len(kps) == 0 {
		return nil, &GpuMat{}
	}
	outKps, desc := (&features2d.BRIEF{}).Compute(image.host(), kps)
	return outKps, descriptorsToGpuMat(desc)
}

// DetectAndCompute detects oriented FAST keypoints in the device image and
// computes their rotated BRIEF descriptors in one call, returning the keypoints
// and a device descriptor [GpuMat] whose i-th row describes the i-th keypoint.
// Pass a nil or empty mask for no masking. It mirrors
// cv::cuda::ORB::detectAndCompute. It panics if image is empty.
func (o *ORB) DetectAndCompute(image, mask *GpuMat) ([]KeyPoint, *GpuMat) {
	kps := o.Detect(image, mask)
	if len(kps) == 0 {
		return nil, &GpuMat{}
	}
	return o.Compute(image, kps)
}

// DetectAndComputeAsync is the streamed form of [ORB.DetectAndCompute]. Because
// this port is synchronous the stream is inert and the result is ready on
// return; it exists for source compatibility with cv::cuda::ORB. A nil stream is
// accepted.
func (o *ORB) DetectAndComputeAsync(image, mask *GpuMat, _ *Stream) ([]KeyPoint, *GpuMat) {
	return o.DetectAndCompute(image, mask)
}

// Convert unpacks a device descriptor [GpuMat] (as returned by
// [ORB.DetectAndCompute] or [ORB.Compute]) into host-side bit-packed descriptor
// rows. It is the CPU analogue of downloading and de-serialising the device
// descriptors, and the inverse of [DescriptorsToGpuMat].
func (o *ORB) Convert(descriptors *GpuMat) [][]byte {
	return descriptorsFromGpuMat(descriptors)
}

// DescriptorSize returns the descriptor length in bytes (32 for the default
// 256-bit rotated BRIEF descriptor). It mirrors
// cv::cuda::ORB::descriptorSize.
func (o *ORB) DescriptorSize() int {
	return 32
}

// DefaultNorm returns the norm type ORB descriptors should be matched with
// ([NormHamming]). It mirrors cv::cuda::ORB::defaultNorm.
func (o *ORB) DefaultNorm() NormType {
	return NormHamming
}
