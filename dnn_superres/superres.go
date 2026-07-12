package dnn_superres

import (
	"fmt"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// UpsamplerType enumerates the super-resolution algorithms this package
// provides. All are classical, weight-free methods; see the package overview for
// which trained OpenCV models are deferred.
type UpsamplerType int

const (
	// Nearest is nearest-neighbour sampling (blocky, exact-preserving).
	Nearest UpsamplerType = iota
	// Bilinear is separable 2-tap linear interpolation (smooth, soft).
	Bilinear
	// Bicubic is Keys/Catmull-Rom bicubic convolution, a = -0.5 (INTER_CUBIC).
	Bicubic
	// Lanczos is Lanczos-4 windowed-sinc interpolation (INTER_LANCZOS4).
	Lanczos
	// EdgeDirected is a NEDI-lite / edge-guided cubic that shaves jaggies along
	// strong edges.
	EdgeDirected
	// FSRCNN is an FSRCNN-STYLE fixed-kernel sharpening finish on a bicubic base
	// (NOT a trained network).
	FSRCNN
)

// String returns the canonical lowercase algorithm name for the type, the same
// spelling accepted by [DnnSuperResImpl.SetModel].
func (t UpsamplerType) String() string {
	switch t {
	case Nearest:
		return "nearest"
	case Bilinear:
		return "bilinear"
	case Bicubic:
		return "bicubic"
	case Lanczos:
		return "lanczos"
	case EdgeDirected:
		return "edge"
	case FSRCNN:
		return "fsrcnn"
	default:
		return fmt.Sprintf("UpsamplerType(%d)", int(t))
	}
}

// ParseUpsamplerType maps an algorithm name to its [UpsamplerType]. Matching is
// case-insensitive and tolerant of surrounding whitespace. Accepted names are
// "nearest", "bilinear", "bicubic", "lanczos", "edge" (aliases "nedi",
// "edge-directed") and "fsrcnn". It returns an error for any other name.
func ParseUpsamplerType(algo string) (UpsamplerType, error) {
	switch strings.ToLower(strings.TrimSpace(algo)) {
	case "nearest":
		return Nearest, nil
	case "bilinear", "linear":
		return Bilinear, nil
	case "bicubic", "cubic":
		return Bicubic, nil
	case "lanczos", "lanczos4":
		return Lanczos, nil
	case "edge", "nedi", "edge-directed", "edgedirected":
		return EdgeDirected, nil
	case "fsrcnn", "fsrcnn-style":
		return FSRCNN, nil
	default:
		return 0, fmt.Errorf("dnn_superres: unknown algorithm %q", algo)
	}
}

// Upsample runs the algorithm named by t on src at the given scale (2, 3 or 4),
// returning a new Mat of size (Rows*scale, Cols*scale). It is the free-function
// dispatch behind [DnnSuperResImpl.Upsample]. It returns an error for an empty
// image, an unsupported scale, or an unknown type.
func (t UpsamplerType) Upsample(src *cv.Mat, scale int) (*cv.Mat, error) {
	switch t {
	case Nearest:
		return UpsampleNearest(src, scale)
	case Bilinear:
		return UpsampleBilinear(src, scale)
	case Bicubic:
		return UpsampleBicubic(src, scale)
	case Lanczos:
		return UpsampleLanczos(src, scale)
	case EdgeDirected:
		return UpsampleEdgeDirected(src, scale)
	case FSRCNN:
		return UpsampleFSRCNN(src, scale)
	default:
		return nil, fmt.Errorf("dnn_superres: unknown UpsamplerType %d", int(t))
	}
}

// DnnSuperResImpl is a stateful single-image super-resolution engine mirroring
// OpenCV's cv::dnn_superres::DnnSuperResImpl. It is configured once with
// [DnnSuperResImpl.SetModel] and then applied repeatedly with
// [DnnSuperResImpl.Upsample]. Unlike the OpenCV class it loads no .pb weight
// file and runs no neural network — every algorithm is a classical resampler
// (see the package overview). The zero value is not ready to use; call
// [NewDnnSuperResImpl] and then SetModel.
type DnnSuperResImpl struct {
	algo  UpsamplerType
	scale int
	set   bool
}

// NewDnnSuperResImpl returns a new, unconfigured super-resolution engine. Call
// [DnnSuperResImpl.SetModel] before [DnnSuperResImpl.Upsample].
func NewDnnSuperResImpl() *DnnSuperResImpl {
	return &DnnSuperResImpl{}
}

// SetModel selects the algorithm and integer scale factor used by subsequent
// calls to [DnnSuperResImpl.Upsample], mirroring the OpenCV method of the same
// name. algo is one of the names accepted by [ParseUpsamplerType]; scale must be
// 2, 3 or 4. It returns an error (and leaves any previous configuration intact)
// for an unknown algorithm or unsupported scale.
func (s *DnnSuperResImpl) SetModel(algo string, scale int) error {
	t, err := ParseUpsamplerType(algo)
	if err != nil {
		return err
	}
	if scale != 2 && scale != 3 && scale != 4 {
		return fmt.Errorf("dnn_superres: unsupported scale %d (want 2, 3 or 4)", scale)
	}
	s.algo = t
	s.scale = scale
	s.set = true
	return nil
}

// Upsample enlarges img using the algorithm and scale set by
// [DnnSuperResImpl.SetModel], returning a new Mat of size (Rows*scale,
// Cols*scale). Grayscale and RGB images are both supported. It returns an error
// if SetModel has not been called, or if img is empty.
func (s *DnnSuperResImpl) Upsample(img *cv.Mat) (*cv.Mat, error) {
	if !s.set {
		return nil, fmt.Errorf("dnn_superres: SetModel must be called before Upsample")
	}
	return s.algo.Upsample(img, s.scale)
}

// GetAlgorithm returns the canonical name of the currently configured algorithm,
// or the empty string if [DnnSuperResImpl.SetModel] has not been called. It
// mirrors OpenCV's getAlgorithm.
func (s *DnnSuperResImpl) GetAlgorithm() string {
	if !s.set {
		return ""
	}
	return s.algo.String()
}

// GetScale returns the currently configured scale factor, or 0 if
// [DnnSuperResImpl.SetModel] has not been called. It mirrors OpenCV's getScale.
func (s *DnnSuperResImpl) GetScale() int {
	if !s.set {
		return 0
	}
	return s.scale
}
