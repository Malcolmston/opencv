package cudafeatures2d

import (
	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/features2d"
)

// KeyPoint is a salient image point, aliased from the features2d package so
// results interoperate directly with it. It mirrors OpenCV's cv::KeyPoint.
type KeyPoint = features2d.KeyPoint

// DMatch records a query/train descriptor correspondence, aliased from the
// features2d package. It mirrors OpenCV's cv::DMatch.
type DMatch = features2d.DMatch

// NormType selects the distance used to compare descriptors, aliased from the
// features2d package.
type NormType = features2d.NormType

// Norm constants re-exported for convenience so callers need not import
// features2d directly.
const (
	// NormHamming counts differing bits between binary descriptors (ORB/BRIEF).
	NormHamming = features2d.NormHamming
	// NormL2 is the Euclidean distance between float descriptors.
	NormL2 = features2d.NormL2
)

// descriptorsToGpuMat packs bit-packed descriptor rows into a device GpuMat with
// one row per descriptor and one column per descriptor byte. It returns an empty
// GpuMat when there are no descriptors. It panics if the rows are ragged.
func descriptorsToGpuMat(rows [][]byte) *GpuMat {
	if len(rows) == 0 {
		return &GpuMat{}
	}
	width := len(rows[0])
	if width == 0 {
		return &GpuMat{}
	}
	m := cv.NewMat(len(rows), width, 1)
	for i, r := range rows {
		if len(r) != width {
			panic("cudafeatures2d: ragged descriptor rows")
		}
		copy(m.Data[i*width:(i+1)*width], r)
	}
	return &GpuMat{mat: m}
}

// descriptorsFromGpuMat unpacks a descriptor GpuMat (as produced by
// descriptorsToGpuMat) back into bit-packed rows. It returns nil for an empty
// GpuMat and panics if the matrix is not single-channel.
func descriptorsFromGpuMat(g *GpuMat) [][]byte {
	if g.Empty() {
		return nil
	}
	m := g.host()
	if m.Channels != 1 {
		panic("cudafeatures2d: descriptor GpuMat must be single-channel")
	}
	rows := make([][]byte, m.Rows)
	for i := 0; i < m.Rows; i++ {
		r := make([]byte, m.Cols)
		copy(r, m.Data[i*m.Cols:(i+1)*m.Cols])
		rows[i] = r
	}
	return rows
}

// DescriptorsToGpuMat packs host descriptor rows (such as those from
// [ORB.Convert]) into a device descriptor [GpuMat] suitable for
// [DescriptorMatcher.Match]. This is the inverse of [ORB.Convert].
func DescriptorsToGpuMat(rows [][]byte) *GpuMat {
	return descriptorsToGpuMat(rows)
}
