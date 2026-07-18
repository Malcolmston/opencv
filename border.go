package cv

// BorderType selects how out-of-bounds samples are synthesised by
// neighbourhood operations and [CopyMakeBorder], mirroring OpenCV's
// cv::BorderTypes.
type BorderType int

const (
	// BorderConstant pads with a fixed value: iiiiii|abcdefgh|iiiiiii.
	BorderConstant BorderType = iota
	// BorderReplicate repeats the edge sample: aaaaaa|abcdefgh|hhhhhhh.
	BorderReplicate
	// BorderReflect mirrors including the edge: fedcba|abcdefgh|hgfedcb.
	BorderReflect
	// BorderWrap tiles the image: cdefgh|abcdefgh|abcdefg.
	BorderWrap
	// BorderReflect101 mirrors excluding the edge: gfedcb|abcdefgh|gfedcba.
	BorderReflect101
)

// BorderInterpolate maps an out-of-range coordinate p in [ -inf, +inf ) to a
// valid index in [0, length) according to borderType, mirroring
// cv2.borderInterpolate. BorderConstant is reported as -1 (the caller supplies
// the constant). It panics if length is not positive.
func BorderInterpolate(p, length int, borderType BorderType) int {
	if length <= 0 {
		panic("cv: BorderInterpolate requires positive length")
	}
	if p >= 0 && p < length {
		return p
	}
	switch borderType {
	case BorderConstant:
		return -1
	case BorderReplicate:
		if p < 0 {
			return 0
		}
		return length - 1
	case BorderWrap:
		p %= length
		if p < 0 {
			p += length
		}
		return p
	case BorderReflect:
		return reflectIndex(p, length, true)
	case BorderReflect101:
		if length == 1 {
			return 0
		}
		return reflectIndex(p, length, false)
	default:
		panic("cv: BorderInterpolate unknown border type")
	}
}

// reflectIndex folds p into [0,length) by mirroring. When includeEdge is true
// the edge sample is duplicated (BorderReflect); otherwise it is not
// (BorderReflect101).
func reflectIndex(p, length int, includeEdge bool) int {
	if length == 1 {
		return 0
	}
	var period int
	if includeEdge {
		period = 2 * length
	} else {
		period = 2 * (length - 1)
	}
	p %= period
	if p < 0 {
		p += period
	}
	if includeEdge {
		if p >= length {
			p = period - 1 - p
		}
	} else {
		if p >= length {
			p = period - p
		}
	}
	return p
}

// CopyMakeBorder returns a copy of src enlarged by the given border widths on
// each side, filling the new pixels according to borderType. For
// BorderConstant the supplied value is used for each channel. This mirrors
// cv2.copyMakeBorder. It panics on negative border widths.
func CopyMakeBorder(src *Mat, top, bottom, left, right int, borderType BorderType, value Scalar) *Mat {
	if top < 0 || bottom < 0 || left < 0 || right < 0 {
		panic("cv: CopyMakeBorder requires non-negative border widths")
	}
	ch := src.Channels
	dst := NewMat(src.Rows+top+bottom, src.Cols+left+right, ch)
	for y := 0; y < dst.Rows; y++ {
		sy := BorderInterpolate(y-top, src.Rows, borderType)
		for x := 0; x < dst.Cols; x++ {
			sx := BorderInterpolate(x-left, src.Cols, borderType)
			di := dst.index(y, x)
			if sy < 0 || sx < 0 {
				for c := 0; c < ch; c++ {
					dst.Data[di+c] = clampToUint8(value[c] + 0.5)
				}
				continue
			}
			si := src.index(sy, sx)
			copy(dst.Data[di:di+ch], src.Data[si:si+ch])
		}
	}
	return dst
}
