package pyramids

import (
	cv "github.com/malcolmston/opencv"
)

// AlphaBlendFloat linearly blends two equally sized grids under a per-pixel
// weight: out = a*mask + b*(1-mask). mask is normally in [0,1] but any value is
// permitted. All three grids must share dimensions.
func AlphaBlendFloat(a, b, mask *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(a, "AlphaBlendFloat")
	pyramidsRequire(b, "AlphaBlendFloat")
	pyramidsRequire(mask, "AlphaBlendFloat")
	pyramidsSameSize(a, b, "AlphaBlendFloat")
	pyramidsSameSize(a, mask, "AlphaBlendFloat")
	out := cv.NewFloatMat(a.Rows, a.Cols)
	for i := range out.Data {
		m := mask.Data[i]
		out.Data[i] = a.Data[i]*m + b.Data[i]*(1-m)
	}
	return out
}

// BlendLaplacian performs seamless multi-resolution blending of two
// single-channel images a and b under a soft mask, following Burt and
// Adelson's algorithm. It builds a Laplacian pyramid of each image and a
// Gaussian pyramid of the mask, blends each band and the base with the mask
// level of matching size, and reconstructs the result. mask is expected in the
// 0..1 range (1 selects a, 0 selects b). All three inputs must share
// dimensions. It panics if levels is not positive.
func BlendLaplacian(a, b, mask *cv.FloatMat, levels int) *cv.FloatMat {
	pyramidsRequire(a, "BlendLaplacian")
	pyramidsRequire(b, "BlendLaplacian")
	pyramidsRequire(mask, "BlendLaplacian")
	pyramidsSameSize(a, b, "BlendLaplacian")
	pyramidsSameSize(a, mask, "BlendLaplacian")

	la := BuildLaplacianPyramid(a, levels)
	lb := BuildLaplacianPyramid(b, levels)
	gm := BuildGaussianPyramid(mask, levels)

	blended := &LaplacianPyramid{Bands: make([]*cv.FloatMat, len(la.Bands))}
	for i := range la.Bands {
		blended.Bands[i] = AlphaBlendFloat(la.Bands[i], lb.Bands[i], gm.Levels[i])
	}
	baseIdx := len(gm.Levels) - 1
	blended.Base = AlphaBlendFloat(la.Base, lb.Base, gm.Levels[baseIdx])
	return blended.Reconstruct()
}

// BlendLaplacianMat performs multi-resolution blending of two 8-bit images
// channel by channel and returns an 8-bit result. a and b must have identical
// dimensions and channel counts. mask is a single-channel 8-bit image where
// 255 fully selects a and 0 fully selects b (values in between blend); it is
// scaled to 0..1 internally. It panics if levels is not positive or the shapes
// disagree.
func BlendLaplacianMat(a, b, mask *cv.Mat, levels int) *cv.Mat {
	if a == nil || b == nil || mask == nil || a.Empty() || b.Empty() || mask.Empty() {
		panic("pyramids: BlendLaplacianMat: empty input")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("pyramids: BlendLaplacianMat: a and b must match in shape")
	}
	if mask.Rows != a.Rows || mask.Cols != a.Cols || mask.Channels != 1 {
		panic("pyramids: BlendLaplacianMat: mask must be single-channel and match size")
	}
	m := ChannelFloat(mask, 0)
	for i := range m.Data {
		m.Data[i] /= 255.0
	}
	planes := make([]*cv.FloatMat, a.Channels)
	for c := 0; c < a.Channels; c++ {
		planes[c] = BlendLaplacian(ChannelFloat(a, c), ChannelFloat(b, c), m, levels)
	}
	return MergeFloat(planes)
}

// MultiBandBlendMat blends an arbitrary number of 8-bit source images into a
// single 8-bit mosaic using per-source soft masks and Laplacian-pyramid
// (multi-band) compositing. All images and masks must share dimensions; images
// must share channel count; masks are single-channel 8-bit weights. At each
// pixel the masks are normalised to sum to one across sources (pixels where all
// masks are zero fall back to a uniform average), so the routine is robust to
// unnormalised or overlapping masks. It panics on empty input, mismatched
// shapes, or a source/mask count mismatch.
func MultiBandBlendMat(images, masks []*cv.Mat, levels int) *cv.Mat {
	if len(images) == 0 || len(images) != len(masks) {
		panic("pyramids: MultiBandBlendMat: need equal, non-empty images and masks")
	}
	rows, cols, ch := images[0].Rows, images[0].Cols, images[0].Channels
	for i := range images {
		if images[i] == nil || images[i].Empty() || images[i].Rows != rows || images[i].Cols != cols || images[i].Channels != ch {
			panic("pyramids: MultiBandBlendMat: image shape mismatch")
		}
		if masks[i] == nil || masks[i].Empty() || masks[i].Rows != rows || masks[i].Cols != cols || masks[i].Channels != 1 {
			panic("pyramids: MultiBandBlendMat: mask shape mismatch")
		}
	}

	// Normalise the masks so they sum to one at every pixel.
	normMasks := make([]*cv.FloatMat, len(masks))
	for i := range masks {
		normMasks[i] = ChannelFloat(masks[i], 0)
	}
	n := len(images)
	for p := 0; p < rows*cols; p++ {
		var sum float64
		for i := 0; i < n; i++ {
			sum += normMasks[i].Data[p]
		}
		if sum <= 0 {
			for i := 0; i < n; i++ {
				normMasks[i].Data[p] = 1.0 / float64(n)
			}
		} else {
			for i := 0; i < n; i++ {
				normMasks[i].Data[p] /= sum
			}
		}
	}

	// Build the blended Laplacian pyramid: weighted sum of each source's bands
	// with the source's Gaussian-pyramid mask, then reconstruct per channel.
	planes := make([]*cv.FloatMat, ch)
	for c := 0; c < ch; c++ {
		var acc *LaplacianPyramid
		gms := make([]*GaussianPyramid, n)
		lps := make([]*LaplacianPyramid, n)
		for i := 0; i < n; i++ {
			lps[i] = BuildLaplacianPyramid(ChannelFloat(images[i], c), levels)
			gms[i] = BuildGaussianPyramid(normMasks[i], levels)
		}
		nb := len(lps[0].Bands)
		acc = &LaplacianPyramid{Bands: make([]*cv.FloatMat, nb)}
		for bi := 0; bi < nb; bi++ {
			band := cv.NewFloatMat(lps[0].Bands[bi].Rows, lps[0].Bands[bi].Cols)
			for i := 0; i < n; i++ {
				w := gms[i].Levels[bi]
				src := lps[i].Bands[bi]
				for k := range band.Data {
					band.Data[k] += src.Data[k] * w.Data[k]
				}
			}
			acc.Bands[bi] = band
		}
		baseIdx := len(gms[0].Levels) - 1
		base := cv.NewFloatMat(lps[0].Base.Rows, lps[0].Base.Cols)
		for i := 0; i < n; i++ {
			w := gms[i].Levels[baseIdx]
			src := lps[i].Base
			for k := range base.Data {
				base.Data[k] += src.Data[k] * w.Data[k]
			}
		}
		acc.Base = base
		planes[c] = acc.Reconstruct()
	}
	return MergeFloat(planes)
}
