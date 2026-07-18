package histogram2

import cv "github.com/malcolmston/opencv"

// CalcBackProject1D projects a one-dimensional histogram back onto an image:
// each output pixel receives the histogram weight of the bin its channel value
// falls into, rescaled so the largest bin maps to 255. The result is a
// single-channel probability map highlighting regions whose intensity matches
// the model histogram — the core of histogram-based tracking (CamShift). It
// returns [ErrEmptyImage] if src is empty and [ErrChannelRange] if channel is
// out of range.
func CalcBackProject1D(src *cv.Mat, channel int, hist *Histogram1D) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if channel < 0 || channel >= src.Channels {
		return nil, ErrChannelRange
	}
	maxBin := 0.0
	for _, v := range hist.Counts {
		if v > maxBin {
			maxBin = v
		}
	}
	// Map each histogram bin to a byte weight.
	weight := make([]uint8, hist.BinCount)
	if maxBin > 0 {
		for i, v := range hist.Counts {
			weight[i] = histogram2clampByte(v / maxBin * 255)
		}
	}
	dst := cv.NewMat(src.Rows, src.Cols, 1)
	total := src.Total()
	for p := 0; p < total; p++ {
		v := float64(src.Data[p*src.Channels+channel])
		dst.Data[p] = weight[hist.BinIndex(v)]
	}
	return dst, nil
}

// CalcBackProject2D projects a two-dimensional histogram of channels chX and
// chY back onto an image, rescaling so the largest bin maps to 255. The result
// is a single-channel probability map. It returns [ErrEmptyImage] if src is
// empty and [ErrChannelRange] if either channel is out of range.
func CalcBackProject2D(src *cv.Mat, chX, chY int, hist *Histogram2D) (*cv.Mat, error) {
	if src.Empty() {
		return nil, ErrEmptyImage
	}
	if chX < 0 || chX >= src.Channels || chY < 0 || chY >= src.Channels {
		return nil, ErrChannelRange
	}
	maxBin := 0.0
	for _, v := range hist.Counts {
		if v > maxBin {
			maxBin = v
		}
	}
	weight := make([]uint8, len(hist.Counts))
	if maxBin > 0 {
		for i, v := range hist.Counts {
			weight[i] = histogram2clampByte(v / maxBin * 255)
		}
	}
	dst := cv.NewMat(src.Rows, src.Cols, 1)
	total := src.Total()
	for p := 0; p < total; p++ {
		bx := int(src.Data[p*src.Channels+chX]) * hist.BinsX / 256
		by := int(src.Data[p*src.Channels+chY]) * hist.BinsY / 256
		dst.Data[p] = weight[by*hist.BinsX+bx]
	}
	return dst, nil
}
