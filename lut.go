package cv

// LUT applies a 256-entry lookup table to every sample of src and returns the
// result, mirroring cv2.LUT. The same table is applied to all channels; it
// panics if table does not have exactly 256 entries.
func LUT(src *Mat, table []uint8) *Mat {
	if len(table) != 256 {
		panic("cv: LUT requires a 256-entry table")
	}
	dst := NewMat(src.Rows, src.Cols, src.Channels)
	for i, v := range src.Data {
		dst.Data[i] = table[v]
	}
	return dst
}

// LUTChannels applies a separate 256-entry table to each channel of src and
// returns the result. It panics unless len(tables) equals src.Channels and each
// table has 256 entries.
func LUTChannels(src *Mat, tables [][]uint8) *Mat {
	if len(tables) != src.Channels {
		panic("cv: LUTChannels requires one table per channel")
	}
	for _, t := range tables {
		if len(t) != 256 {
			panic("cv: LUTChannels requires 256-entry tables")
		}
	}
	dst := NewMat(src.Rows, src.Cols, src.Channels)
	ch := src.Channels
	for i := 0; i < len(src.Data); i++ {
		dst.Data[i] = tables[i%ch][src.Data[i]]
	}
	return dst
}
