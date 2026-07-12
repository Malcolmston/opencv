package datamatrix

// This file wraps the GF(256) Reed-Solomon codec (galois.go) with the ECC200
// block-interleaving scheme (ISO/IEC 16022 §5.4). Symbols up to 48x48 use a
// single block and reduce to the plain rsEncode/rsCorrect path; larger symbols
// split the data codewords round-robin into several blocks, compute error
// codewords for each block independently and interleave those error codewords
// in the symbol so a localised smudge damages few codewords per block.

// interleavedCodewords returns the full codeword stream (data followed by the
// interleaved error-correction codewords) for data already padded to
// info.dataCW.
func interleavedCodewords(info symbolInfo, data []int) []int {
	bc := info.blockCount()
	full := make([]int, info.totalCW())
	copy(full, data)
	if bc == 1 {
		ecc := rsEncode(data, info.eccCW)[len(data):]
		copy(full[info.dataCW:], ecc)
		return full
	}
	for b := 0; b < bc; b++ {
		dl := info.blockDataLen(b)
		blockData := make([]int, 0, dl)
		for d := b; d < info.dataCW; d += bc {
			blockData = append(blockData, data[d])
		}
		el := info.blockErrLen(b)
		ecc := rsEncode(blockData, el)[len(blockData):]
		for e := 0; e < el; e++ {
			full[info.dataCW+b+e*bc] = ecc[e]
		}
	}
	return full
}

// recoverData reverses interleavedCodewords: it error-corrects each block of a
// freshly read (and possibly damaged) codeword stream and returns the corrected
// data codewords (length info.dataCW) plus the total number of codewords
// repaired.
func recoverData(info symbolInfo, full []int) ([]int, int, error) {
	bc := info.blockCount()
	if bc == 1 {
		corrected, n, err := rsCorrect(full, info.eccCW)
		if err != nil {
			return nil, 0, err
		}
		return corrected[:info.dataCW], n, nil
	}
	data := make([]int, info.dataCW)
	repaired := 0
	for b := 0; b < bc; b++ {
		dl := info.blockDataLen(b)
		el := info.blockErrLen(b)
		msg := make([]int, 0, dl+el)
		var dataIdx []int
		for d := b; d < info.dataCW; d += bc {
			msg = append(msg, full[d])
			dataIdx = append(dataIdx, d)
		}
		for e := 0; e < el; e++ {
			msg = append(msg, full[info.dataCW+b+e*bc])
		}
		corrected, n, err := rsCorrect(msg, el)
		if err != nil {
			return nil, 0, err
		}
		repaired += n
		for i, idx := range dataIdx {
			data[idx] = corrected[i]
		}
	}
	return data, repaired, nil
}
