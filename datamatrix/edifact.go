package datamatrix

// This file implements the EDIFACT encodation scheme of ECC200. EDIFACT packs
// four 6-bit values into three codewords; each value is the low six bits of an
// ASCII character in the range 0x20..0x5E. The scheme is entered with the
// latch codeword 240 and left with the 6-bit unlatch value 0b011111 (31).
//
// This codec only latches into EDIFACT for runs whose length is a multiple of
// four native characters, so the data is always byte-aligned; the unlatch is
// then written as a single trailing codeword whose low two bits are padding.
// The decoder (lowdecode.go) reads six bits at a time and returns to ASCII the
// moment it reads the unlatch value, discarding the remaining bits of that
// codeword.

const edifactUnlatch = 0x1F // 011111

// isEDIFACTNative reports whether b can be encoded directly in EDIFACT.
func isEDIFACTNative(b byte) bool { return b >= 0x20 && b <= 0x5E }

// edifactValue returns the 6-bit EDIFACT value for a native character.
func edifactValue(b byte) int { return int(b) & 0x3F }

// edifactByteToASCII converts a decoded 6-bit EDIFACT value back to its ASCII
// byte.
func edifactByteToASCII(v int) byte {
	if v < 0x20 {
		return byte(v | 0x40)
	}
	return byte(v)
}

// packEDIFACT encodes a run of native characters whose length is a multiple of
// four into codewords, then appends the single unlatch codeword that returns
// the decoder to ASCII. Every four characters become exactly three codewords,
// so the output before the unlatch is byte-aligned.
func packEDIFACT(chars []byte) []int {
	out := make([]int, 0, len(chars)/4*3+1)
	var acc, bits int
	push := func(v, n int) {
		acc = acc<<n | v
		bits += n
		for bits >= 8 {
			bits -= 8
			out = append(out, (acc>>bits)&0xFF)
		}
	}
	for _, c := range chars {
		push(edifactValue(c), 6)
	}
	// chars is a multiple of four, so bits == 0 here and the stream is
	// byte-aligned. The unlatch occupies its own codeword with two padding bits.
	out = append(out, edifactUnlatch<<2)
	return out
}
