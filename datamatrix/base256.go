package datamatrix

// This file implements the Base 256 encodation scheme of ECC200, which stores
// arbitrary 8-bit bytes. The scheme is entered with the latch codeword 231,
// followed by a one- or two-codeword length indicator and then the raw bytes.
// Every codeword after the latch (the length indicator and the data bytes) is
// obfuscated with the 255-state randomising algorithm from ISO/IEC 16022 so
// that long runs of identical bytes do not create large blank areas; the
// randomisation depends on the codeword's absolute 1-based position within the
// symbol's data codewords, so Base 256 encoding must know where it sits in the
// stream.

const base256Latch = 231

// randomize255 applies the 255-state randomisation to a Base 256 codeword at
// the given 1-based position.
func randomize255(value, position int) int {
	pseudo := ((149 * position) % 255) + 1
	t := value + pseudo
	if t <= 255 {
		return t
	}
	return t - 256
}

// unrandomize255 reverses randomize255.
func unrandomize255(value, position int) int {
	pseudo := ((149 * position) % 255) + 1
	t := value - pseudo
	if t >= 0 {
		return t
	}
	return t + 256
}

// packBase256 encodes data as a Base 256 segment (latch, randomised length
// indicator and randomised bytes). startPos is the 1-based position that the
// latch codeword 231 occupies in the data-codeword stream; the length indicator
// and payload follow it.
func packBase256(data []byte, startPos int) []int {
	out := []int{base256Latch}
	pos := startPos + 1 // first randomised codeword sits after the latch
	writeRand := func(v int) {
		out = append(out, randomize255(v, pos))
		pos++
	}
	n := len(data)
	if n <= 249 {
		writeRand(n)
	} else {
		writeRand(n/250 + 249)
		writeRand(n % 250)
	}
	for _, b := range data {
		writeRand(int(b))
	}
	return out
}
