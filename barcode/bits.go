package barcode

// bitBuffer accumulates a most-significant-bit-first stream of bits, used to
// assemble a QR data segment before it is packed into codeword bytes.
type bitBuffer struct {
	bits []bool
}

// appendBits appends the low n bits of value, most significant first.
func (b *bitBuffer) appendBits(value, n int) {
	for i := n - 1; i >= 0; i-- {
		b.bits = append(b.bits, (value>>i)&1 != 0)
	}
}

// len returns the number of bits accumulated so far.
func (b *bitBuffer) len() int {
	return len(b.bits)
}

// bytes packs the buffered bits into bytes, most significant bit first. The
// caller is responsible for having padded to a byte boundary.
func (b *bitBuffer) bytes() []byte {
	out := make([]byte, (len(b.bits)+7)/8)
	for i, bit := range b.bits {
		if bit {
			out[i>>3] |= 1 << (7 - (i & 7))
		}
	}
	return out
}

// bitReader reads a most-significant-bit-first stream from a byte slice.
type bitReader struct {
	data []byte
	pos  int
}

// read consumes the next n bits and returns them as an integer, reporting false
// if fewer than n bits remain.
func (r *bitReader) read(n int) (int, bool) {
	if r.pos+n > len(r.data)*8 {
		return 0, false
	}
	v := 0
	for i := 0; i < n; i++ {
		byteIdx := r.pos >> 3
		bitIdx := 7 - (r.pos & 7)
		v = v<<1 | int((r.data[byteIdx]>>bitIdx)&1)
		r.pos++
	}
	return v, true
}
