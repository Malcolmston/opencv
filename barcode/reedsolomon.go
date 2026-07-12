package barcode

// This file implements a small Reed-Solomon error-correcting codec over the
// Galois field GF(256). It is the mathematical core shared by the QR encoder
// and decoder in this package: QR symbols append RS "error correction"
// codewords so that a reader can recover the message even when some modules are
// misread. The implementation is self-contained (standard library only) and
// deterministic.
//
// The field uses the QR/Data-Matrix primitive polynomial x^8 + x^4 + x^3 + x^2
// + 1 (0x11D) with generator element alpha = 2, matching the ISO/IEC 18004 QR
// specification. Codewords are bytes; polynomials are byte slices in
// big-endian order (index 0 is the highest-degree coefficient), the convention
// used by the classic "Reed-Solomon codes for coders" reference.

// gfExp and gfLog are the exponent and logarithm tables of GF(256). gfExp is
// doubled in length so that gfExp[a+b] never needs a modular reduction for
// exponents in [0,510].
var (
	gfExp [512]byte
	gfLog [256]byte
)

func init() {
	x := 1
	for i := 0; i < 255; i++ {
		gfExp[i] = byte(x)
		gfLog[x] = byte(i)
		x <<= 1
		if x&0x100 != 0 {
			x ^= 0x11D
		}
	}
	for i := 255; i < 512; i++ {
		gfExp[i] = gfExp[i-255]
	}
}

// gfMul multiplies two field elements.
func gfMul(a, b byte) byte {
	if a == 0 || b == 0 {
		return 0
	}
	return gfExp[int(gfLog[a])+int(gfLog[b])]
}

// gfDiv divides a by b in the field. It panics if b is zero, mirroring a
// division by zero.
func gfDiv(a, b byte) byte {
	if b == 0 {
		panic("barcode: Reed-Solomon division by zero")
	}
	if a == 0 {
		return 0
	}
	return gfExp[(int(gfLog[a])-int(gfLog[b])+255)%255]
}

// gfInverse returns the multiplicative inverse of x. It panics if x is zero.
func gfInverse(x byte) byte {
	if x == 0 {
		panic("barcode: Reed-Solomon inverse of zero")
	}
	return gfExp[(255-int(gfLog[x]))%255]
}

// gfPow raises x to an integer power (which may be negative), reducing the
// exponent modulo 255.
func gfPow(x byte, power int) byte {
	if x == 0 {
		if power == 0 {
			return 1
		}
		return 0
	}
	i := (int(gfLog[x])*power)%255 + 255
	return gfExp[i%255]
}

// gfPolyScale multiplies every coefficient of p by the scalar x.
func gfPolyScale(p []byte, x byte) []byte {
	out := make([]byte, len(p))
	for i, c := range p {
		out[i] = gfMul(c, x)
	}
	return out
}

// gfPolyAdd adds two polynomials (right-aligned, i.e. by matching lowest-degree
// terms), which in GF(2) is a coefficient-wise XOR.
func gfPolyAdd(p, q []byte) []byte {
	r := make([]byte, len(p))
	if len(q) > len(r) {
		r = make([]byte, len(q))
	}
	for i := 0; i < len(p); i++ {
		r[i+len(r)-len(p)] = p[i]
	}
	for i := 0; i < len(q); i++ {
		r[i+len(r)-len(q)] ^= q[i]
	}
	return r
}

// gfPolyMul multiplies two polynomials.
func gfPolyMul(p, q []byte) []byte {
	r := make([]byte, len(p)+len(q)-1)
	for j := 0; j < len(q); j++ {
		for i := 0; i < len(p); i++ {
			r[i+j] ^= gfMul(p[i], q[j])
		}
	}
	return r
}

// gfPolyEval evaluates polynomial p at x using Horner's method.
func gfPolyEval(p []byte, x byte) byte {
	y := p[0]
	for i := 1; i < len(p); i++ {
		y = gfMul(y, x) ^ p[i]
	}
	return y
}

// reverseBytes returns a reversed copy of b.
func reverseBytes(b []byte) []byte {
	out := make([]byte, len(b))
	for i := range b {
		out[len(b)-1-i] = b[i]
	}
	return out
}

// rsGeneratorPoly builds the RS generator polynomial with nsym roots
// alpha^0 .. alpha^(nsym-1).
func rsGeneratorPoly(nsym int) []byte {
	g := []byte{1}
	for i := 0; i < nsym; i++ {
		g = gfPolyMul(g, []byte{1, gfPow(2, i)})
	}
	return g
}

// ReedSolomonEncode computes nsym Reed-Solomon error-correction codewords for
// the given data codewords and returns them (length nsym). The full transmitted
// codeword is data followed by these EC bytes; see [ReedSolomonDecode].
func ReedSolomonEncode(data []byte, nsym int) []byte {
	if nsym <= 0 {
		return nil
	}
	gen := rsGeneratorPoly(nsym)
	buf := make([]byte, len(data)+len(gen)-1)
	copy(buf, data)
	for i := 0; i < len(data); i++ {
		coef := buf[i]
		if coef != 0 {
			for j := 1; j < len(gen); j++ {
				buf[i+j] ^= gfMul(gen[j], coef)
			}
		}
	}
	ecc := make([]byte, nsym)
	copy(ecc, buf[len(data):])
	return ecc
}

// rsCalcSyndromes returns the nsym+1 syndromes of msg (a data+EC codeword);
// element 0 is always zero, matching the reference convention. The message is
// error-free exactly when syndromes 1..nsym are all zero.
func rsCalcSyndromes(msg []byte, nsym int) []byte {
	synd := make([]byte, nsym+1)
	for i := 0; i < nsym; i++ {
		synd[i+1] = gfPolyEval(msg, gfPow(2, i))
	}
	return synd
}

// rsFindErrorLocator runs the Berlekamp-Massey algorithm to derive the error
// locator polynomial from the syndromes.
func rsFindErrorLocator(synd []byte, nsym int) []byte {
	errLoc := []byte{1}
	oldLoc := []byte{1}
	syndShift := 0
	if len(synd) > nsym {
		syndShift = len(synd) - nsym
	}
	for i := 0; i < nsym; i++ {
		k := i + syndShift
		delta := synd[k]
		for j := 1; j < len(errLoc); j++ {
			delta ^= gfMul(errLoc[len(errLoc)-1-j], synd[k-j])
		}
		oldLoc = append(oldLoc, 0)
		if delta != 0 {
			if len(oldLoc) > len(errLoc) {
				newLoc := gfPolyScale(oldLoc, delta)
				oldLoc = gfPolyScale(errLoc, gfInverse(delta))
				errLoc = newLoc
			}
			errLoc = gfPolyAdd(errLoc, gfPolyScale(oldLoc, delta))
		}
	}
	for len(errLoc) > 0 && errLoc[0] == 0 {
		errLoc = errLoc[1:]
	}
	return errLoc
}

// rsFindErrors performs a Chien search: it returns the byte positions (indices
// into a message of length nmess) at which errors occur, or nil if the number
// of roots does not match the locator degree (an uncorrectable pattern).
func rsFindErrors(errLoc []byte, nmess int) []int {
	errs := len(errLoc) - 1
	var positions []int
	// The locator's roots lie at alpha^-p, where p is the error's power-of-x
	// position. Scan the whole field, recover p, and convert to an array index.
	for i := 0; i < 255; i++ {
		if gfPolyEval(errLoc, gfExp[i]) == 0 {
			p := (255 - i) % 255
			if p < nmess {
				positions = append(positions, nmess-1-p)
			}
		}
	}
	if len(positions) != errs {
		return nil
	}
	return positions
}

// rsFindErrataLocator builds the errata locator polynomial from error
// positions expressed as coefficient positions.
func rsFindErrataLocator(coefPos []int) []byte {
	eLoc := []byte{1}
	for _, p := range coefPos {
		eLoc = gfPolyMul(eLoc, gfPolyAdd([]byte{1}, []byte{gfPow(2, p), 0}))
	}
	return eLoc
}

// rsFindErrorEvaluator computes Omega(x) = (synd(x) * errLoc(x)) mod x^(nsym+1).
func rsFindErrorEvaluator(synd, errLoc []byte, nsym int) []byte {
	mul := gfPolyMul(synd, errLoc)
	return mul[len(mul)-(nsym+1):]
}

// rsCorrectErrata applies the Forney algorithm to correct the errors at the
// given positions, returning the corrected message.
func rsCorrectErrata(msg, synd []byte, errPos []int) []byte {
	coefPos := make([]int, len(errPos))
	for i, p := range errPos {
		coefPos[i] = len(msg) - 1 - p
	}
	errLoc := rsFindErrataLocator(coefPos)
	errEval := reverseBytes(rsFindErrorEvaluator(reverseBytes(synd), errLoc, len(errLoc)-1))

	x := make([]byte, len(coefPos))
	for i, cp := range coefPos {
		x[i] = gfPow(2, cp-255)
	}

	e := make([]byte, len(msg))
	for i, xi := range x {
		xiInv := gfInverse(xi)
		var prime byte = 1
		for j := range x {
			if j != i {
				prime = gfMul(prime, 1^gfMul(xiInv, x[j]))
			}
		}
		if prime == 0 {
			return nil
		}
		y := gfPolyEval(reverseBytes(errEval), xiInv)
		y = gfMul(gfPow(xi, 1), y)
		e[errPos[i]] = gfDiv(y, prime)
	}
	return gfPolyAdd(msg, e)
}

// ReedSolomonDecode takes a received codeword (data codewords followed by nsym
// EC codewords) and attempts to correct up to nsym/2 byte errors. It returns
// the corrected codeword (same length and layout as the input) and true on
// success, or nil and false when the errors exceed the correction capacity.
func ReedSolomonDecode(msg []byte, nsym int) ([]byte, bool) {
	out := make([]byte, len(msg))
	copy(out, msg)
	synd := rsCalcSyndromes(out, nsym)
	if maxByte(synd) == 0 {
		return out, true
	}
	errLoc := rsFindErrorLocator(synd, nsym)
	errPos := rsFindErrors(errLoc, len(out))
	if errPos == nil {
		return nil, false
	}
	out = rsCorrectErrata(out, synd, errPos)
	if out == nil {
		return nil, false
	}
	if maxByte(rsCalcSyndromes(out, nsym)) != 0 {
		return nil, false
	}
	return out, true
}

// maxByte returns the largest element of b, or 0 for an empty slice.
func maxByte(b []byte) byte {
	var m byte
	for _, v := range b {
		if v > m {
			m = v
		}
	}
	return m
}
