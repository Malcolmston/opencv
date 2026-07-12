package datamatrix

// This file implements arithmetic in the Galois field GF(256) and a
// Reed-Solomon codec (encoder plus a full syndrome/Berlekamp-Massey/Chien/
// Forney error-correcting decoder) using the parameters mandated by the
// ECC200 flavour of Data Matrix (ISO/IEC 16022):
//
//   - field-generating primitive polynomial 0x12D (x^8+x^5+x^3+x^2+1),
//   - field generator element alpha = 2,
//   - first consecutive root exponent (fcr) = 1, so the Reed-Solomon
//     generator polynomial has roots alpha^1 .. alpha^n.
//
// These choices reproduce the exact generator-polynomial coefficients listed
// in the ECC200 specification (for example the five error-correction
// codewords of the 10x10 symbol use the polynomial {228, 48, 15, 111, 62}).
//
// The implementation follows the well-known, self-consistent reference from
// the "Reed-Solomon codes for coders" derivation, re-cast into Go and
// generalised to an arbitrary first-consecutive-root so it matches ECC200.

const (
	gfPrim      = 0x12d // primitive polynomial for GF(256) used by ECC200
	gfGenerator = 2     // field generator element (alpha)
	gfFCR       = 1     // first consecutive root exponent for ECC200
)

// gfExp and gfLog are the antilog and log tables for GF(256). gfExp is sized
// 512 so that gfExp[gfLog[a]+gfLog[b]] never needs an explicit modulo.
var (
	gfExp [512]int
	gfLog [256]int
)

func init() {
	x := 1
	for i := 0; i < 255; i++ {
		gfExp[i] = x
		gfLog[x] = i
		x <<= 1
		if x&0x100 != 0 {
			x ^= gfPrim
		}
	}
	for i := 255; i < 512; i++ {
		gfExp[i] = gfExp[i-255]
	}
}

func gfMul(a, b int) int {
	if a == 0 || b == 0 {
		return 0
	}
	return gfExp[gfLog[a]+gfLog[b]]
}

func gfDiv(a, b int) int {
	if b == 0 {
		panic("datamatrix: division by zero in GF(256)")
	}
	if a == 0 {
		return 0
	}
	return gfExp[(gfLog[a]+255-gfLog[b])%255]
}

// gfPow returns x raised to power in GF(256); power may be negative.
func gfPow(x, power int) int {
	if x == 0 {
		if power == 0 {
			return 1
		}
		return 0
	}
	i := (gfLog[x] * power) % 255
	if i < 0 {
		i += 255
	}
	return gfExp[i]
}

func gfInverse(x int) int {
	return gfExp[255-gfLog[x]]
}

// gfPolyScale multiplies every coefficient of p by scalar x.
func gfPolyScale(p []int, x int) []int {
	out := make([]int, len(p))
	for i := range p {
		out[i] = gfMul(p[i], x)
	}
	return out
}

// gfPolyAdd adds (XORs) two polynomials whose coefficients are stored
// highest-degree first, aligning them on their lowest-degree term.
func gfPolyAdd(p, q []int) []int {
	n := len(p)
	if len(q) > n {
		n = len(q)
	}
	out := make([]int, n)
	for i := range p {
		out[i+n-len(p)] = p[i]
	}
	for i := range q {
		out[i+n-len(q)] ^= q[i]
	}
	return out
}

// gfPolyMul multiplies two polynomials over GF(256).
func gfPolyMul(p, q []int) []int {
	out := make([]int, len(p)+len(q)-1)
	for j := range q {
		for i := range p {
			out[i+j] ^= gfMul(p[i], q[j])
		}
	}
	return out
}

// gfPolyEval evaluates polynomial p (highest-degree first) at x using Horner's
// method.
func gfPolyEval(p []int, x int) int {
	y := p[0]
	for i := 1; i < len(p); i++ {
		y = gfMul(y, x) ^ p[i]
	}
	return y
}

// rsGeneratorPoly returns the Reed-Solomon generator polynomial for nsym
// error-correction symbols, with roots alpha^fcr .. alpha^(fcr+nsym-1).
func rsGeneratorPoly(nsym int) []int {
	g := []int{1}
	for i := 0; i < nsym; i++ {
		g = gfPolyMul(g, []int{1, gfPow(gfGenerator, i+gfFCR)})
	}
	return g
}

// rsEncode returns the full codeword (data followed by nsym error-correction
// codewords) for the given data codewords.
func rsEncode(data []int, nsym int) []int {
	gen := rsGeneratorPoly(nsym)
	out := make([]int, len(data)+nsym)
	copy(out, data)
	for i := 0; i < len(data); i++ {
		coef := out[i]
		if coef != 0 {
			for j := 1; j < len(gen); j++ {
				out[i+j] ^= gfMul(gen[j], coef)
			}
		}
	}
	// The synthetic division clobbered the data portion; restore it so the
	// result is a proper systematic codeword.
	copy(out, data)
	return out
}

// rsCalcSyndromes computes the nsym syndromes of msg. The returned slice is
// prefixed with a leading zero (length nsym+1) to match the downstream
// Berlekamp-Massey routine.
func rsCalcSyndromes(msg []int, nsym int) []int {
	synd := make([]int, nsym+1)
	for i := 0; i < nsym; i++ {
		synd[i+1] = gfPolyEval(msg, gfPow(gfGenerator, i+gfFCR))
	}
	return synd
}

// rsFindErrorLocator runs the Berlekamp-Massey algorithm and returns the error
// locator polynomial (highest-degree first).
func rsFindErrorLocator(synd []int, nsym int) []int {
	errLoc := []int{1}
	oldLoc := []int{1}
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
	// Strip leading zeros.
	for len(errLoc) > 0 && errLoc[0] == 0 {
		errLoc = errLoc[1:]
	}
	return errLoc
}

// rsFindErrors performs a Chien search, returning the error positions as
// indices into a codeword of length nmess (0 = highest-degree coefficient).
func rsFindErrors(errLoc []int, nmess int) ([]int, bool) {
	errs := len(errLoc) - 1
	var errPos []int
	for i := 0; i < nmess; i++ {
		if gfPolyEval(errLoc, gfPow(gfGenerator, i)) == 0 {
			errPos = append(errPos, nmess-1-i)
		}
	}
	if len(errPos) != errs {
		return nil, false
	}
	return errPos, true
}

// rsFindErrataLocator builds the errata locator polynomial for the given
// coefficient positions.
func rsFindErrataLocator(coefPos []int) []int {
	eLoc := []int{1}
	for _, p := range coefPos {
		eLoc = gfPolyMul(eLoc, gfPolyAdd([]int{1}, []int{gfPow(gfGenerator, p), 0}))
	}
	return eLoc
}

// rsFindErrorEvaluator computes Omega(x) = (S(x)*errLoc) mod x^(nsym+1).
func rsFindErrorEvaluator(synd, errLoc []int, nsym int) []int {
	product := gfPolyMul(synd, errLoc)
	if len(product) > nsym+1 {
		product = product[len(product)-(nsym+1):]
	}
	return product
}

func reverseInts(p []int) []int {
	out := make([]int, len(p))
	for i := range p {
		out[len(p)-1-i] = p[i]
	}
	return out
}

// rsCorrectErrata corrects the errors at the given positions in msg using the
// Forney algorithm and returns the corrected codeword.
func rsCorrectErrata(msg, synd, errPos []int) []int {
	coefPos := make([]int, len(errPos))
	for i, p := range errPos {
		coefPos[i] = len(msg) - 1 - p
	}
	errLoc := rsFindErrataLocator(coefPos)
	errEval := reverseInts(rsFindErrorEvaluator(reverseInts(synd), errLoc, len(errLoc)-1))

	// X holds the error locators.
	x := make([]int, len(coefPos))
	for i := range coefPos {
		l := 255 - coefPos[i]
		x[i] = gfPow(gfGenerator, -l)
	}

	e := make([]int, len(msg))
	for i, xi := range x {
		xiInv := gfInverse(xi)
		errLocPrime := 1
		for j := range x {
			if j != i {
				errLocPrime = gfMul(errLocPrime, 1^gfMul(xiInv, x[j]))
			}
		}
		if errLocPrime == 0 {
			// Should not happen for a correctable pattern.
			continue
		}
		y := gfPolyEval(reverseInts(errEval), xiInv)
		y = gfMul(gfPow(xi, 1-gfFCR), y)
		e[errPos[i]] = gfDiv(y, errLocPrime)
	}
	return gfPolyAdd(msg, e)
}

// rsCorrect attempts to correct up to nsym/2 errors in the codeword msg. It
// returns the corrected codeword and the number of errors repaired, or an
// error if the codeword cannot be corrected.
func rsCorrect(msg []int, nsym int) ([]int, int, error) {
	out := make([]int, len(msg))
	copy(out, msg)
	synd := rsCalcSyndromes(out, nsym)
	if maxInt(synd) == 0 {
		return out, 0, nil
	}
	errLoc := rsFindErrorLocator(synd, nsym)
	errPos, ok := rsFindErrors(reverseInts(errLoc), len(out))
	if !ok {
		return nil, 0, errTooManyErrors
	}
	out = rsCorrectErrata(out, synd, errPos)
	if maxInt(rsCalcSyndromes(out, nsym)) != 0 {
		return nil, 0, errTooManyErrors
	}
	return out, len(errPos), nil
}

func maxInt(v []int) int {
	m := 0
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}
