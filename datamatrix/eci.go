package datamatrix

// This file encodes and decodes Extended Channel Interpretation (ECI)
// assignments. In ECC200 an ECI is introduced by the ASCII codeword 241
// followed by one, two or three codewords that carry the ECI number, using the
// range-based scheme of ISO/IEC 16022. An ECI selects the character set /
// interpretation of the following data (for example ECI 000003 is ISO-8859-1
// and ECI 000026 is UTF-8).

const eciCodeword = 241

// eciMaxValue is the largest ECI number this codec can represent.
const eciMaxValue = 999999

// encodeECI returns the codewords (including the leading 241) that select the
// given ECI number.
func encodeECI(eci int) []int {
	switch {
	case eci <= 126:
		return []int{eciCodeword, eci + 1}
	case eci <= 16382:
		v := eci - 127
		return []int{eciCodeword, v/254 + 128, v%254 + 1}
	default:
		v := eci - 16383
		return []int{eciCodeword, v/64516 + 192, (v/254)%254 + 1, v%254 + 1}
	}
}

// decodeECI reads the ECI number that follows a 241 codeword. cw is the slice of
// data codewords starting immediately after the 241; it returns the ECI value
// and the number of codewords consumed.
func decodeECI(cw []int) (eci, used int, err error) {
	if len(cw) == 0 {
		return 0, 0, errBadCodewords
	}
	c1 := cw[0]
	switch {
	case c1 <= 127:
		return c1 - 1, 1, nil
	case c1 <= 191:
		if len(cw) < 2 {
			return 0, 0, errBadCodewords
		}
		return (c1-128)*254 + (cw[1] - 1) + 127, 2, nil
	default:
		if len(cw) < 3 {
			return 0, 0, errBadCodewords
		}
		return (c1-192)*64516 + (cw[1]-1)*254 + (cw[2] - 1) + 16383, 3, nil
	}
}
