package datamatrix

// symbolSpec describes one supported (square) ECC200 Data Matrix symbol size.
//
// Only single data-region square symbols are modelled here; see the package
// documentation for the DEFERRED list. The mapping-matrix side length equals
// Size-2 because a one-module finder/timing border surrounds the single data
// region.
type symbolSpec struct {
	// Size is the full symbol side length in modules, finder pattern included.
	Size int
	// DataCW is the number of data codewords the symbol carries.
	DataCW int
	// ECCW is the number of Reed-Solomon error-correction codewords.
	ECCW int
}

// TotalCW returns the total number of codewords (data + error correction).
func (s symbolSpec) TotalCW() int { return s.DataCW + s.ECCW }

// MappingSize returns the side length of the interior mapping matrix.
func (s symbolSpec) MappingSize() int { return s.Size - 2 }

// symbolSpecs lists the supported square ECC200 symbols in ascending capacity
// order. The data/error codeword counts are taken directly from ISO/IEC 16022.
var symbolSpecs = []symbolSpec{
	{Size: 10, DataCW: 3, ECCW: 5},
	{Size: 12, DataCW: 5, ECCW: 7},
	{Size: 14, DataCW: 8, ECCW: 10},
	{Size: 16, DataCW: 12, ECCW: 12},
	{Size: 18, DataCW: 18, ECCW: 14},
	{Size: 20, DataCW: 22, ECCW: 18},
}

// smallestSymbolFor returns the smallest supported symbol whose data-codeword
// capacity is at least need, and reports whether one exists.
func smallestSymbolFor(need int) (symbolSpec, bool) {
	for _, s := range symbolSpecs {
		if s.DataCW >= need {
			return s, true
		}
	}
	return symbolSpec{}, false
}

// symbolBySize returns the spec for a given full symbol side length.
func symbolBySize(size int) (symbolSpec, bool) {
	for _, s := range symbolSpecs {
		if s.Size == size {
			return s, true
		}
	}
	return symbolSpec{}, false
}
