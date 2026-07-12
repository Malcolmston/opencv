package mcc

// CheckerType selects which reference chart the detector and color model use.
// Both supported charts share the classic 24-patch, 4-row by 6-column layout;
// they differ only in their tabulated reference colors.
type CheckerType int

const (
	// Macbeth24 is the classic 24-patch X-Rite / GretagMacbeth ColorChecker,
	// using the widely-cited sRGB (D65) reference values. Patches are ordered
	// row-major: row 0 is the "natural colors" row (dark skin ... bluish green),
	// rows 1-2 the "miscellaneous" and "primary/secondary" rows, and row 3 the
	// six-step neutral gray scale (white ... black).
	Macbeth24 CheckerType = iota
	// Vinyl is the same 24-patch layout tabulated from the BabelColor community
	// average of many measured ColorChecker charts. Its values differ slightly
	// from [Macbeth24] and make a useful second, independent reference set.
	Vinyl
)

// String returns the chart's name.
func (t CheckerType) String() string {
	switch t {
	case Macbeth24:
		return "Macbeth24"
	case Vinyl:
		return "Vinyl"
	default:
		return "Unknown"
	}
}

// Rows returns the number of patch rows in the chart's grid (4 for both
// supported charts).
func (t CheckerType) Rows() int { return 4 }

// Cols returns the number of patch columns in the chart's grid (6 for both
// supported charts).
func (t CheckerType) Cols() int { return 6 }

// NumPatches returns the number of patches in the chart (24 for both supported
// charts).
func (t CheckerType) NumPatches() int { return t.Rows() * t.Cols() }

// Patch is a single reference color of a chart, holding its conventional name,
// its 8-bit sRGB value and the CIE L*a*b* (D65) value derived from that sRGB
// under this package's color math (see [RGBToLab]).
type Patch struct {
	Name string
	RGB  [3]uint8
	Lab  [3]float64
}

// patchName holds the conventional name of each of the 24 patches in grid
// order.
var patchNames = [24]string{
	"dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
	"orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
	"blue", "green", "red", "yellow", "magenta", "cyan",
	"white", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black",
}

// macbethRGB holds the classic ColorChecker sRGB (D65) reference values in grid
// order.
var macbethRGB = [24][3]uint8{
	{115, 82, 68}, {194, 150, 130}, {98, 122, 157}, {87, 108, 67}, {133, 128, 177}, {103, 189, 170},
	{214, 126, 44}, {80, 91, 166}, {193, 90, 99}, {94, 60, 108}, {157, 188, 64}, {224, 163, 46},
	{56, 61, 150}, {70, 148, 73}, {175, 54, 60}, {231, 199, 31}, {187, 86, 149}, {8, 133, 161},
	{243, 243, 242}, {200, 200, 200}, {160, 160, 160}, {122, 122, 121}, {85, 85, 85}, {52, 52, 52},
}

// vinylRGB holds the BabelColor-average sRGB (D65) reference values in grid
// order; an independent tabulation of the same 24-patch layout.
var vinylRGB = [24][3]uint8{
	{115, 83, 68}, {196, 147, 127}, {91, 122, 156}, {90, 108, 64}, {130, 128, 176}, {92, 190, 172},
	{224, 124, 47}, {68, 91, 170}, {198, 82, 97}, {94, 58, 106}, {159, 189, 63}, {230, 162, 39},
	{35, 63, 147}, {67, 149, 74}, {180, 49, 57}, {238, 198, 20}, {193, 84, 151}, {0, 136, 170},
	{245, 245, 243}, {200, 202, 202}, {161, 163, 163}, {121, 121, 122}, {82, 84, 86}, {49, 49, 51},
}

// charts caches the assembled [Patch] slices, one per CheckerType, built once at
// package initialisation so that reference Lab values are computed a single time.
var charts = map[CheckerType][]Patch{}

func init() {
	build := func(rgb [24][3]uint8) []Patch {
		out := make([]Patch, 24)
		for i := range rgb {
			out[i] = Patch{
				Name: patchNames[i],
				RGB:  rgb[i],
				Lab:  RGBToLab(rgb[i][0], rgb[i][1], rgb[i][2]),
			}
		}
		return out
	}
	charts[Macbeth24] = build(macbethRGB)
	charts[Vinyl] = build(vinylRGB)
}

// ReferenceChart returns the reference patches of the given chart in row-major
// grid order (patch index = row*Cols + col). The returned slice is a fresh copy
// that callers may modify freely. It panics on an unknown CheckerType.
func ReferenceChart(t CheckerType) []Patch {
	c, ok := charts[t]
	if !ok {
		panic("mcc: ReferenceChart unknown CheckerType")
	}
	out := make([]Patch, len(c))
	copy(out, c)
	return out
}

// ReferenceRGB returns the reference sRGB values of the given chart as float64
// triples in the 0..255 range and grid order, ready to feed to [TrainCCM] as the
// target colors.
func ReferenceRGB(t CheckerType) [][3]float64 {
	c := charts[t]
	out := make([][3]float64, len(c))
	for i, p := range c {
		out[i] = [3]float64{float64(p.RGB[0]), float64(p.RGB[1]), float64(p.RGB[2])}
	}
	return out
}

// ReferenceLab returns the reference CIE L*a*b* values of the given chart in grid
// order.
func ReferenceLab(t CheckerType) [][3]float64 {
	c := charts[t]
	out := make([][3]float64, len(c))
	for i, p := range c {
		out[i] = p.Lab
	}
	return out
}
