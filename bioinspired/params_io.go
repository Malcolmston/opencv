package bioinspired

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
)

// Validate reports whether the parvo (OPL+IPL) parameters are in range: the
// sensitivity/gain fields must lie in [0,1] and the temporal/spatial constants
// must be non-negative.
func (p OPLandIplParvoParameters) Validate() error {
	unit := []struct {
		name string
		v    float64
	}{
		{"PhotoreceptorsLocalAdaptationSensitivity", p.PhotoreceptorsLocalAdaptationSensitivity},
		{"HorizontalCellsGain", p.HorizontalCellsGain},
		{"GanglionCellsSensitivity", p.GanglionCellsSensitivity},
	}
	for _, f := range unit {
		if f.v < 0 || f.v > 1 {
			return fmt.Errorf("bioinspired: OPLandIplParvo.%s must be in [0,1], got %g", f.name, f.v)
		}
	}
	nonneg := []struct {
		name string
		v    float64
	}{
		{"PhotoreceptorsTemporalConstant", p.PhotoreceptorsTemporalConstant},
		{"PhotoreceptorsSpatialConstant", p.PhotoreceptorsSpatialConstant},
		{"HCellsTemporalConstant", p.HCellsTemporalConstant},
		{"HCellsSpatialConstant", p.HCellsSpatialConstant},
	}
	for _, f := range nonneg {
		if f.v < 0 {
			return fmt.Errorf("bioinspired: OPLandIplParvo.%s must be >= 0, got %g", f.name, f.v)
		}
	}
	return nil
}

// Validate reports whether the magno (IPL) parameters are in range: the
// compression sensitivity must lie in [0,1], the gain must be non-negative and
// the temporal/spatial constants must be non-negative.
func (p IplMagnoParameters) Validate() error {
	if p.V0CompressionParameter < 0 || p.V0CompressionParameter > 1 {
		return fmt.Errorf("bioinspired: IplMagno.V0CompressionParameter must be in [0,1], got %g", p.V0CompressionParameter)
	}
	if p.MagnoGain < 0 {
		return fmt.Errorf("bioinspired: IplMagno.MagnoGain must be >= 0, got %g", p.MagnoGain)
	}
	nonneg := []struct {
		name string
		v    float64
	}{
		{"ParasolCellsTau", p.ParasolCellsTau},
		{"ParasolCellsK", p.ParasolCellsK},
		{"AmacrinCellsTemporalCutFrequency", p.AmacrinCellsTemporalCutFrequency},
		{"LocalAdaptintegrationTau", p.LocalAdaptintegrationTau},
		{"LocalAdaptintegrationK", p.LocalAdaptintegrationK},
	}
	for _, f := range nonneg {
		if f.v < 0 {
			return fmt.Errorf("bioinspired: IplMagno.%s must be >= 0, got %g", f.name, f.v)
		}
	}
	return nil
}

// Validate reports whether every parameter of both retina pathways is in range.
// It returns the first violation found, or nil if the parameters are usable.
func (p RetinaParameters) Validate() error {
	if err := p.OPLandIplParvo.Validate(); err != nil {
		return err
	}
	return p.IplMagno.Validate()
}

// paramLines lists the text keys, in write order, with getters and setters onto
// a *RetinaParameters. Booleans are encoded as 0/1.
var paramLines = []struct {
	key string
	get func(p *RetinaParameters) float64
	set func(p *RetinaParameters, v float64)
}{
	{"OPLandIplParvo.ColorMode",
		func(p *RetinaParameters) float64 { return b2f(p.OPLandIplParvo.ColorMode) },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.ColorMode = v != 0 }},
	{"OPLandIplParvo.PhotoreceptorsLocalAdaptationSensitivity",
		func(p *RetinaParameters) float64 { return p.OPLandIplParvo.PhotoreceptorsLocalAdaptationSensitivity },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.PhotoreceptorsLocalAdaptationSensitivity = v }},
	{"OPLandIplParvo.PhotoreceptorsTemporalConstant",
		func(p *RetinaParameters) float64 { return p.OPLandIplParvo.PhotoreceptorsTemporalConstant },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.PhotoreceptorsTemporalConstant = v }},
	{"OPLandIplParvo.PhotoreceptorsSpatialConstant",
		func(p *RetinaParameters) float64 { return p.OPLandIplParvo.PhotoreceptorsSpatialConstant },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.PhotoreceptorsSpatialConstant = v }},
	{"OPLandIplParvo.HorizontalCellsGain",
		func(p *RetinaParameters) float64 { return p.OPLandIplParvo.HorizontalCellsGain },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.HorizontalCellsGain = v }},
	{"OPLandIplParvo.HCellsTemporalConstant",
		func(p *RetinaParameters) float64 { return p.OPLandIplParvo.HCellsTemporalConstant },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.HCellsTemporalConstant = v }},
	{"OPLandIplParvo.HCellsSpatialConstant",
		func(p *RetinaParameters) float64 { return p.OPLandIplParvo.HCellsSpatialConstant },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.HCellsSpatialConstant = v }},
	{"OPLandIplParvo.GanglionCellsSensitivity",
		func(p *RetinaParameters) float64 { return p.OPLandIplParvo.GanglionCellsSensitivity },
		func(p *RetinaParameters, v float64) { p.OPLandIplParvo.GanglionCellsSensitivity = v }},
	{"IplMagno.ParasolCellsBeta",
		func(p *RetinaParameters) float64 { return p.IplMagno.ParasolCellsBeta },
		func(p *RetinaParameters, v float64) { p.IplMagno.ParasolCellsBeta = v }},
	{"IplMagno.ParasolCellsTau",
		func(p *RetinaParameters) float64 { return p.IplMagno.ParasolCellsTau },
		func(p *RetinaParameters, v float64) { p.IplMagno.ParasolCellsTau = v }},
	{"IplMagno.ParasolCellsK",
		func(p *RetinaParameters) float64 { return p.IplMagno.ParasolCellsK },
		func(p *RetinaParameters, v float64) { p.IplMagno.ParasolCellsK = v }},
	{"IplMagno.AmacrinCellsTemporalCutFrequency",
		func(p *RetinaParameters) float64 { return p.IplMagno.AmacrinCellsTemporalCutFrequency },
		func(p *RetinaParameters, v float64) { p.IplMagno.AmacrinCellsTemporalCutFrequency = v }},
	{"IplMagno.V0CompressionParameter",
		func(p *RetinaParameters) float64 { return p.IplMagno.V0CompressionParameter },
		func(p *RetinaParameters, v float64) { p.IplMagno.V0CompressionParameter = v }},
	{"IplMagno.LocalAdaptintegrationTau",
		func(p *RetinaParameters) float64 { return p.IplMagno.LocalAdaptintegrationTau },
		func(p *RetinaParameters, v float64) { p.IplMagno.LocalAdaptintegrationTau = v }},
	{"IplMagno.LocalAdaptintegrationK",
		func(p *RetinaParameters) float64 { return p.IplMagno.LocalAdaptintegrationK },
		func(p *RetinaParameters, v float64) { p.IplMagno.LocalAdaptintegrationK = v }},
	{"IplMagno.MagnoGain",
		func(p *RetinaParameters) float64 { return p.IplMagno.MagnoGain },
		func(p *RetinaParameters, v float64) { p.IplMagno.MagnoGain = v }},
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// WriteRetinaParameters serialises a parameter set to a deterministic, plain
// text format: one "key value" pair per line, keys in a fixed order, booleans
// encoded as 0/1. The output round-trips through [ReadRetinaParameters].
func WriteRetinaParameters(p RetinaParameters) string {
	var sb strings.Builder
	for _, pl := range paramLines {
		fmt.Fprintf(&sb, "%s %s\n", pl.key, strconv.FormatFloat(pl.get(&p), 'g', -1, 64))
	}
	return sb.String()
}

// ReadRetinaParameters parses the text produced by [WriteRetinaParameters].
// Parsing starts from [DefaultRetinaParameters], so any key omitted from the
// text keeps its default value. Blank lines and lines beginning with '#' are
// ignored. Unknown keys and malformed numbers return an error, as do parameters
// that fail [RetinaParameters.Validate].
func ReadRetinaParameters(text string) (RetinaParameters, error) {
	byKey := make(map[string]func(p *RetinaParameters, v float64), len(paramLines))
	for _, pl := range paramLines {
		byKey[pl.key] = pl.set
	}

	p := DefaultRetinaParameters()
	sc := bufio.NewScanner(strings.NewReader(text))
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		fields := strings.Fields(raw)
		if len(fields) != 2 {
			return RetinaParameters{}, fmt.Errorf("bioinspired: malformed parameter line %d: %q", line, raw)
		}
		set, ok := byKey[fields[0]]
		if !ok {
			return RetinaParameters{}, fmt.Errorf("bioinspired: unknown parameter key %q on line %d", fields[0], line)
		}
		v, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return RetinaParameters{}, fmt.Errorf("bioinspired: bad value %q for %q on line %d: %w", fields[1], fields[0], line, err)
		}
		set(&p, v)
	}
	if err := sc.Err(); err != nil {
		return RetinaParameters{}, fmt.Errorf("bioinspired: reading parameters: %w", err)
	}
	if err := p.Validate(); err != nil {
		return RetinaParameters{}, err
	}
	return p, nil
}
