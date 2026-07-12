package hdr

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	cv "github.com/malcolmston/opencv"
)

// --- Statistics --------------------------------------------------------------

// MinMax returns the smallest and largest sample across every channel of the
// radiance map.
func (r *Radiance) MinMax() (min, max float64) {
	min, max = math.Inf(1), math.Inf(-1)
	for _, v := range r.Data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}

// Mean returns the arithmetic mean of every sample across all channels.
func (r *Radiance) Mean() float64 {
	if len(r.Data) == 0 {
		return 0
	}
	var s float64
	for _, v := range r.Data {
		s += v
	}
	return s / float64(len(r.Data))
}

// LuminanceFloatMat returns the relative luminance of the radiance map as a
// single-channel [cv.FloatMat] (Rec.709 weights for colour, a copy for a
// single-channel map).
func (r *Radiance) LuminanceFloatMat() *cv.FloatMat {
	lum := r.luminance()
	out := cv.NewFloatMat(r.Rows, r.Cols)
	copy(out.Data, lum.data)
	return out
}

// LogAverageLuminance returns the log-average (geometric mean) luminance, the
// key statistic most tone mappers anchor on. Non-positive luminances are guarded
// with a small epsilon.
func (r *Radiance) LogAverageLuminance() float64 {
	lum := r.luminance()
	logAvg, _ := logAvgLuminance(lum)
	return logAvg
}

// DynamicRange returns the scene's dynamic range in stops (powers of two),
// log2(maxLum/minLum), computed over the strictly positive luminances. It
// returns 0 when fewer than two distinct positive luminances exist.
func (r *Radiance) DynamicRange() float64 {
	lum := r.luminance()
	minL, maxL := math.Inf(1), math.Inf(-1)
	found := false
	for _, v := range lum.data {
		if v <= 0 {
			continue
		}
		found = true
		if v < minL {
			minL = v
		}
		if v > maxL {
			maxL = v
		}
	}
	if !found || minL <= 0 || maxL <= minL {
		return 0
	}
	return math.Log2(maxL / minL)
}

// Scale returns a new radiance map with every sample multiplied by f. It is the
// exposure-adjustment counterpart of multiplying an LDR image's brightness.
func (r *Radiance) Scale(f float64) *Radiance {
	out := NewRadiance(r.Rows, r.Cols, r.Channels)
	for i, v := range r.Data {
		out.Data[i] = v * f
	}
	return out
}

// Normalized returns a new radiance map divided by its maximum sample so that
// the peak radiance is 1. A non-positive maximum leaves the data unchanged.
func (r *Radiance) Normalized() *Radiance {
	_, max := r.MinMax()
	if max <= 0 {
		return r.Clone()
	}
	return r.Scale(1.0 / max)
}

// --- PFM I/O -----------------------------------------------------------------

// WritePFM encodes the radiance map as a Portable Float Map (PFM), a simple
// binary float image format understood by many HDR tools. Colour maps are
// written as "PF" (three channels), single-channel maps as "Pf"; other channel
// counts return an error. Samples are stored as little-endian float32, bottom
// row first, matching the PFM convention.
func WritePFM(w io.Writer, r *Radiance) error {
	if r == nil || len(r.Data) == 0 {
		return errors.New("hdr: WritePFM on empty radiance")
	}
	var magic string
	switch r.Channels {
	case 1:
		magic = "Pf"
	case 3:
		magic = "PF"
	default:
		return fmt.Errorf("hdr: WritePFM supports 1 or 3 channels, got %d", r.Channels)
	}
	bw := bufio.NewWriter(w)
	// Negative scale signals little-endian byte order.
	if _, err := fmt.Fprintf(bw, "%s\n%d %d\n-1.0\n", magic, r.Cols, r.Rows); err != nil {
		return err
	}
	buf := make([]byte, 4)
	ch := r.Channels
	// PFM stores rows bottom-to-top.
	for y := r.Rows - 1; y >= 0; y-- {
		for x := 0; x < r.Cols; x++ {
			base := (y*r.Cols + x) * ch
			for c := 0; c < ch; c++ {
				binary.LittleEndian.PutUint32(buf, math.Float32bits(float32(r.Data[base+c])))
				if _, err := bw.Write(buf); err != nil {
					return err
				}
			}
		}
	}
	return bw.Flush()
}

// ReadPFM decodes a Portable Float Map written by [WritePFM] (or any other tool)
// into a radiance map. Both little- and big-endian PFMs are accepted; the byte
// order is taken from the sign of the scale line.
func ReadPFM(rd io.Reader) (*Radiance, error) {
	br := bufio.NewReader(rd)
	magic, err := readPFMToken(br)
	if err != nil {
		return nil, err
	}
	var ch int
	switch magic {
	case "PF":
		ch = 3
	case "Pf":
		ch = 1
	default:
		return nil, fmt.Errorf("hdr: ReadPFM bad magic %q", magic)
	}
	wTok, err := readPFMToken(br)
	if err != nil {
		return nil, err
	}
	hTok, err := readPFMToken(br)
	if err != nil {
		return nil, err
	}
	sTok, err := readPFMToken(br)
	if err != nil {
		return nil, err
	}
	var cols, rows int
	if _, err := fmt.Sscan(wTok, &cols); err != nil {
		return nil, fmt.Errorf("hdr: ReadPFM bad width: %w", err)
	}
	if _, err := fmt.Sscan(hTok, &rows); err != nil {
		return nil, fmt.Errorf("hdr: ReadPFM bad height: %w", err)
	}
	var scale float64
	if _, err := fmt.Sscan(sTok, &scale); err != nil {
		return nil, fmt.Errorf("hdr: ReadPFM bad scale: %w", err)
	}
	if cols <= 0 || rows <= 0 {
		return nil, errors.New("hdr: ReadPFM non-positive dimensions")
	}
	little := scale < 0
	out := NewRadiance(rows, cols, ch)
	buf := make([]byte, 4)
	for y := rows - 1; y >= 0; y-- {
		for x := 0; x < cols; x++ {
			base := (y*cols + x) * ch
			for c := 0; c < ch; c++ {
				if _, err := io.ReadFull(br, buf); err != nil {
					return nil, err
				}
				var bits uint32
				if little {
					bits = binary.LittleEndian.Uint32(buf)
				} else {
					bits = binary.BigEndian.Uint32(buf)
				}
				out.Data[base+c] = float64(math.Float32frombits(bits))
			}
		}
	}
	return out, nil
}

// readPFMToken reads the next whitespace-delimited token of a PFM header.
func readPFMToken(br *bufio.Reader) (string, error) {
	var buf []byte
	// Skip leading whitespace.
	for {
		b, err := br.ReadByte()
		if err != nil {
			return "", err
		}
		if b == ' ' || b == '\n' || b == '\r' || b == '\t' {
			continue
		}
		buf = append(buf, b)
		break
	}
	for {
		b, err := br.ReadByte()
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}
		if b == ' ' || b == '\n' || b == '\r' || b == '\t' {
			break
		}
		buf = append(buf, b)
	}
	return string(buf), nil
}

// --- Radiance RGBE (.hdr) I/O -----------------------------------------------

// WriteHDR encodes a three-channel radiance map in the classic Radiance RGBE
// (.hdr / .pic) format using a shared exponent per pixel, the standard
// interchange format for HDR imagery. Scanlines are written uncompressed (the
// "flat" RGBE encoding, which every reader accepts). It returns an error for
// non-three-channel maps.
func WriteHDR(w io.Writer, r *Radiance) error {
	if r == nil || len(r.Data) == 0 {
		return errors.New("hdr: WriteHDR on empty radiance")
	}
	if r.Channels != 3 {
		return fmt.Errorf("hdr: WriteHDR requires 3 channels, got %d", r.Channels)
	}
	bw := bufio.NewWriter(w)
	if _, err := io.WriteString(bw, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n"); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(bw, "-Y %d +X %d\n", r.Rows, r.Cols); err != nil {
		return err
	}
	var rgbe [4]byte
	for i := 0; i < r.Rows*r.Cols; i++ {
		base := i * 3
		floatToRGBE(r.Data[base+0], r.Data[base+1], r.Data[base+2], &rgbe)
		if _, err := bw.Write(rgbe[:]); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// ReadHDR decodes an uncompressed Radiance RGBE stream written by [WriteHDR]
// into a three-channel radiance map. Run-length-encoded scanlines are not
// supported (this reader is the exact counterpart of the flat writer).
func ReadHDR(rd io.Reader) (*Radiance, error) {
	br := bufio.NewReader(rd)
	// Header: read lines until a blank line, then the resolution line.
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return nil, err
		}
		if line == "\n" || line == "\r\n" {
			break
		}
	}
	resLine, err := br.ReadString('\n')
	if err != nil {
		return nil, err
	}
	var rows, cols int
	if _, err := fmt.Sscanf(resLine, "-Y %d +X %d", &rows, &cols); err != nil {
		return nil, fmt.Errorf("hdr: ReadHDR bad resolution line %q: %w", resLine, err)
	}
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("hdr: ReadHDR non-positive dimensions")
	}
	out := NewRadiance(rows, cols, 3)
	var rgbe [4]byte
	for i := 0; i < rows*cols; i++ {
		if _, err := io.ReadFull(br, rgbe[:]); err != nil {
			return nil, err
		}
		rr, gg, bb := rgbeToFloat(rgbe)
		base := i * 3
		out.Data[base+0] = rr
		out.Data[base+1] = gg
		out.Data[base+2] = bb
	}
	return out, nil
}

// floatToRGBE packs three floats into a 4-byte shared-exponent RGBE sample.
func floatToRGBE(r, g, b float64, out *[4]byte) {
	if r < 0 {
		r = 0
	}
	if g < 0 {
		g = 0
	}
	if b < 0 {
		b = 0
	}
	max := r
	if g > max {
		max = g
	}
	if b > max {
		max = b
	}
	if max < 1e-32 {
		out[0], out[1], out[2], out[3] = 0, 0, 0, 0
		return
	}
	mant, exp := math.Frexp(max) // max = mant * 2^exp, mant in [0.5,1)
	scale := mant * 256.0 / max
	out[0] = byte(r * scale)
	out[1] = byte(g * scale)
	out[2] = byte(b * scale)
	out[3] = byte(exp + 128)
}

// rgbeToFloat unpacks a shared-exponent RGBE sample into three floats.
func rgbeToFloat(in [4]byte) (r, g, b float64) {
	if in[3] == 0 {
		return 0, 0, 0
	}
	f := math.Ldexp(1.0, int(in[3])-(128+8))
	r = (float64(in[0]) + 0.5) * f
	g = (float64(in[1]) + 0.5) * f
	b = (float64(in[2]) + 0.5) * f
	return r, g, b
}
