package videoio

import (
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"os"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// PaletteWebSafe is the deterministic 216-colour web-safe palette used by the
// default GIF writer. Its channel levels are spaced 51 apart, bounding the
// nearest-colour error to about 26 per channel.
var PaletteWebSafe = color.Palette(palette.WebSafe)

// PalettePlan9 is the 256-colour palette from the Plan 9 operating system. It
// covers the colour cube more evenly than the web-safe palette and usually
// reproduces photographic frames with less error, at the cost of using all 256
// slots.
var PalettePlan9 = color.Palette(palette.Plan9)

// PalettedGIFWriter encodes frames to an animated GIF with caller-controlled
// quantization: any [color.Palette] may be supplied (a fixed one such as
// [PaletteWebSafe] / [PalettePlan9], or one built by [AdaptivePalette]), the
// loop count is configurable, and each frame carries its own delay. The canvas
// size is fixed by the first frame. The zero value is not usable — obtain one
// from [NewPalettedGIFWriter].
type PalettedGIFWriter struct {
	path      string
	palette   color.Palette
	loopCount int
	disposal  byte
	bounds    image.Rectangle
	images    []*image.Paletted
	delays    []int
	released  bool
}

// NewPalettedGIFWriter creates a GIF writer that maps every frame onto pal. A
// nil or empty palette falls back to [PaletteWebSafe]; a palette longer than 256
// colours is rejected, matching the GIF format limit. loopCount follows the GIF
// convention: 0 loops forever, a positive n plays n+1 times total. No file is
// created until [PalettedGIFWriter.Release].
func NewPalettedGIFWriter(path string, pal color.Palette, loopCount int) (*PalettedGIFWriter, error) {
	if path == "" {
		return nil, fmt.Errorf("videoio: NewPalettedGIFWriter: empty path")
	}
	if len(pal) == 0 {
		pal = PaletteWebSafe
	}
	if len(pal) > 256 {
		return nil, fmt.Errorf("videoio: NewPalettedGIFWriter: palette has %d colours, max 256", len(pal))
	}
	if loopCount < 0 {
		loopCount = 0
	}
	return &PalettedGIFWriter{path: path, palette: pal, loopCount: loopCount, disposal: gif.DisposalNone}, nil
}

// WriteFrame quantizes frame to the writer's palette and appends it with a
// display duration of delayCentis centiseconds, so each frame may play for a
// different length of time. The first frame fixes the canvas; later frames are
// drawn at the origin and clipped. It errors on an empty frame or after Release.
func (w *PalettedGIFWriter) WriteFrame(frame *cv.Mat, delayCentis int) error {
	if w == nil {
		return fmt.Errorf("videoio: WriteFrame on nil PalettedGIFWriter")
	}
	if w.released {
		return fmt.Errorf("videoio: PalettedGIFWriter.WriteFrame after Release")
	}
	if frame.Empty() {
		return fmt.Errorf("videoio: PalettedGIFWriter.WriteFrame: empty frame")
	}
	if delayCentis < 0 {
		delayCentis = 0
	}
	if len(w.images) == 0 {
		w.bounds = image.Rect(0, 0, frame.Cols, frame.Rows)
	}
	src := frame.ToImage()
	paletted := image.NewPaletted(w.bounds, w.palette)
	draw.Draw(paletted, w.bounds.Intersect(src.Bounds()), src, src.Bounds().Min, draw.Src)
	w.images = append(w.images, paletted)
	w.delays = append(w.delays, delayCentis)
	return nil
}

// Release encodes every frame to the destination GIF and finalizes the writer.
// Calling it twice, or with no frames, returns an error.
func (w *PalettedGIFWriter) Release() error {
	if w == nil {
		return fmt.Errorf("videoio: Release on nil PalettedGIFWriter")
	}
	if w.released {
		return fmt.Errorf("videoio: PalettedGIFWriter Release called twice")
	}
	if len(w.images) == 0 {
		return fmt.Errorf("videoio: PalettedGIFWriter Release: no frames written")
	}
	w.released = true

	disposal := make([]byte, len(w.images))
	for i := range disposal {
		disposal[i] = w.disposal
	}
	g := &gif.GIF{
		Image:     w.images,
		Delay:     append([]int(nil), w.delays...),
		Disposal:  disposal,
		LoopCount: w.loopCount,
		Config: image.Config{
			ColorModel: w.palette,
			Width:      w.bounds.Dx(),
			Height:     w.bounds.Dy(),
		},
	}
	f, err := os.Create(w.path)
	if err != nil {
		return fmt.Errorf("videoio: PalettedGIFWriter Release create %q: %w", w.path, err)
	}
	if err := gif.EncodeAll(f, g); err != nil {
		f.Close()
		return fmt.Errorf("videoio: PalettedGIFWriter Release encode %q: %w", w.path, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("videoio: PalettedGIFWriter Release close %q: %w", w.path, err)
	}
	return nil
}

// WriteGIFDelays encodes frames to an animated GIF at path with an independent
// per-frame delay (centiseconds) and a chosen palette and loop count. It is the
// batch counterpart to [PalettedGIFWriter]. len(delays) must equal len(frames).
// A nil palette defaults to [PaletteWebSafe].
func WriteGIFDelays(path string, frames []*cv.Mat, delays []int, pal color.Palette, loopCount int) error {
	if len(frames) == 0 {
		return fmt.Errorf("videoio: WriteGIFDelays: no frames")
	}
	if len(delays) != len(frames) {
		return fmt.Errorf("videoio: WriteGIFDelays: %d delays for %d frames", len(delays), len(frames))
	}
	w, err := NewPalettedGIFWriter(path, pal, loopCount)
	if err != nil {
		return err
	}
	for i, frame := range frames {
		if err := w.WriteFrame(frame, delays[i]); err != nil {
			return fmt.Errorf("videoio: WriteGIFDelays frame %d: %w", i, err)
		}
	}
	return w.Release()
}

// AdaptivePalette builds a palette of at most maxColors entries tailored to the
// given frames, using popularity quantization: every pixel is binned into a
// coarse RGB cube (4 bits per channel), the most populous bins are chosen, and
// each bin contributes its centre colour. The result is deterministic — the
// same frames and maxColors always yield the same palette — and is a genuine
// improvement over a fixed palette for footage with a limited colour range.
// maxColors is clamped to [1, 256]. A transparent slot is not reserved.
func AdaptivePalette(frames []*cv.Mat, maxColors int) color.Palette {
	if maxColors < 1 {
		maxColors = 1
	}
	if maxColors > 256 {
		maxColors = 256
	}
	counts := make(map[uint16]int)
	for _, m := range frames {
		if m == nil || m.Empty() {
			continue
		}
		img := m.ToImage()
		b := img.Bounds()
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				r, g, bl, _ := img.At(x, y).RGBA()
				key := uint16(r>>12)<<8 | uint16(g>>12)<<4 | uint16(bl>>12)
				counts[key]++
			}
		}
	}
	if len(counts) == 0 {
		return PaletteWebSafe
	}

	type bin struct {
		key   uint16
		count int
	}
	bins := make([]bin, 0, len(counts))
	for k, c := range counts {
		bins = append(bins, bin{key: k, count: c})
	}
	// Sort by descending popularity, breaking ties by key for determinism.
	sort.Slice(bins, func(i, j int) bool {
		if bins[i].count != bins[j].count {
			return bins[i].count > bins[j].count
		}
		return bins[i].key < bins[j].key
	})
	if len(bins) > maxColors {
		bins = bins[:maxColors]
	}
	pal := make(color.Palette, 0, len(bins))
	for _, bn := range bins {
		r := uint8((bn.key>>8)&0xf) * 17 // scale 4-bit level (0..15) to 0..255
		g := uint8((bn.key>>4)&0xf) * 17
		bl := uint8(bn.key&0xf) * 17
		pal = append(pal, color.RGBA{R: r, G: g, B: bl, A: 255})
	}
	return pal
}
