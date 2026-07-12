package videoio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"os"

	cv "github.com/malcolmston/opencv"
)

// pngSignature is the fixed eight-byte magic that opens every PNG and APNG file.
var pngSignature = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

// APNG dispose and blend operation codes, as defined by the APNG specification.
const (
	apngDisposeNone     = 0
	apngDisposeBkgd     = 1
	apngDisposePrevious = 2
	apngBlendSource     = 0
	apngBlendOver       = 1
)

// APNGWriter accumulates frames and encodes them as a single Animated PNG when
// released. Unlike GIF, APNG is a true-colour format, so frames are written
// without palette quantization and survive the round trip pixel-for-pixel. The
// canvas size is fixed by the first frame; later frames are composited at the
// origin and clipped to those bounds. The zero value is not usable — obtain one
// from [NewAPNGWriter].
type APNGWriter struct {
	path        string
	delayCentis int
	bounds      image.Rectangle
	frames      []*image.RGBA
	delays      []int
	loopCount   int
	released    bool
}

// NewAPNGWriter creates a writer that encodes every frame passed to
// [APNGWriter.Write] into an APNG file at path when [APNGWriter.Release] runs.
// delayCentis is each frame's display duration in centiseconds; a non-positive
// value is clamped to 0. The animation loops forever; use
// [APNGWriter.SetLoopCount] to bound it. No file is created until Release.
func NewAPNGWriter(path string, delayCentis int) (*APNGWriter, error) {
	if path == "" {
		return nil, fmt.Errorf("videoio: NewAPNGWriter: empty path")
	}
	if delayCentis < 0 {
		delayCentis = 0
	}
	return &APNGWriter{path: path, delayCentis: delayCentis}, nil
}

// SetLoopCount limits how many times viewers replay the animation; 0 (the
// default) means loop forever. It must be called before [APNGWriter.Release].
func (w *APNGWriter) SetLoopCount(n int) {
	if n < 0 {
		n = 0
	}
	w.loopCount = n
}

// Write appends frame to the animation, using the writer's default per-frame
// delay. The first frame fixes the canvas size; later frames are drawn at the
// origin and clipped. It errors on an empty frame or after Release.
func (w *APNGWriter) Write(frame *cv.Mat) error {
	return w.WriteFrame(frame, w.delayCentis)
}

// WriteFrame appends frame with an explicit display duration of delayCentis
// centiseconds, overriding the writer's default for this frame only. This is
// how variable-rate animations are built. It errors on an empty frame or after
// Release.
func (w *APNGWriter) WriteFrame(frame *cv.Mat, delayCentis int) error {
	if w == nil {
		return fmt.Errorf("videoio: WriteFrame on nil writer")
	}
	if w.released {
		return fmt.Errorf("videoio: WriteFrame after Release")
	}
	if frame.Empty() {
		return fmt.Errorf("videoio: WriteFrame: empty frame")
	}
	if delayCentis < 0 {
		delayCentis = 0
	}
	if len(w.frames) == 0 {
		w.bounds = image.Rect(0, 0, frame.Cols, frame.Rows)
	}
	canvas := image.NewRGBA(w.bounds)
	// Fill an opaque background first so every frame is fully opaque regardless
	// of size. This keeps the PNG colour type identical across frames, which is
	// required because they all share one IHDR in the APNG stream.
	draw.Draw(canvas, canvas.Bounds(), image.NewUniform(color.RGBA{A: 255}), image.Point{}, draw.Src)
	src := frame.ToImage()
	draw.Draw(canvas, w.bounds.Intersect(src.Bounds()), src, src.Bounds().Min, draw.Src)
	w.frames = append(w.frames, canvas)
	w.delays = append(w.delays, delayCentis)
	return nil
}

// Release encodes every written frame into the destination APNG and finalizes
// the writer. Each frame is stored full-size with dispose "none" and blend
// "source", so it fully replaces the previous one. Calling Release twice, or on
// a writer with no frames, returns an error.
func (w *APNGWriter) Release() error {
	if w == nil {
		return fmt.Errorf("videoio: Release on nil APNG writer")
	}
	if w.released {
		return fmt.Errorf("videoio: APNG Release called twice")
	}
	if len(w.frames) == 0 {
		return fmt.Errorf("videoio: APNG Release: no frames written")
	}
	w.released = true

	data, err := encodeAPNG(w.frames, w.delays, w.loopCount)
	if err != nil {
		return fmt.Errorf("videoio: APNG Release encode %q: %w", w.path, err)
	}
	if err := os.WriteFile(w.path, data, 0o644); err != nil {
		return fmt.Errorf("videoio: APNG Release write %q: %w", w.path, err)
	}
	return nil
}

// WriteAPNG encodes frames into an APNG at path, giving every frame the same
// delay of delayCentis centiseconds. It is a convenience wrapper around
// [NewAPNGWriter]. The canvas is taken from the first frame; later frames are
// placed at the origin and clipped. It errors if frames is empty.
func WriteAPNG(path string, frames []*cv.Mat, delayCentis int) error {
	return WriteAPNGDelays(path, frames, uniformDelays(len(frames), delayCentis))
}

// WriteAPNGDelays encodes frames into an APNG at path with an independent delay
// (in centiseconds) for each frame, enabling variable-rate playback. len(delays)
// must equal len(frames). It errors if frames is empty or the lengths differ.
func WriteAPNGDelays(path string, frames []*cv.Mat, delays []int) error {
	if len(frames) == 0 {
		return fmt.Errorf("videoio: WriteAPNGDelays: no frames")
	}
	if len(delays) != len(frames) {
		return fmt.Errorf("videoio: WriteAPNGDelays: %d delays for %d frames", len(delays), len(frames))
	}
	w, err := NewAPNGWriter(path, 0)
	if err != nil {
		return err
	}
	for i, frame := range frames {
		if err := w.WriteFrame(frame, delays[i]); err != nil {
			return fmt.Errorf("videoio: WriteAPNGDelays frame %d: %w", i, err)
		}
	}
	return w.Release()
}

// ReadAPNG decodes the APNG (or plain PNG) at path and returns every frame as a
// three-channel RGB Mat together with the matching per-frame delays in
// centiseconds. Frame offsets and the APNG dispose/blend operations are honoured
// by compositing onto a full-size canvas, so the returned Mats are complete
// canvas-sized frames. A plain, non-animated PNG decodes as a single frame.
func ReadAPNG(path string) ([]*cv.Mat, []int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("videoio: ReadAPNG read %q: %w", path, err)
	}
	frames, delays, err := decodeAPNG(raw)
	if err != nil {
		return nil, nil, fmt.Errorf("videoio: ReadAPNG %q: %w", path, err)
	}
	return frames, delays, nil
}

// OpenAPNG decodes the APNG at path and returns a [VideoCapture] over its
// frames, so an animated PNG can be streamed with the same grab/retrieve API as
// any other source.
func OpenAPNG(path string) (*VideoCapture, error) {
	frames, delays, err := ReadAPNG(path)
	if err != nil {
		return nil, err
	}
	return newCapture(frames, delays), nil
}

// pngChunk is a decoded PNG chunk: its four-character type and its raw data
// (the CRC has already been consumed and is not retained).
type pngChunk struct {
	typ  string
	data []byte
}

// parsePNGChunks splits a PNG/APNG byte stream into its ordered chunks. It stops
// after IEND and validates only the signature and chunk framing, not CRCs.
func parsePNGChunks(b []byte) ([]pngChunk, error) {
	if len(b) < 8 || !bytes.Equal(b[:8], pngSignature) {
		return nil, fmt.Errorf("not a PNG stream")
	}
	var chunks []pngChunk
	off := 8
	for off+8 <= len(b) {
		length := int(binary.BigEndian.Uint32(b[off:]))
		typ := string(b[off+4 : off+8])
		off += 8
		if length < 0 || off+length+4 > len(b) {
			return nil, fmt.Errorf("truncated %q chunk", typ)
		}
		chunks = append(chunks, pngChunk{typ: typ, data: b[off : off+length]})
		off += length + 4 // data + CRC
		if typ == "IEND" {
			break
		}
	}
	return chunks, nil
}

// writeChunk emits one PNG chunk (length, type, data, CRC) to w.
func writeChunk(w io.Writer, typ string, data []byte) error {
	var hdr [8]byte
	binary.BigEndian.PutUint32(hdr[:4], uint32(len(data)))
	copy(hdr[4:], typ)
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := w.Write(data); err != nil {
		return err
	}
	crc := crc32.NewIEEE()
	crc.Write([]byte(typ))
	crc.Write(data)
	var c [4]byte
	binary.BigEndian.PutUint32(c[:], crc.Sum32())
	_, err := w.Write(c[:])
	return err
}

// encodeAPNG builds a complete APNG byte stream from full-canvas RGBA frames.
// Each frame is independently PNG-encoded and its IDAT payload reused verbatim:
// the first frame's payload becomes the file's IDAT, later frames become fdAT
// chunks. Every frame is tagged dispose "none" / blend "source", so the result
// is valid whether or not a viewer understands APNG (non-APNG viewers show the
// first frame).
func encodeAPNG(frames []*image.RGBA, delays []int, loopCount int) ([]byte, error) {
	if len(frames) == 0 {
		return nil, fmt.Errorf("no frames")
	}
	bounds := frames[0].Bounds()
	w32 := uint32(bounds.Dx())
	h32 := uint32(bounds.Dy())

	// Pre-encode each frame and pull out its concatenated IDAT payload.
	var ihdr []byte
	idats := make([][]byte, len(frames))
	for i, fr := range frames {
		var buf bytes.Buffer
		if err := png.Encode(&buf, fr); err != nil {
			return nil, fmt.Errorf("frame %d: %w", i, err)
		}
		chunks, err := parsePNGChunks(buf.Bytes())
		if err != nil {
			return nil, fmt.Errorf("frame %d: %w", i, err)
		}
		var payload []byte
		for _, ch := range chunks {
			switch ch.typ {
			case "IHDR":
				if i == 0 {
					ihdr = append([]byte(nil), ch.data...)
				}
			case "IDAT":
				payload = append(payload, ch.data...)
			}
		}
		if payload == nil {
			return nil, fmt.Errorf("frame %d: no IDAT produced", i)
		}
		idats[i] = payload
	}

	var out bytes.Buffer
	out.Write(pngSignature)
	if err := writeChunk(&out, "IHDR", ihdr); err != nil {
		return nil, err
	}

	// acTL: number of frames and number of plays (0 == infinite).
	actl := make([]byte, 8)
	binary.BigEndian.PutUint32(actl[0:], uint32(len(frames)))
	binary.BigEndian.PutUint32(actl[4:], uint32(loopCount))
	if err := writeChunk(&out, "acTL", actl); err != nil {
		return nil, err
	}

	var seq uint32
	for i := range frames {
		num, den := delayFraction(delays[i])
		fctl := make([]byte, 26)
		binary.BigEndian.PutUint32(fctl[0:], seq)
		binary.BigEndian.PutUint32(fctl[4:], w32)
		binary.BigEndian.PutUint32(fctl[8:], h32)
		// x_offset and y_offset stay 0: every frame is full-canvas.
		binary.BigEndian.PutUint16(fctl[20:], num)
		binary.BigEndian.PutUint16(fctl[22:], den)
		fctl[24] = apngDisposeNone
		fctl[25] = apngBlendSource
		if err := writeChunk(&out, "fcTL", fctl); err != nil {
			return nil, err
		}
		seq++

		if i == 0 {
			if err := writeChunk(&out, "IDAT", idats[0]); err != nil {
				return nil, err
			}
			continue
		}
		fdat := make([]byte, 4+len(idats[i]))
		binary.BigEndian.PutUint32(fdat[0:], seq)
		copy(fdat[4:], idats[i])
		if err := writeChunk(&out, "fdAT", fdat); err != nil {
			return nil, err
		}
		seq++
	}

	if err := writeChunk(&out, "IEND", nil); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// fcTLMeta holds the geometry, timing and compositing operations of one APNG
// frame, parsed from its fcTL chunk.
type fcTLMeta struct {
	width, height  int
	xOff, yOff     int
	delayNum       uint16
	delayDen       uint16
	dispose, blend byte
}

// decodeAPNG reconstructs every animation frame from an APNG (or a single frame
// from a plain PNG), compositing onto a canvas so the results are full-size RGB
// Mats, and returns the matching delays in centiseconds.
func decodeAPNG(raw []byte) ([]*cv.Mat, []int, error) {
	chunks, err := parsePNGChunks(raw)
	if err != nil {
		return nil, nil, err
	}

	var ihdr []byte
	var metas []fcTLMeta
	var payloads [][]byte
	cur := -1

	for _, ch := range chunks {
		switch ch.typ {
		case "IHDR":
			ihdr = ch.data
		case "fcTL":
			if len(ch.data) < 26 {
				return nil, nil, fmt.Errorf("short fcTL")
			}
			metas = append(metas, parseFcTL(ch.data))
			payloads = append(payloads, nil)
			cur = len(payloads) - 1
		case "IDAT":
			if cur < 0 {
				// Default image with no controlling fcTL: not part of the
				// animation. Start a synthetic frame so a plain PNG still decodes.
				metas = append(metas, fcTLMeta{width: canvasW(ihdr), height: canvasH(ihdr), delayNum: 0, delayDen: 100})
				payloads = append(payloads, nil)
				cur = 0
			}
			payloads[cur] = append(payloads[cur], ch.data...)
		case "fdAT":
			if cur < 0 || len(ch.data) < 4 {
				return nil, nil, fmt.Errorf("stray fdAT")
			}
			payloads[cur] = append(payloads[cur], ch.data[4:]...)
		}
	}
	if len(ihdr) < 13 {
		return nil, nil, fmt.Errorf("missing IHDR")
	}
	if len(payloads) == 0 {
		return nil, nil, fmt.Errorf("no image data")
	}

	cw, chh := canvasW(ihdr), canvasH(ihdr)
	canvas := image.NewRGBA(image.Rect(0, 0, cw, chh))
	frames := make([]*cv.Mat, 0, len(payloads))
	delays := make([]int, 0, len(payloads))
	var saved *image.RGBA

	for i, payload := range payloads {
		m := metas[i]
		sub, err := decodeSubImage(ihdr, m, payload)
		if err != nil {
			return nil, nil, fmt.Errorf("frame %d: %w", i, err)
		}
		if m.dispose == apngDisposePrevious {
			saved = cloneRGBA(canvas)
		}
		region := image.Rect(m.xOff, m.yOff, m.xOff+m.width, m.yOff+m.height)
		op := draw.Src
		if m.blend == apngBlendOver {
			op = draw.Over
		}
		draw.Draw(canvas, region, sub, sub.Bounds().Min, op)

		frames = append(frames, cv.FromImage(canvas))
		delays = append(delays, centisFromFraction(m.delayNum, m.delayDen))

		switch m.dispose {
		case apngDisposeBkgd:
			draw.Draw(canvas, region, image.Transparent, image.Point{}, draw.Src)
		case apngDisposePrevious:
			if saved != nil {
				draw.Draw(canvas, canvas.Bounds(), saved, canvas.Bounds().Min, draw.Src)
			}
		}
	}
	return frames, delays, nil
}

// parseFcTL decodes the fixed-layout fcTL chunk body into an fcTLMeta.
func parseFcTL(d []byte) fcTLMeta {
	return fcTLMeta{
		width:    int(binary.BigEndian.Uint32(d[4:])),
		height:   int(binary.BigEndian.Uint32(d[8:])),
		xOff:     int(binary.BigEndian.Uint32(d[12:])),
		yOff:     int(binary.BigEndian.Uint32(d[16:])),
		delayNum: binary.BigEndian.Uint16(d[20:]),
		delayDen: binary.BigEndian.Uint16(d[22:]),
		dispose:  d[24],
		blend:    d[25],
	}
}

// decodeSubImage rebuilds a standalone PNG for one frame — the shared IHDR with
// its dimensions patched to the frame's, the frame's compressed payload as a
// single IDAT, and IEND — then decodes it with the standard library.
func decodeSubImage(ihdr []byte, m fcTLMeta, payload []byte) (image.Image, error) {
	hdr := append([]byte(nil), ihdr...)
	binary.BigEndian.PutUint32(hdr[0:], uint32(m.width))
	binary.BigEndian.PutUint32(hdr[4:], uint32(m.height))

	var buf bytes.Buffer
	buf.Write(pngSignature)
	if err := writeChunk(&buf, "IHDR", hdr); err != nil {
		return nil, err
	}
	if err := writeChunk(&buf, "IDAT", payload); err != nil {
		return nil, err
	}
	if err := writeChunk(&buf, "IEND", nil); err != nil {
		return nil, err
	}
	return png.Decode(&buf)
}

// canvasW and canvasH read the image dimensions from an IHDR chunk body.
func canvasW(ihdr []byte) int {
	if len(ihdr) < 4 {
		return 0
	}
	return int(binary.BigEndian.Uint32(ihdr[0:]))
}

func canvasH(ihdr []byte) int {
	if len(ihdr) < 8 {
		return 0
	}
	return int(binary.BigEndian.Uint32(ihdr[4:]))
}

// delayFraction expresses a centisecond delay as the num/den fraction APNG
// stores, using a denominator of 100 so num is simply the centisecond count.
func delayFraction(centis int) (num, den uint16) {
	if centis < 0 {
		centis = 0
	}
	return uint16(centis), 100
}

// centisFromFraction converts an APNG num/den delay (seconds) into
// centiseconds. A zero denominator is treated as 100 per the specification.
func centisFromFraction(num, den uint16) int {
	if den == 0 {
		den = 100
	}
	return int(float64(num)*100/float64(den) + 0.5)
}
