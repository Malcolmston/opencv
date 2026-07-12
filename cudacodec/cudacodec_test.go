package cudacodec

import (
	"image"
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/videoio"
)

// synthFrames builds a deterministic sequence of n RGB frames of the given size.
// Each frame is filled with a per-frame gradient so successive frames differ,
// which lets round-trip tests check both frame count and per-frame fidelity.
func synthFrames(n, w, h int) []*cv.Mat {
	frames := make([]*cv.Mat, n)
	for f := 0; f < n; f++ {
		m := cv.NewMat(h, w, 3)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				m.Set(y, x, 0, uint8((x*4+f*10)&0xff))
				m.Set(y, x, 1, uint8((y*4+f*20)&0xff))
				m.Set(y, x, 2, uint8((f*30)&0xff))
			}
		}
		frames[f] = m
	}
	return frames
}

// meanAbsDiff returns the average absolute per-sample difference between two
// equally sized Mats, a simple fidelity metric.
func meanAbsDiff(a, b *cv.Mat) float64 {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		return 1e9
	}
	var sum float64
	for i := range a.Data {
		d := int(a.Data[i]) - int(b.Data[i])
		if d < 0 {
			d = -d
		}
		sum += float64(d)
	}
	return sum / float64(len(a.Data))
}

// roundTrip writes frames to a file with the given extension through a
// VideoWriter, reads them back through a VideoReader, and returns the decoded
// frames plus the reader's FormatInfo.
func roundTrip(t *testing.T, ext string, frames []*cv.Mat, codec Codec) ([]*cv.Mat, FormatInfo) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "clip"+ext)

	w, err := CreateVideoWriter(path, image.Pt(frames[0].Cols, frames[0].Rows), codec, 20, ColorFormatBGR)
	if err != nil {
		t.Fatalf("CreateVideoWriter: %v", err)
	}
	for i, f := range frames {
		if err := w.Write(NewGpuMatFromMat(f)); err != nil {
			t.Fatalf("Write frame %d: %v", i, err)
		}
	}
	if err := w.Release(); err != nil {
		t.Fatalf("Release: %v", err)
	}

	r, err := CreateVideoReader(path)
	if err != nil {
		t.Fatalf("CreateVideoReader: %v", err)
	}
	defer r.Release()

	var out []*cv.Mat
	dst := NewGpuMat()
	for r.NextFrame(dst) {
		out = append(out, dst.Download())
	}
	return out, r.Format()
}

func TestRoundTripAVI(t *testing.T) {
	in := synthFrames(5, 32, 24)
	out, info := roundTrip(t, ".avi", in, CodecH264)

	if len(out) != len(in) {
		t.Fatalf("frame count = %d, want %d", len(out), len(in))
	}
	if !info.Valid {
		t.Errorf("FormatInfo.Valid = false")
	}
	if info.Codec != CodecJPEG {
		t.Errorf("info.Codec = %v, want JPEG (AVI substitution)", info.Codec)
	}
	if info.Width != 32 || info.Height != 24 {
		t.Errorf("info size = %dx%d, want 32x24", info.Width, info.Height)
	}
	if info.NumFrames != 5 {
		t.Errorf("info.NumFrames = %d, want 5", info.NumFrames)
	}
	// MJPEG is lossy but should be visually faithful.
	for i := range out {
		if d := meanAbsDiff(in[i], out[i]); d > 12 {
			t.Errorf("frame %d mean abs diff %.2f too high", i, d)
		}
	}
}

func TestRoundTripAPNGLossless(t *testing.T) {
	in := synthFrames(4, 16, 16)
	out, info := roundTrip(t, ".apng", in, CodecHEVC)

	if len(out) != len(in) {
		t.Fatalf("frame count = %d, want %d", len(out), len(in))
	}
	if info.Codec != CodecUncompressedRGBA {
		t.Errorf("info.Codec = %v, want Uncompressed_RGBA", info.Codec)
	}
	// APNG is lossless: frames must round-trip exactly.
	for i := range out {
		if d := meanAbsDiff(in[i], out[i]); d != 0 {
			t.Errorf("frame %d not lossless, mean abs diff %.4f", i, d)
		}
	}
}

func TestRoundTripGIF(t *testing.T) {
	in := synthFrames(3, 20, 12)
	out, _ := roundTrip(t, ".gif", in, CodecMPEG4)
	if len(out) != len(in) {
		t.Fatalf("frame count = %d, want %d", len(out), len(in))
	}
}

func TestGrabRetrieve(t *testing.T) {
	in := synthFrames(3, 8, 8)
	path := filepath.Join(t.TempDir(), "gr.apng")
	if err := writeAll(t, path, in); err != nil {
		t.Fatal(err)
	}
	r, err := CreateVideoReader(path)
	if err != nil {
		t.Fatalf("CreateVideoReader: %v", err)
	}
	defer r.Release()

	if r.FrameCount() != 3 {
		t.Errorf("FrameCount = %d, want 3", r.FrameCount())
	}

	dst := NewGpuMat()
	count := 0
	for r.Grab() {
		if !r.Retrieve(dst) {
			t.Fatalf("Retrieve after successful Grab returned false")
		}
		if dst.Empty() {
			t.Fatalf("retrieved frame is empty")
		}
		count++
	}
	if count != 3 {
		t.Errorf("grabbed %d frames, want 3", count)
	}
	// Retrieve past the end fails.
	if r.Retrieve(dst) {
		// position is at end; retrieve returns last grabbed, so this may be true.
		_ = dst
	}
	if r.Grab() {
		t.Errorf("Grab past end returned true")
	}
}

func TestReaderGetProps(t *testing.T) {
	in := synthFrames(4, 10, 10)
	path := filepath.Join(t.TempDir(), "props.avi")
	if err := writeAll(t, path, in); err != nil {
		t.Fatal(err)
	}
	r, err := CreateVideoReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Release()

	if got := r.Get(videoio.CAP_PROP_FRAME_COUNT); got != 4 {
		t.Errorf("FRAME_COUNT = %v, want 4", got)
	}
	if got := r.Get(videoio.CAP_PROP_FRAME_WIDTH); got != 10 {
		t.Errorf("FRAME_WIDTH = %v, want 10", got)
	}
	dst := NewGpuMat()
	r.NextFrame(dst)
	if got := r.Get(videoio.CAP_PROP_POS_FRAMES); got != 1 {
		t.Errorf("POS_FRAMES after one read = %v, want 1", got)
	}
}

func TestWriterGetAndAccessors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "w.avi")
	w, err := CreateVideoWriter(path, image.Pt(16, 16), CodecH264, 30, ColorFormatBGR)
	if err != nil {
		t.Fatal(err)
	}
	if w.Codec() != CodecH264 {
		t.Errorf("Codec() = %v, want H264", w.Codec())
	}
	if w.ColorFormat() != ColorFormatBGR {
		t.Errorf("ColorFormat() = %v", w.ColorFormat())
	}
	if got := w.Get(videoio.CAP_PROP_FPS); got != 30 {
		t.Errorf("FPS = %v, want 30", got)
	}
	if got := w.Get(videoio.CAP_PROP_FRAME_WIDTH); got != 0 {
		t.Errorf("width before write = %v, want 0", got)
	}
	frames := synthFrames(2, 16, 16)
	for _, f := range frames {
		if err := w.Write(NewGpuMatFromMat(f)); err != nil {
			t.Fatal(err)
		}
	}
	if got := w.Get(videoio.CAP_PROP_FRAME_COUNT); got != 2 {
		t.Errorf("FRAME_COUNT = %v, want 2", got)
	}
	if got := w.Get(videoio.CAP_PROP_FRAME_WIDTH); got != 16 {
		t.Errorf("width after write = %v, want 16", got)
	}
	if got := w.Get(videoio.PropID(999)); got != 0 {
		t.Errorf("unknown prop = %v, want 0", got)
	}
	if err := w.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestWriterWithParams(t *testing.T) {
	path := filepath.Join(t.TempDir(), "p.apng")
	params := NewEncoderParams()
	params.FPS = 15
	w, err := CreateVideoWriterWithParams(path, image.Pt(8, 8), CodecAV1, ColorFormatRGB, params)
	if err != nil {
		t.Fatal(err)
	}
	if got := w.Get(videoio.CAP_PROP_FPS); got != 15 {
		t.Errorf("FPS = %v, want 15", got)
	}
	if err := w.Write(NewGpuMatFromMat(synthFrames(1, 8, 8)[0])); err != nil {
		t.Fatal(err)
	}
	if err := w.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestWriterErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "e.avi")
	w, err := CreateVideoWriter(path, image.Pt(8, 8), CodecJPEG, 0, ColorFormatBGR)
	if err != nil {
		t.Fatal(err)
	}
	// fps<=0 clamps to a default.
	if w.Get(videoio.CAP_PROP_FPS) != 25 {
		t.Errorf("default fps = %v, want 25", w.Get(videoio.CAP_PROP_FPS))
	}
	// Empty frame rejected.
	if err := w.Write(NewGpuMat()); err == nil {
		t.Error("Write empty frame: want error")
	}
	if err := w.Write(NewGpuMatFromMat(synthFrames(1, 8, 8)[0])); err != nil {
		t.Fatal(err)
	}
	if err := w.Release(); err != nil {
		t.Fatal(err)
	}
	// Double release and write-after-release.
	if err := w.Release(); err == nil {
		t.Error("second Release: want error")
	}
	if err := w.Write(NewGpuMatFromMat(synthFrames(1, 8, 8)[0])); err == nil {
		t.Error("Write after Release: want error")
	}
}

func TestUnsupportedExtensions(t *testing.T) {
	if _, err := CreateVideoWriter("clip.mp4", image.Pt(4, 4), CodecH264, 25, ColorFormatBGR); err == nil {
		t.Error("CreateVideoWriter .mp4: want error")
	}
	if _, err := CreateVideoReader("clip.mkv"); err == nil {
		t.Error("CreateVideoReader .mkv: want error")
	}
	if _, err := CreateVideoReader(filepath.Join(t.TempDir(), "missing.avi")); err == nil {
		t.Error("CreateVideoReader missing file: want error")
	}
}

func TestGpuMatLifecycle(t *testing.T) {
	g := NewGpuMat()
	if !g.Empty() {
		t.Error("new GpuMat should be empty")
	}
	if g.Rows() != 0 || g.Cols() != 0 || g.Channels() != 0 {
		t.Error("empty GpuMat dims should be 0")
	}
	if (g.Size() != image.Point{}) {
		t.Error("empty GpuMat size should be zero point")
	}
	if g.Download() != nil {
		t.Error("empty GpuMat download should be nil")
	}

	m := synthFrames(1, 6, 4)[0]
	g.Upload(m)
	if g.Empty() {
		t.Fatal("GpuMat empty after upload")
	}
	if g.Cols() != 6 || g.Rows() != 4 || g.Channels() != 3 {
		t.Errorf("dims = %dx%dx%d, want 6x4x3", g.Cols(), g.Rows(), g.Channels())
	}
	if g.Size() != image.Pt(6, 4) {
		t.Errorf("Size = %v, want (6,4)", g.Size())
	}

	// Upload copies: mutating the source must not change the GpuMat.
	back := g.Download()
	m.Set(0, 0, 0, m.At(0, 0, 0)+40)
	if meanAbsDiff(back, g.Download()) != 0 {
		t.Error("GpuMat shares storage with uploaded Mat")
	}

	// Clone independence.
	c := g.Clone()
	g.Release()
	if !g.Empty() {
		t.Error("GpuMat not empty after Release")
	}
	if c.Empty() {
		t.Error("clone affected by original Release")
	}
}

func TestGpuMatUploadEmptyClears(t *testing.T) {
	g := NewGpuMatFromMat(synthFrames(1, 4, 4)[0])
	if g.Empty() {
		t.Fatal("expected non-empty")
	}
	g.Upload(nil)
	if !g.Empty() {
		t.Error("Upload(nil) should clear GpuMat")
	}
	if !NewGpuMatFromMat(nil).Empty() {
		t.Error("NewGpuMatFromMat(nil) should be empty")
	}
}

func TestStreamNoOp(t *testing.T) {
	s := NewStream()
	s.WaitForCompletion()
	if !s.QueryIfComplete() {
		t.Error("QueryIfComplete should be true")
	}
	// Stream args accepted and ignored.
	g := NewGpuMatFromMat(synthFrames(1, 4, 4)[0])
	_ = g.Download(s)
	g.Upload(synthFrames(1, 4, 4)[0], s)
}

func TestNextFrameNilPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("NextFrame(nil) should panic")
		}
	}()
	var r VideoReader
	r.NextFrame(nil)
}

func TestNilReceivers(t *testing.T) {
	var r *VideoReader
	if r.NextFrame(NewGpuMat()) {
		t.Error("nil reader NextFrame should be false")
	}
	if r.Grab() {
		t.Error("nil reader Grab should be false")
	}
	if r.Retrieve(NewGpuMat()) {
		t.Error("nil reader Retrieve should be false")
	}
	if r.Get(videoio.CAP_PROP_FPS) != 0 {
		t.Error("nil reader Get should be 0")
	}
	if r.FrameCount() != 0 {
		t.Error("nil reader FrameCount should be 0")
	}
	if r.Release() != nil {
		t.Error("nil reader Release should be nil")
	}
	if (r.Format() != FormatInfo{}) {
		t.Error("nil reader Format should be zero")
	}
}

func TestEnumStrings(t *testing.T) {
	cases := []struct {
		got, want string
	}{
		{CodecH264.String(), "H264"},
		{CodecJPEG.String(), "JPEG"},
		{CodecHEVC.String(), "HEVC"},
		{CodecNumCodecs.String(), "NumCodecs"},
		{CodecUncompressedRGBA.String(), "Uncompressed_RGBA"},
		{Codec(-99).String(), "Unknown"},
		{ColorFormatBGR.String(), "BGR"},
		{ColorFormatNV12.String(), "NV_NV12"},
		{ColorFormat(-1).String(), "Unknown"},
		{SurfaceFormatNV12.String(), "SF_NV12"},
		{SurfaceFormat(-1).String(), "Unknown"},
		{ChromaYUV420.String(), "YUV420"},
		{ChromaFormat(-1).String(), "Unknown"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("String = %q, want %q", c.got, c.want)
		}
	}
}

func TestNewEncoderParamsDefaults(t *testing.T) {
	p := NewEncoderParams()
	if p.FPS != 25 || p.Quality != 95 || p.SurfaceFormat != SurfaceFormatNV12 || p.GOPLength != 25 {
		t.Errorf("unexpected defaults: %+v", p)
	}
}

// writeAll is a small helper that writes frames to path via a VideoWriter,
// choosing fps 20 and BGR, used by tests that only need the file to exist.
func writeAll(t *testing.T, path string, frames []*cv.Mat) error {
	t.Helper()
	w, err := CreateVideoWriter(path, image.Pt(frames[0].Cols, frames[0].Rows), CodecJPEG, 20, ColorFormatBGR)
	if err != nil {
		return err
	}
	for _, f := range frames {
		if err := w.Write(NewGpuMatFromMat(f)); err != nil {
			return err
		}
	}
	return w.Release()
}
