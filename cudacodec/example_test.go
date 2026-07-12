package cudacodec_test

import (
	"fmt"
	"image"
	"os"
	"path/filepath"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudacodec"
)

// gradient builds one small RGB frame whose red channel encodes the frame index,
// so a decoded sequence can be told apart frame by frame.
func gradient(idx, w, h int) *cv.Mat {
	m := cv.NewMat(h, w, 3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.Set(y, x, 0, uint8(idx*20))
			m.Set(y, x, 1, uint8(x*8))
			m.Set(y, x, 2, uint8(y*8))
		}
	}
	return m
}

// Example encodes a short synthetic clip with a VideoWriter and reads it back
// with a VideoReader, printing the frame count that survived the round trip.
func Example() {
	path := filepath.Join(os.TempDir(), "cudacodec_example_clip.avi")

	w, err := cudacodec.CreateVideoWriter(path, image.Pt(24, 16),
		cudacodec.CodecH264, 25, cudacodec.ColorFormatBGR)
	if err != nil {
		fmt.Println("writer:", err)
		return
	}
	for i := 0; i < 4; i++ {
		if err := w.Write(cudacodec.NewGpuMatFromMat(gradient(i, 24, 16))); err != nil {
			fmt.Println("write:", err)
			return
		}
	}
	if err := w.Release(); err != nil {
		fmt.Println("release:", err)
		return
	}

	r, err := cudacodec.CreateVideoReader(path)
	if err != nil {
		fmt.Println("reader:", err)
		return
	}
	defer r.Release()

	count := 0
	dst := cudacodec.NewGpuMat()
	for r.NextFrame(dst) {
		count++
	}
	fmt.Printf("decoded %d frames, requested codec substituted as %v\n",
		count, r.Format().Codec)
	// Output: decoded 4 frames, requested codec substituted as JPEG
}

// ExampleVideoReader_Format shows the FormatInfo a reader reports for a decoded
// clip.
func ExampleVideoReader_Format() {
	path := filepath.Join(os.TempDir(), "cudacodec_fmt_clip.apng")

	frames := []*cv.Mat{gradient(0, 8, 8), gradient(1, 8, 8)}
	if err := func() error {
		w, err := cudacodec.CreateVideoWriter(path, image.Pt(8, 8),
			cudacodec.CodecHEVC, 10, cudacodec.ColorFormatRGB)
		if err != nil {
			return err
		}
		for _, f := range frames {
			if err := w.Write(cudacodec.NewGpuMatFromMat(f)); err != nil {
				return err
			}
		}
		return w.Release()
	}(); err != nil {
		fmt.Println(err)
		return
	}

	r, err := cudacodec.CreateVideoReader(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer r.Release()

	info := r.Format()
	fmt.Printf("%dx%d, %d frames, valid=%v, codec=%v\n",
		info.Width, info.Height, info.NumFrames, info.Valid, info.Codec)
	// Output: 8x8, 2 frames, valid=true, codec=Uncompressed_RGBA
}

// ExampleNewGpuMat demonstrates the host-side GpuMat wrapper: uploading a Mat
// and downloading it back.
func ExampleNewGpuMat() {
	src := gradient(2, 4, 4)

	g := cudacodec.NewGpuMat()
	g.Upload(src)
	fmt.Printf("empty=%v size=%v\n", g.Empty(), g.Size())

	out := g.Download()
	fmt.Printf("downloaded %dx%d, first pixel R=%d\n", out.Cols, out.Rows, out.At(0, 0, 0))
	// Output:
	// empty=false size=(4,4)
	// downloaded 4x4, first pixel R=40
}
