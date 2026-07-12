// Command examples demonstrates the cv package end-to-end: it synthesises a
// small colour image, converts it to grayscale, blurs it, runs Canny edge
// detection, annotates a copy with a rectangle and label, and writes each
// intermediate result to a PNG file in the working directory.
//
// Run it with:
//
//	go run ./examples
//
// It writes example_source.png, example_gray.png, example_blur.png,
// example_canny.png, example_annotated.png, example_contours.png and
// example_warp.png.
package main

import (
	"fmt"
	"log"

	cv "github.com/malcolmston/opencv"
)

func main() {
	const w, h = 200, 150

	// 1. Synthesise a colour image: a dark background with a bright square and
	// a filled circle, so there is real structure for the edge detector.
	src := cv.NewMat(h, w, 3)
	src.SetTo(30)
	cv.Rectangle(src, cv.Point{X: 40, Y: 30}, cv.Point{X: 120, Y: 110},
		cv.NewScalar(220, 200, 40), cv.Filled)
	cv.Circle(src, cv.Point{X: 140, Y: 90}, 28,
		cv.NewScalar(40, 120, 230), cv.Filled)
	write("example_source.png", src)

	// 2. Convert to grayscale.
	gray := cv.CvtColor(src, cv.ColorRGB2Gray)
	write("example_gray.png", gray)

	// 3. Gaussian blur to reduce noise before edge detection.
	blur := cv.GaussianBlur(gray, 5, 1.4)
	write("example_blur.png", blur)

	// 4. Canny edge detection.
	edges := cv.Canny(blur, 50, 150)
	write("example_canny.png", edges)

	// 5. Annotate a copy of the source with a rectangle and a caption.
	annotated := src.Clone()
	cv.Rectangle(annotated, cv.Point{X: 40, Y: 30}, cv.Point{X: 120, Y: 110},
		cv.NewScalar(255, 0, 0), 2)
	cv.PutText(annotated, "cv demo", cv.Point{X: 12, Y: 140}, 2,
		cv.NewScalar(255, 255, 255))
	write("example_annotated.png", annotated)

	// 6. Contour detection: threshold the grayscale image, find external
	// contours and draw them (with their bounding boxes) on a black canvas.
	bin, _ := cv.Threshold(gray, 128, 255, cv.ThreshBinary)
	contours, _ := cv.FindContours(bin, cv.RetrExternal, cv.ChainApproxSimple)
	contourVis := cv.NewMat(h, w, 3)
	cv.DrawContours(contourVis, contours, -1, cv.NewScalar(0, 255, 0), 2)
	for _, c := range contours {
		r := cv.BoundingRect(c)
		cv.Rectangle(contourVis,
			cv.Point{X: r.X, Y: r.Y},
			cv.Point{X: r.X + r.Width - 1, Y: r.Y + r.Height - 1},
			cv.NewScalar(255, 0, 0), 1)
	}
	write("example_contours.png", contourVis)
	fmt.Printf("found %d external contour(s)\n", len(contours))

	// 7. Perspective warp: map the rectangle's four corners to a full-frame
	// axis-aligned rectangle, straightening it into a bird's-eye view.
	srcQuad := [4]cv.Point{{X: 40, Y: 30}, {X: 120, Y: 30}, {X: 120, Y: 110}, {X: 40, Y: 110}}
	dstQuad := [4]cv.Point{{X: 0, Y: 0}, {X: w - 1, Y: 0}, {X: w - 1, Y: h - 1}, {X: 0, Y: h - 1}}
	m := cv.GetPerspectiveTransform(srcQuad, dstQuad)
	warped := cv.WarpPerspective(src, m, w, h, cv.InterLinear)
	write("example_warp.png", warped)

	fmt.Println("wrote example_source.png, example_gray.png, example_blur.png, example_canny.png, example_annotated.png, example_contours.png, example_warp.png")
}

func write(path string, m *cv.Mat) {
	if err := cv.ImWrite(path, m); err != nil {
		log.Fatalf("write %s: %v", path, err)
	}
}
