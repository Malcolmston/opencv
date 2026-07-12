package tracking_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/tracking"
)

// texturedPatch renders a 60×60 grayscale frame with a dark background and a
// textured 16×16 bright square (with asymmetric corner marks) centred at
// (cx, cy), giving the correlation filters the intensity variation they need.
func texturedPatch(cx, cy int) *cv.Mat {
	m := cv.NewMat(60, 60, 1)
	for i := range m.Data {
		m.Data[i] = 30
	}
	set := func(x, y int, v uint8) {
		if x >= 0 && x < 60 && y >= 0 && y < 60 {
			m.Data[y*60+x] = v
		}
	}
	for y := cy - 8; y <= cy+8; y++ {
		for x := cx - 8; x <= cx+8; x++ {
			set(x, y, 120)
		}
	}
	set(cx-5, cy-5, 240)
	set(cx+5, cy-5, 200)
	set(cx-5, cy+5, 210)
	for y := cy - 6; y <= cy-3; y++ {
		for x := cx - 6; x <= cx-3; x++ {
			set(x, y, 240)
		}
	}
	return m
}

// ExampleTrackerMOSSE initialises the MOSSE correlation filter and follows the
// patch as it moves, printing the per-frame peak-to-sidelobe confidence.
func ExampleTrackerMOSSE() {
	tr := tracking.NewTrackerMOSSE()
	tr.Init(texturedPatch(20, 20), cv.Rect{X: 12, Y: 12, Width: 16, Height: 16})
	box, conf := tr.UpdateConfidence(texturedPatch(23, 21)) // moved by (3, 1)
	cx, cy := box.X+box.Width/2, box.Y+box.Height/2
	fmt.Printf("center=(%d,%d) confident=%v\n", cx, cy, conf > tr.MinPSR)
	// Output: center=(23,21) confident=true
}

// ExampleTrackerDCF follows a patch with the genuine FFT-based KCF tracker.
func ExampleTrackerDCF() {
	tr := tracking.NewTrackerDCF()
	tr.Init(texturedPatch(20, 20), cv.Rect{X: 12, Y: 12, Width: 16, Height: 16})
	box, _ := tr.Update(texturedPatch(23, 21))
	fmt.Printf("center=(%d,%d)\n", box.X+box.Width/2, box.Y+box.Height/2)
	// Output: center=(22,21)
}

// ExampleMultiTracker tracks two objects at once, each with a different tracker.
func ExampleMultiTracker() {
	frame := func(ax, ay, bx, by int) *cv.Mat {
		m := cv.NewMat(80, 80, 1)
		for i := range m.Data {
			m.Data[i] = 30
		}
		stamp := func(cx, cy int) {
			p := texturedPatch(30, 30) // reuse the textured square
			for y := -8; y <= 8; y++ {
				for x := -8; x <= 8; x++ {
					if cx+x >= 0 && cx+x < 80 && cy+y >= 0 && cy+y < 80 {
						m.Data[(cy+y)*80+(cx+x)] = p.Data[(30+y)*60+(30+x)]
					}
				}
			}
		}
		stamp(ax, ay)
		stamp(bx, by)
		return m
	}
	mt := tracking.NewMultiTracker()
	mt.Add(tracking.NewTrackerMOSSE(), frame(20, 20, 60, 60), cv.Rect{X: 12, Y: 12, Width: 16, Height: 16})
	mt.Add(tracking.NewTrackerMOSSE(), frame(20, 20, 60, 60), cv.Rect{X: 52, Y: 52, Width: 16, Height: 16})
	boxes, _ := mt.Update(frame(22, 21, 58, 59))
	fmt.Printf("obj0=(%d,%d) obj1=(%d,%d)\n",
		boxes[0].X+boxes[0].Width/2, boxes[0].Y+boxes[0].Height/2,
		boxes[1].X+boxes[1].Width/2, boxes[1].Y+boxes[1].Height/2)
	// Output: obj0=(22,21) obj1=(58,59)
}

// ExampleFFT2 shows the forward and inverse 2D FFT round-tripping a small grid.
func ExampleFFT2() {
	m := tracking.NewComplexMat(2, 2)
	m.Data[0] = complex(1, 0)
	m.Data[1] = complex(2, 0)
	m.Data[2] = complex(3, 0)
	m.Data[3] = complex(4, 0)
	spec := tracking.FFT2(m)
	back := tracking.IFFT2(spec)
	fmt.Printf("DC=%.0f recovered=%.0f\n", real(spec.Data[0]), real(back.Data[3]))
	// Output: DC=10 recovered=4
}
