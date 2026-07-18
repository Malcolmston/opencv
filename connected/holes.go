package connected

import cv "github.com/malcolmston/opencv"

// connectedDual returns the topological dual of conn: the background is analysed
// with 4-connectivity when the foreground uses 8-connectivity, and vice versa.
// Using the dual keeps the Euler number consistent and matches the classic
// hole-filling convention.
func connectedDual(conn Connectivity) Connectivity {
	if conn == Conn8 {
		return Conn4
	}
	return Conn8
}

// connectedBorderReachable returns a boolean mask marking every background pixel
// (zero sample) of src that is connected to the image border under bgConn. Holes
// are exactly the background pixels this flood does not reach.
func connectedBorderReachable(src *cv.Mat, bgConn Connectivity) []bool {
	w, h := src.Cols, src.Rows
	visited := make([]bool, w*h)
	inRegion := func(i int) bool { return src.Data[i] == 0 && !visited[i] }
	visit := func(i int) { visited[i] = true }
	seed := func(x, y int) {
		i := y*w + x
		if src.Data[i] == 0 && !visited[i] {
			connectedScanFill(w, h, x, y, bgConn, inRegion, visit)
		}
	}
	for x := 0; x < w; x++ {
		seed(x, 0)
		seed(x, h-1)
	}
	for y := 0; y < h; y++ {
		seed(0, y)
		seed(w-1, y)
	}
	return visited
}

// FillHoles fills the interior holes of the foreground: every background region
// not connected to the image border is turned into foreground. The result is a
// binary image (255 foreground, 0 background). conn is the foreground
// connectivity; the background is analysed with the dual connectivity. src is
// not modified.
func FillHoles(src *cv.Mat, conn Connectivity) *cv.Mat {
	connectedRequireBinary(src, "FillHoles")
	connectedCheckConn(conn, "FillHoles")
	reachable := connectedBorderReachable(src, connectedDual(conn))
	out := cv.NewMat(src.Rows, src.Cols, 1)
	for i := range out.Data {
		if src.Data[i] != 0 || !reachable[i] {
			out.Data[i] = 255
		}
	}
	return out
}

// HolesMask returns a binary image marking only the filled-in holes: background
// pixels that are enclosed by the foreground and therefore not connected to the
// image border. Enclosed pixels are 255, everything else 0.
func HolesMask(src *cv.Mat, conn Connectivity) *cv.Mat {
	connectedRequireBinary(src, "HolesMask")
	connectedCheckConn(conn, "HolesMask")
	reachable := connectedBorderReachable(src, connectedDual(conn))
	out := cv.NewMat(src.Rows, src.Cols, 1)
	for i := range out.Data {
		if src.Data[i] == 0 && !reachable[i] {
			out.Data[i] = 255
		}
	}
	return out
}

// CountHoles returns the number of holes in the foreground: the number of
// background components that are fully enclosed and not connected to the image
// border. conn is the foreground connectivity; holes are counted under the dual
// connectivity.
func CountHoles(src *cv.Mat, conn Connectivity) int {
	connectedRequireBinary(src, "CountHoles")
	connectedCheckConn(conn, "CountHoles")
	bgConn := connectedDual(conn)
	reachable := connectedBorderReachable(src, bgConn)
	// Enclosed background pixels, labelled under the background connectivity;
	// each label is one hole.
	enclosed := make([]bool, len(src.Data))
	for i := range enclosed {
		enclosed[i] = src.Data[i] == 0 && !reachable[i]
	}
	_, count := connectedLabelMask(enclosed, src.Cols, src.Rows, bgConn)
	return count
}

// EulerNumber returns the Euler number of the binary image: the number of
// connected foreground components minus the number of holes. conn is the
// foreground connectivity; holes are measured under the dual connectivity, so
// the result is topologically consistent.
func EulerNumber(src *cv.Mat, conn Connectivity) int {
	connectedRequireBinary(src, "EulerNumber")
	connectedCheckConn(conn, "EulerNumber")
	return CountComponents(src, conn) - CountHoles(src, conn)
}
