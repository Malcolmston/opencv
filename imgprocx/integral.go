package imgprocx

import cv "github.com/malcolmston/opencv"

// IntegralImage computes the summed-area tables of img. It returns two tables,
// each of size (Rows+1)×(Cols+1) with a zero first row and first column, where
//
//	sum[y+1][x+1]   = Σ intensity(j, i)     for 0<=j<=y, 0<=i<=x
//	sqSum[y+1][x+1] = Σ intensity(j, i)²    for 0<=j<=y, 0<=i<=x
//
// and a pixel's intensity is the sum of its channel samples (see [RectSum]).
// The extra leading row and column let the sum over any rectangle be read from
// four lookups without special-casing the image border, matching OpenCV's
// cv::integral layout.
//
// Use [RectSum] to read the total intensity over a rectangle from the returned
// sum table in constant time.
func IntegralImage(img *cv.Mat) (sum, sqSum [][]float64) {
	rows, cols := img.Rows, img.Cols
	sum = make([][]float64, rows+1)
	sqSum = make([][]float64, rows+1)
	for i := range sum {
		sum[i] = make([]float64, cols+1)
		sqSum[i] = make([]float64, cols+1)
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := pixelSum(img, y, x)
			sum[y+1][x+1] = v + sum[y][x+1] + sum[y+1][x] - sum[y][x]
			sqSum[y+1][x+1] = v*v + sqSum[y][x+1] + sqSum[y+1][x] - sqSum[y][x]
		}
	}
	return sum, sqSum
}

// RectSum returns the total intensity over the half-open rectangle
// [x0, x1) × [y0, y1) from a summed-area table produced by [IntegralImage],
// using the four-corner formula
//
//	sum[y1][x1] - sum[y0][x1] - sum[y1][x0] + sum[y0][x0].
//
// The coordinates are in image space (0 <= x0 <= x1 <= Cols, 0 <= y0 <= y1 <=
// Rows); the table indexing offset of one is applied internally. It panics if
// the rectangle is malformed or extends outside the table.
func RectSum(sum [][]float64, x0, y0, x1, y1 int) float64 {
	if x0 > x1 || y0 > y1 {
		panic("imgprocx: RectSum requires x0<=x1 and y0<=y1")
	}
	if x0 < 0 || y0 < 0 || y1 >= len(sum) || x1 >= len(sum[0]) {
		panic("imgprocx: RectSum rectangle out of range")
	}
	return sum[y1][x1] - sum[y0][x1] - sum[y1][x0] + sum[y0][x0]
}
