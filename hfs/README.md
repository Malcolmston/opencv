# hfs — pure-Go port of OpenCV's hfs module (Hierarchical Feature Selection for efficient image

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/hfs.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/hfs)

Package hfs is a pure-Go port of OpenCV's hfs module (Hierarchical Feature
Selection for efficient image segmentation) built on the stdlib-only OpenCV
port github.com/malcolmston/opencv (imported here as cv).

1. SLIC superpixels (Achanta et al., 2012) group pixels into compact,
boundary-adhering atomic regions. The grid spacing is controlled by
SlicSpixelSize, the colour-vs-space trade-off by SpatialWeight and the number
of Lloyd iterations by NumSlicIter. 2. A first Efficient Graph-Based (EGB)
merge — the Felzenszwalb & Huttenlocher minimum-spanning-forest criterion
(2004) — runs over the region-adjacency graph of the superpixels in a
colour+texture feature space, using SegEgbThresholdI, after which regions
smaller than MinRegionSizeI pixels are absorbed into their most similar
neighbour. 3. A second EGB merge with SegEgbThresholdII repeats the process on
the stage-I regions, followed by absorption of regions below MinRegionSizeII.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/hfs"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./hfs/`.

```go
// A 64×64 RGB image split into four solid-colour quadrants.
	const size = 64
	img := cv.NewMat(size, size, 3)
	colors := [4][3]uint8{{220, 30, 30}, {30, 200, 30}, {30, 30, 220}, {230, 220, 40}}
	half := size / 2
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			q := 0
			if x >= half {
				q++
			}
			if y >= half {
				q += 2
			}
			img.SetPixel(y, x, colors[q][:])
		}
	}

	seg := hfs.CreateWithDefaults(img.Rows, img.Cols)
	drawn := seg.PerformSegmentCpu(img, true)

	fmt.Printf("regions=%d output=%dx%dx%d\n", seg.NumSegments(), drawn.Rows, drawn.Cols, drawn.Channels)
```

```
regions=4 output=64x64x3
```

## Exported surface

### Types

| Type | What it is |
| --- | --- |
| `DrawMode` | DrawMode selects how `HfsSegment.DrawSegmentation` colours each region. |
| `HfsSegment` | HfsSegment performs Hierarchical Feature Selection segmentation for a fixed image size. |

<details>
<summary><code>HfsSegment</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Create(height, width int, segEgbThresholdI float64, minRegionSizeI int, segEgbThresholdII float64, minRegionSizeII int, spatialWeight float64, slicSpixelSize, numSlicIter int) *HfsSegment` | Create returns an `HfsSegment` configured for images of the given height and width with every parameter specified explicitly. |
| `func CreateWithDefaults(height, width int) *HfsSegment` | CreateWithDefaults returns an `HfsSegment` for images of the given height and width using the OpenCV default parameters (see the Default* constants). |
| `func (h *HfsSegment) DrawSegmentation(mode DrawMode) *cv.Mat` | DrawSegmentation renders the most recent segmentation as a three-channel image using the given `DrawMode`. |
| `func (h *HfsSegment) GetMinRegionSizeI() int` | GetMinRegionSizeI returns the stage-I minimum region size in pixels. |
| `func (h *HfsSegment) GetMinRegionSizeII() int` | GetMinRegionSizeII returns the stage-II minimum region size in pixels. |
| `func (h *HfsSegment) GetNumSlicIter() int` | GetNumSlicIter returns the number of SLIC Lloyd iterations. |
| `func (h *HfsSegment) GetSegEgbThresholdI() float64` | GetSegEgbThresholdI returns the stage-I EGB merge threshold. |
| `func (h *HfsSegment) GetSegEgbThresholdII() float64` | GetSegEgbThresholdII returns the stage-II EGB merge threshold. |
| `func (h *HfsSegment) GetSlicSpixelSize() int` | GetSlicSpixelSize returns the nominal SLIC superpixel edge length in pixels. |
| `func (h *HfsSegment) GetSpatialWeight() float64` | GetSpatialWeight returns the SLIC colour-vs-space compactness weight. |
| `func (h *HfsSegment) Height() int` | Height returns the configured image height. |
| `func (h *HfsSegment) Labels() (labels []int, rows, cols int)` | Labels returns a copy of the per-pixel region labelling from the most recent segmentation together with its dimensions. |
| `func (h *HfsSegment) NumSegments() int` | NumSegments returns the number of regions produced by the most recent segmentation, or 0 before any segmentation has run. |
| `func (h *HfsSegment) PerformSegmentCpu(src *cv.Mat, ifDraw bool) *cv.Mat` | PerformSegmentCpu segments src and stores the result. |
| `func (h *HfsSegment) PerformSegmentGpu(src *cv.Mat, ifDraw bool) *cv.Mat` | PerformSegmentGpu is provided for API compatibility with OpenCV's HfsSegment. |
| `func (h *HfsSegment) SetMinRegionSizeI(v int)` | SetMinRegionSizeI sets the stage-I minimum region size in pixels. |
| `func (h *HfsSegment) SetMinRegionSizeII(v int)` | SetMinRegionSizeII sets the stage-II minimum region size in pixels. |
| `func (h *HfsSegment) SetNumSlicIter(v int)` | SetNumSlicIter sets the number of SLIC Lloyd iterations. |
| `func (h *HfsSegment) SetSegEgbThresholdI(v float64)` | SetSegEgbThresholdI sets the stage-I EGB merge threshold. |
| `func (h *HfsSegment) SetSegEgbThresholdII(v float64)` | SetSegEgbThresholdII sets the stage-II EGB merge threshold. |
| `func (h *HfsSegment) SetSlicSpixelSize(v int)` | SetSlicSpixelSize sets the nominal SLIC superpixel edge length in pixels. |
| `func (h *HfsSegment) SetSpatialWeight(v float64)` | SetSpatialWeight sets the SLIC colour-vs-space compactness weight. |
| `func (h *HfsSegment) Width() int` | Width returns the configured image width. |

</details>

### Constants

`DefaultSegEgbThresholdI`, `DefaultMinRegionSizeI`, `DefaultSegEgbThresholdII`, `DefaultMinRegionSizeII`, `DefaultSpatialWeight`, `DefaultSlicSpixelSize`, `DefaultNumSlicIter`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/hfs).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
