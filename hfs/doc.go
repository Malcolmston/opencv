// Package hfs is a pure-Go port of OpenCV's hfs module (Hierarchical Feature
// Selection for efficient image segmentation) built on the stdlib-only OpenCV
// port github.com/malcolmston/opencv (imported here as cv).
//
// The module implements the segmentation pipeline of Yun Liu et al., "Efficient
// Hierarchical Graph-Based Segmentation of RGBD Videos" and the accompanying
// OpenCV contrib module. Given an RGB image it produces a dense region labelling
// by chaining three classic building blocks, each reimplemented here in terms of
// the root cv package and the Go standard library only:
//
//  1. SLIC superpixels (Achanta et al., 2012) group pixels into compact,
//     boundary-adhering atomic regions. The grid spacing is controlled by
//     SlicSpixelSize, the colour-vs-space trade-off by SpatialWeight and the
//     number of Lloyd iterations by NumSlicIter.
//  2. A first Efficient Graph-Based (EGB) merge — the Felzenszwalb &
//     Huttenlocher minimum-spanning-forest criterion (2004) — runs over the
//     region-adjacency graph of the superpixels in a colour+texture feature
//     space, using SegEgbThresholdI, after which regions smaller than
//     MinRegionSizeI pixels are absorbed into their most similar neighbour.
//  3. A second EGB merge with SegEgbThresholdII repeats the process on the
//     stage-I regions, followed by absorption of regions below MinRegionSizeII.
//
// # Usage
//
// Construct a segmenter for a fixed image size with [Create] (or
// [CreateWithDefaults]), tune it through the Set/Get accessors, then call
// [HfsSegment.PerformSegmentCpu]:
//
//	seg := hfs.CreateWithDefaults(img.Rows, img.Cols)
//	seg.SetSlicSpixelSize(8)
//	drawn := seg.PerformSegmentCpu(img, true) // average-colour rendering
//	labels, rows, cols := seg.Labels()        // dense region labelling
//
// [HfsSegment.PerformSegmentGpu] is provided for API compatibility with the
// OpenCV class; this port has no GPU backend, so it simply forwards to the CPU
// implementation and returns an identical result.
//
// # Feature space
//
// Every superpixel is summarised by a four-dimensional feature vector: the mean
// CIE L*a*b* colour of its pixels (via [cv.CvtColor] with [cv.ColorRGB2Lab]) plus
// a scalar texture response, the mean gradient magnitude inside the region. All
// components are normalised to roughly [0, 1] so that the EGB thresholds behave
// consistently across images; edge weights in the region-adjacency graph are the
// Euclidean distance between the feature vectors of adjacent regions.
//
// # Rendering and labels
//
// Because a [cv.Mat] stores unsigned 8-bit samples it cannot hold an arbitrary
// number of distinct region labels. The full labelling is therefore exposed as a
// flat []int through [HfsSegment.Labels]; [HfsSegment.DrawSegmentation] turns it
// back into a viewable three-channel image, colouring each region either by its
// average source colour ([DrawAverageColor]) or by a deterministic pseudo-random
// colour ([DrawRandomColor]).
//
// # Determinism
//
// All routines are deterministic: SLIC seeds on a fixed grid, the graph merges
// use stable sorting and a union-by-size disjoint-set forest, and the random
// rendering draws from a seeded generator. Given the same inputs the package
// always produces identical output, and every emitted label is a single
// 4-connected component.
package hfs
