// Package flann is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's flann module: FLANN, "Fast Library for Approximate
// Nearest Neighbors". It builds search structures over collections of vectors
// and answers nearest-neighbour queries — the k closest points to a query
// (k-NN) and every point within a fixed radius.
//
// Like the parent package, flann is written entirely against the Go standard
// library (math, sort, math/bits, math/rand, container/heap). It uses no cgo
// and no third-party dependencies, and it does not import the other cv/*
// subpackages. The data it works on is plain Go: a dataset of real-valued
// descriptors is a [][]float64 (one row per point, columns are dimensions) and
// a dataset of binary descriptors is a [][]byte (one row per point, each byte
// eight packed bits). This keeps the API independent of the image-oriented
// cv.Mat type.
//
// # The Index interface
//
// Every search structure implements the generic [Index] interface, whose type
// parameter is the query (and dataset element) type — []float64 for the
// real-valued indices, []byte for the binary one:
//
//	type Index[T any] interface {
//		KnnSearch(query T, k int) []Neighbor
//		RadiusSearch(query T, radius float64) []Neighbor
//	}
//
// Both methods return a slice of [Neighbor], each pairing a dataset row index
// with its distance to the query. Results are always sorted ascending by
// distance, ties broken by ascending index, so searches are deterministic and
// two indices that examine the same candidate set return byte-identical
// results.
//
// # The indices
//
// The following index types are provided:
//
//   - [LinearIndex] is the exact brute-force baseline: it scans every point.
//     Generic over the element type, it is constructed with [NewLinearIndex]
//     for L2 (Euclidean) search over [][]float64, with [NewLinearBinaryIndex]
//     for Hamming search over [][]byte, or with [NewLinearIndexFunc] for a
//     custom [DistanceFunc]. Because it is exact, it is the reference every
//     approximate index is measured against.
//
//   - [KDTreeIndex] is a single median-split k-d tree over [][]float64. With
//     its default settings it performs an exact k-NN and radius search (its
//     results match [LinearIndex] exactly); setting MaxChecks bounds the
//     backtracking to trade recall for speed on high-dimensional data.
//
//   - [KMeansIndex] is a hierarchical k-means tree over [][]float64. The points
//     are recursively partitioned into clusters and searched best-bin-first.
//     With Checks == 0 the search is exhaustive (exact); a positive Checks
//     bounds the number of points examined and yields an approximate result
//     with high recall.
//
//   - [LSHIndex] is a multi-table locality-sensitive hash over binary
//     descriptors [][]byte using the Hamming distance. Each table hashes a
//     random subset of bit positions; a query gathers candidates from the
//     matching bucket of every table and ranks them exactly. It excels at
//     finding a near-exact binary match among many distractors.
//
//   - [KDForestIndex] is a randomized multi-tree k-d forest over [][]float64.
//     Each tree splits on a dimension drawn randomly from the few of highest
//     variance, so the trees are decorrelated and a shared best-bin-first
//     traversal explores complementary regions. It is exact with MaxChecks == 0
//     and trades recall for speed under a positive budget — the classic FLANN
//     structure for moderate-to-high dimension.
//
//   - [HierarchicalClusteringIndex] recursively clusters [][]float64 around
//     randomly chosen data points, so it works under any [DistanceFunc], not
//     just L2 (see [NewHierarchicalClusteringIndexFunc]). It is exact with
//     Checks == 0.
//
//   - [CompositeIndex] queries a k-d forest and a [KMeansIndex] together and
//     merges their candidates, recovering points either structure alone would
//     miss.
//
//   - [AutotunedIndex] picks a structure and check budget automatically to reach
//     a requested precision, measured against exact search on a sample of the
//     data.
//
// [KnnSearchBatch] and [RadiusSearchBatch] answer many queries in one call, and
// [Recall] and [Precision] score any approximate index against an exact one.
//
// # Distances
//
// [DistL2] is the Euclidean distance between two float vectors and [DistHamming]
// is the number of differing bits between two byte vectors. Additional float
// distances are provided: [DistL1] (Manhattan), [DistMinkowski] (order-p, with
// the [MinkowskiDist] constructor for a bound [DistanceFunc]), [DistChiSquare]
// (histogram dissimilarity), [DistHellinger] and [DistCosine]. All are exported
// so callers can score candidates themselves, and each is expressed in the same
// units the index using it reports and the radius of RadiusSearch is measured
// in.
//
// # Persistence
//
// [KDForestIndex] and [AutotunedIndex] serialize through encoding/gob; the
// [Save] and [Load] helpers wrap an index to and from any io.Writer/io.Reader,
// and a reloaded index answers queries identically to the original.
//
// # Determinism
//
// Structures whose construction draws on randomness — [KMeansIndex]'s
// k-means++ seeding and Lloyd refinement, and [LSHIndex]'s choice of hash bit
// positions — take an explicit int64 seed, so repeated builds on the same data
// produce identical trees, hashes and query results. There is no hidden global
// state.
//
// # Errors and panics
//
// Constructors validate their dataset and panic with a descriptive message on
// programmer error — a ragged dataset whose rows differ in length, or an
// out-of-range parameter such as an LSH key wider than 64 bits — mirroring the
// validate-and-panic convention of the parent package. Building an index over
// an empty dataset is allowed; its searches simply return no neighbours. The
// distance functions panic on a length mismatch, like a Go slice index error.
//
// # Not implemented
//
// The following parts of OpenCV's flann remain out of scope: incremental
// addition or removal of points from a built index, GPU acceleration, and the
// on-disk format of the original C++ library (this package serializes with Go's
// own encoding/gob instead, via [Save] and [Load]).
package flann
