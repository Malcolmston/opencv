# flann — from-scratch, standard-library-only port of a useful subset of OpenCV's flann module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/flann.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/flann)

Package flann is a from-scratch, standard-library-only port of a useful subset
of OpenCV's flann module: FLANN, "Fast Library for Approximate Nearest
Neighbors". It builds search structures over collections of vectors and
answers nearest-neighbour queries — the k closest points to a query (k-NN)
and every point within a fixed radius.

Like the parent package, flann is written entirely against the Go standard
library (math, sort, math/bits, math/rand, container/heap). It uses no cgo and
no third-party dependencies, and it does not import the other cv/*
subpackages. The data it works on is plain Go: a dataset of real-valued
descriptors is a [][]float64 (one row per point, columns are dimensions) and a
dataset of binary descriptors is a [][]byte (one row per point, each byte
eight packed bits). This keeps the API independent of the image-oriented
cv.Mat type.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/flann"
```

## Usage

This is the package's own `ExampleAutotunedIndex`, so it compiles and its output is
asserted on every `go test ./flann/`.

```go
data := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{50, 50}, {50, 51}, {51, 50}, {51, 51},
	}
	auto := flann.NewAutotunedIndex(data, 1.0, 1)

	nb := auto.KnnSearch([]float64{50.2, 50.2}, 1)[0]
	fmt.Printf("nearest is index %d\n", nb.Index)
```

```
nearest is index 4
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DistChiSquare(a, b []float64) float64` | DistChiSquare returns the (symmetric) chi-square distance between two non-negative vectors, sum((a`i`-b`i`)^2 / (a`i`+b`i`)), a standard measure of… |
| `func DistCosine(a, b []float64) float64` | DistCosine returns the cosine distance between two real-valued vectors, 1 - (a·b)/(\|a\|\|b\|), which lies in [0, 2]: 0 for vectors pointing the… |
| `func DistHamming(a, b []byte) float64` | DistHamming returns the Hamming distance between two binary descriptors: the number of bit positions at which they differ. |
| `func DistHellinger(a, b []float64) float64` | DistHellinger returns the Hellinger distance between two non-negative vectors (typically probability distributions or histograms), sqrt(0.5 *… |
| `func DistL1(a, b []float64) float64` | DistL1 returns the Manhattan (L1, "city block") distance between two real-valued vectors, sum(\|a`i`-b`i`\|). |
| `func DistL2(a, b []float64) float64` | DistL2 returns the Euclidean (L2) distance between two real-valued vectors, sqrt(sum((a`i`-b`i`)^2)). |
| `func DistMinkowski(a, b []float64, p float64) float64` | DistMinkowski returns the order-p Minkowski distance between two real-valued vectors, (sum(\|a`i`-b`i`\|^p))^(1/p). |
| `func KnnSearchBatch[T any](idx Index[T], queries []T, k int) [][]Neighbor` | KnnSearchBatch answers many k-nearest-neighbour queries in one call, the analogue of OpenCV's matrix-valued flann::Index::knnSearch. |
| `func Load(r io.Reader, v any) error` | Load reads a value previously written by `Save` from r into v, which must be a pointer, e.g. |
| `func Precision[T any](approx, exact Index[T], queries []T, k int) float64` | Precision measures how many of an approximate index's returned neighbours are genuine, averaged over queries. |
| `func RadiusSearchBatch[T any](idx Index[T], queries []T, radius float64) [][]Neighbor` | RadiusSearchBatch answers many radius queries in one call. |
| `func Recall[T any](approx, exact Index[T], queries []T, k int) float64` | Recall measures how many true nearest neighbours an approximate index recovers, averaged over queries. |
| `func Save(w io.Writer, v any) error` | Save writes an index (or any gob-serializable value) to w using encoding/gob. |

### Types

| Type | What it is |
| --- | --- |
| `AutotunedIndex` | AutotunedIndex chooses an approximate search configuration automatically to meet a target precision, mirroring OpenCV's flann AutotunedIndex. |
| `CompositeIndex` | CompositeIndex searches a real-valued dataset with two complementary structures at once — a randomized k-d forest (`KDForestIndex`) and a… |
| `DistanceFunc` | DistanceFunc reports the distance between two elements of type T. It must be symmetric and non-negative, returning 0 for identical inputs, so that… |
| `HierarchicalClusteringIndex` | HierarchicalClusteringIndex partitions a real-valued dataset by recursive clustering: at each node it picks Branching cluster centres at random from… |
| `Index` | Index is the common interface implemented by every FLANN search structure. |
| `KDForestIndex` | KDForestIndex is a randomized multi-tree k-d forest over a real-valued dataset, the classic FLANN structure for approximate nearest-neighbour search… |
| `KDTreeIndex` | KDTreeIndex is a single median-split k-d tree over a real-valued dataset, searched with backtracking. |
| `KMeansIndex` | KMeansIndex is a hierarchical k-means tree over a real-valued dataset. |
| `LSHIndex` | LSHIndex is a multi-table locality-sensitive hash over binary descriptors, using the Hamming distance. |
| `LinearIndex` | LinearIndex is the exact brute-force baseline: KnnSearch and RadiusSearch scan the entire dataset. |
| `Neighbor` | Neighbor is a single search result: the row index of a point in the dataset the index was built over, and that point's Distance to the query. |

<details>
<summary><code>AutotunedIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAutotunedIndex(data [][]float64, targetPrecision float64, seed int64) *AutotunedIndex` | NewAutotunedIndex builds an index tuned to answer k-NN queries at about targetPrecision (clamped to [0, 1]), the expected fraction of true nearest… |
| `func (a *AutotunedIndex) AchievedPrecision() float64` | AchievedPrecision reports the precision the selected configuration attained on the tuning sample, an estimate of its true precision. |
| `func (a *AutotunedIndex) Algorithm() string` | Algorithm reports the structure autotuning selected: "kdforest", "kmeans", or "linear" for an empty dataset. |
| `func (a *AutotunedIndex) Checks() int` | Checks reports the check budget autotuning selected; 0 means the search is exact. |
| `func (a *AutotunedIndex) GobDecode(data []byte) error` | GobDecode implements gob.GobDecoder for `AutotunedIndex`, rebuilding the index and re-running autotuning so the loaded index selects the same… |
| `func (a *AutotunedIndex) GobEncode() ([]byte, error)` | GobEncode implements gob.GobEncoder for `AutotunedIndex`. |
| `func (a *AutotunedIndex) KnnSearch(query []float64, k int) []Neighbor` | KnnSearch returns the k nearest neighbours of query using the selected configuration. |
| `func (a *AutotunedIndex) RadiusSearch(query []float64, radius float64) []Neighbor` | RadiusSearch returns every point within radius of query. |
| `func (a *AutotunedIndex) Size() int` | Size returns the number of points in the index. |
| `func (a *AutotunedIndex) TargetPrecision() float64` | TargetPrecision reports the precision the index was asked to reach. |

</details>

<details>
<summary><code>CompositeIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCompositeIndex(data [][]float64, trees, branching, leafSize int, seed int64) *CompositeIndex` | NewCompositeIndex builds a composite index over the real-valued dataset. |
| `func (c *CompositeIndex) KnnSearch(query []float64, k int) []Neighbor` | KnnSearch returns the k nearest neighbours of query by querying both sub-indices and re-ranking the union of their candidates. |
| `func (c *CompositeIndex) RadiusSearch(query []float64, radius float64) []Neighbor` | RadiusSearch returns every point within radius of query. |
| `func (c *CompositeIndex) Size() int` | Size returns the number of points in the index. |

</details>

<details>
<summary><code>DistanceFunc</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MinkowskiDist(p float64) DistanceFunc[[]float64]` | MinkowskiDist returns a `DistanceFunc` that computes the order-p Minkowski distance, suitable for `NewLinearIndexFunc`. |

</details>

<details>
<summary><code>HierarchicalClusteringIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHierarchicalClusteringIndex(data [][]float64, branching, leafSize, trees int, seed int64) *HierarchicalClusteringIndex` | NewHierarchicalClusteringIndex builds a hierarchical clustering tree under the Euclidean (`DistL2`) distance. |
| `func NewHierarchicalClusteringIndexFunc(data [][]float64, branching, leafSize, trees int, seed int64, dist DistanceFunc[[]float64]) *HierarchicalClusteringIndex` | NewHierarchicalClusteringIndexFunc is `NewHierarchicalClusteringIndex` with an explicit `DistanceFunc`, so the tree can cluster and search under L1,… |
| `func (h *HierarchicalClusteringIndex) KnnSearch(query []float64, k int) []Neighbor` | KnnSearch returns the k nearest neighbours of query, traversing every tree best-bin-first through a shared queue. |
| `func (h *HierarchicalClusteringIndex) RadiusSearch(query []float64, radius float64) []Neighbor` | RadiusSearch returns every point within radius of query, sorted ascending by distance. |
| `func (h *HierarchicalClusteringIndex) Size() int` | Size returns the number of points in the index. |
| `func (h *HierarchicalClusteringIndex) Trees() int` | Trees returns the number of randomized trees in the index. |

</details>

<details>
<summary><code>KDForestIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKDForestIndex(data [][]float64, trees int, seed int64) *KDForestIndex` | NewKDForestIndex builds a randomized k-d forest over the real-valued dataset. |
| `func (f *KDForestIndex) GobDecode(data []byte) error` | GobDecode implements gob.GobDecoder, reconstructing a `KDForestIndex` — data, parameters and all randomized trees — from bytes produced by… |
| `func (f *KDForestIndex) GobEncode() ([]byte, error)` | GobEncode implements gob.GobEncoder, letting a `KDForestIndex` be persisted with `Save` or any encoding/gob encoder. |
| `func (f *KDForestIndex) KnnSearch(query []float64, k int) []Neighbor` | KnnSearch returns the k nearest neighbours of query, searching every tree through one shared priority queue. |
| `func (f *KDForestIndex) RadiusSearch(query []float64, radius float64) []Neighbor` | RadiusSearch returns every point within radius of query, sorted ascending by distance. |
| `func (f *KDForestIndex) Size() int` | Size returns the number of points in the index. |
| `func (f *KDForestIndex) Trees() int` | Trees returns the number of randomized trees in the forest. |

</details>

<details>
<summary><code>KDTreeIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKDTreeIndex(data [][]float64) *KDTreeIndex` | NewKDTreeIndex builds a k-d tree over the real-valued dataset. |
| `func (t *KDTreeIndex) KnnSearch(query []float64, k int) []Neighbor` | KnnSearch returns the k nearest neighbours of query. |
| `func (t *KDTreeIndex) RadiusSearch(query []float64, radius float64) []Neighbor` | RadiusSearch returns every point within radius of query, sorted ascending by distance. |
| `func (t *KDTreeIndex) Size() int` | Size returns the number of points in the index. |

</details>

<details>
<summary><code>KMeansIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKMeansIndex(data [][]float64, branching, leafSize int, seed int64) *KMeansIndex` | NewKMeansIndex builds a hierarchical k-means tree over the real-valued dataset. |
| `func (idx *KMeansIndex) Centroids() [][]float64` | Centroids returns the centres of the top-level clusters of the k-means tree — the means of the root node's immediate children — which are the… |
| `func (idx *KMeansIndex) ClusterCount() int` | ClusterCount returns the number of leaf clusters in the k-means tree, the length of the slice `KMeansIndex.LeafCentroids` returns. |
| `func (idx *KMeansIndex) KnnSearch(query []float64, k int) []Neighbor` | KnnSearch returns the k nearest neighbours of query, traversing the tree best-bin-first. |
| `func (idx *KMeansIndex) LeafCentroids() [][]float64` | LeafCentroids returns the centre of every leaf of the k-means tree, i.e. |
| `func (idx *KMeansIndex) RadiusSearch(query []float64, radius float64) []Neighbor` | RadiusSearch returns every point within radius of query. |
| `func (idx *KMeansIndex) Size() int` | Size returns the number of points in the index. |

</details>

<details>
<summary><code>LSHIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLSHIndex(data [][]byte, tableNumber, keySize int, seed int64) *LSHIndex` | NewLSHIndex builds a multi-table LSH index over the binary dataset. |
| `func (idx *LSHIndex) KnnSearch(query []byte, k int) []Neighbor` | KnnSearch returns up to the k nearest neighbours of query among the LSH candidate set, ranked by exact Hamming distance. |
| `func (idx *LSHIndex) RadiusSearch(query []byte, radius float64) []Neighbor` | RadiusSearch returns the candidates within Hamming distance radius of query, sorted ascending by distance. |
| `func (idx *LSHIndex) Size() int` | Size returns the number of points in the index. |

</details>

<details>
<summary><code>LinearIndex</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLinearBinaryIndex(data [][]byte) *LinearIndex[[]byte]` | NewLinearBinaryIndex builds an exact Hamming brute-force index over the binary dataset. |
| `func NewLinearIndex(data [][]float64) *LinearIndex[[]float64]` | NewLinearIndex builds an exact L2 (Euclidean) brute-force index over the real-valued dataset. |
| `func NewLinearIndexFunc[T any](data []T, dist DistanceFunc[T]) *LinearIndex[T]` | NewLinearIndexFunc builds an exact brute-force index over an arbitrary dataset scored by the supplied distance function. |
| `func (li *LinearIndex[T]) KnnSearch(query T, k int) []Neighbor` | KnnSearch returns the k nearest neighbours of query by scanning every point. |
| `func (li *LinearIndex[T]) RadiusSearch(query T, radius float64) []Neighbor` | RadiusSearch returns every point within radius of query, sorted ascending by distance. |
| `func (li *LinearIndex[T]) Size() int` | Size returns the number of points in the index. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/flann).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
