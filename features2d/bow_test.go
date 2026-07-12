package features2d

import (
	"math"
	"testing"
)

// clusteredDescriptors builds float descriptors drawn tightly around k
// well-separated centres, returning the descriptors and the centre each came
// from.
func clusteredDescriptors(centers [][]float64, perCluster int) ([][]float64, []int) {
	var rows [][]float64
	var labels []int
	for ci, c := range centers {
		for p := 0; p < perCluster; p++ {
			row := make([]float64, len(c))
			// Deterministic tiny offset.
			off := (float64(p) - float64(perCluster)/2) * 1e-3
			for j := range row {
				row[j] = c[j] + off
			}
			rows = append(rows, row)
			labels = append(labels, ci)
		}
	}
	return rows, labels
}

func TestBOWKMeansTrainerRecoversClusters(t *testing.T) {
	centers := [][]float64{
		{0, 0}, {10, 0}, {0, 10}, {10, 10},
	}
	rows, _ := clusteredDescriptors(centers, 20)
	trainer := NewBOWKMeansTrainer(4, 0, 0)
	trainer.Add(NewFloatDescriptors(rows))
	if trainer.DescriptorsCount() != len(rows) {
		t.Fatalf("descriptor count %d != %d", trainer.DescriptorsCount(), len(rows))
	}
	vocab := trainer.Cluster()
	if vocab.Len() != 4 {
		t.Fatalf("expected 4 words, got %d", vocab.Len())
	}
	// Every true centre must be close to some learned word.
	for _, c := range centers {
		best := math.Inf(1)
		for _, w := range vocab.Float {
			if d := squaredL2(c, w); d < best {
				best = d
			}
		}
		if best > 1 {
			t.Fatalf("no learned word near centre %v (min dist^2=%.3f)", c, best)
		}
	}
}

func TestBOWImgDescriptorExtractor(t *testing.T) {
	centers := [][]float64{
		{0, 0}, {10, 0}, {0, 10}, {10, 10},
	}
	rows, _ := clusteredDescriptors(centers, 20)
	trainer := NewBOWKMeansTrainer(4, 0, 0)
	trainer.Add(NewFloatDescriptors(rows))
	vocab := trainer.Cluster()

	ext := NewBOWImgDescriptorExtractor(NewBFMatcher(NormL2))
	ext.SetVocabulary(vocab)
	if ext.VocabularySize() != 4 {
		t.Fatalf("vocab size %d", ext.VocabularySize())
	}

	// An "image" made of descriptors from clusters 0 and 1 only should yield a
	// histogram with mass concentrated in two bins that sums to 1.
	img, _ := clusteredDescriptors([][]float64{centers[0], centers[1]}, 5)
	hist := ext.Compute(NewFloatDescriptors(img))
	if len(hist) != 4 {
		t.Fatalf("histogram length %d", len(hist))
	}
	var sum float64
	nonzero := 0
	for _, v := range hist {
		sum += v
		if v > 0 {
			nonzero++
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("histogram not L1-normalised: sum=%.6f", sum)
	}
	if nonzero != 2 {
		t.Fatalf("expected mass in exactly 2 bins, got %d", nonzero)
	}
}

func TestBOWEmptyImageHistogram(t *testing.T) {
	vocab := NewFloatDescriptors([][]float64{{0, 0}, {1, 1}})
	ext := NewBOWImgDescriptorExtractor(NewBFMatcher(NormL2))
	ext.SetVocabulary(vocab)
	hist := ext.Compute(NewFloatDescriptors(nil))
	if len(hist) != 2 || hist[0] != 0 || hist[1] != 0 {
		t.Fatalf("expected all-zero histogram, got %v", hist)
	}
}
