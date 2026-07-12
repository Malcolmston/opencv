package features2d

import "fmt"

func ExampleHammingDistance() {
	// 0b1010 vs 0b0101 differ in all four low bits.
	fmt.Println(HammingDistance([]byte{0b1010}, []byte{0b0101}))
	// Output: 4
}

func ExampleBFMatcher_Match() {
	// Each query descriptor has an identical twin in the train set.
	query := NewBinaryDescriptors([][]byte{{0xAA}, {0x0F}})
	train := NewBinaryDescriptors([][]byte{{0x0F}, {0xAA}})

	m := NewBFMatcher(NormHamming)
	for _, match := range m.Match(query, train) {
		fmt.Printf("query %d -> train %d (distance %.0f)\n",
			match.QueryIdx, match.TrainIdx, match.Distance)
	}
	// Output:
	// query 0 -> train 1 (distance 0)
	// query 1 -> train 0 (distance 0)
}

func ExampleRatioTest() {
	// A KnnMatch (k=2) result: the first query is unambiguous, the second is
	// not, so Lowe's ratio test keeps only the first.
	knn := [][]DMatch{
		{{QueryIdx: 0, TrainIdx: 5, Distance: 2}, {QueryIdx: 0, TrainIdx: 8, Distance: 20}},
		{{QueryIdx: 1, TrainIdx: 3, Distance: 9}, {QueryIdx: 1, TrainIdx: 7, Distance: 10}},
	}
	for _, m := range RatioTest(knn, 0.75) {
		fmt.Printf("kept query %d -> train %d\n", m.QueryIdx, m.TrainIdx)
	}
	// Output:
	// kept query 0 -> train 5
}

func ExampleORB_DetectAndCompute() {
	img := buildScene(110)
	kps, desc := NewORB(300).DetectAndCompute(img)
	// Descriptors are 32-byte (256-bit) binary strings, one per keypoint.
	fmt.Println(len(desc) == len(kps), len(desc[0]))
	// Output: true 32
}
