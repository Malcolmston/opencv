package imghash2

import (
	"encoding/hex"
	"math"
	"math/bits"
)

// HammingDistance returns the number of differing bits between two equal-length
// binary hashes. It is the free-function form of [Hash.Hamming] and panics on a
// length mismatch.
func HammingDistance(a, b Hash) int { return a.Hamming(b) }

// NormalizedHamming returns the Hamming distance between two equal-length binary
// hashes divided by their bit count, a value in [0, 1]. It is the free-function
// form of [Hash.NormalizedHamming] and panics if the hashes differ in length or
// are empty.
func NormalizedHamming(a, b Hash) float64 { return a.NormalizedHamming(b) }

// Similarity returns a normalised similarity score in [0, 1] between two binary
// hashes: 1 for identical fingerprints, falling toward 0 as bits diverge. It is
// the free-function form of [Hash.Similarity].
func Similarity(a, b Hash) float64 { return a.Similarity(b) }

// L1Distance returns the L1 (Manhattan) distance between two equal-length
// descriptors, the free-function form of [FloatHash.L1].
func L1Distance(a, b FloatHash) float64 { return a.L1(b) }

// L2Distance returns the L2 (Euclidean) distance between two equal-length
// descriptors, the free-function form of [FloatHash.L2].
func L2Distance(a, b FloatHash) float64 { return a.L2(b) }

// PopCount returns the total number of set bits across all bytes of b, its
// Hamming weight.
func PopCount(b []byte) int {
	n := 0
	for _, v := range b {
		n += bits.OnesCount8(v)
	}
	return n
}

// HexEncode returns the lower-case hexadecimal text form of raw hash bytes. For
// a [Hash] value prefer [Hash.String], which does the same thing.
func HexEncode(b []byte) string { return hex.EncodeToString(b) }

// HexDecode parses hexadecimal text into raw bytes, returning an error for
// invalid or odd-length input. For a [Hash] value prefer [ParseHash].
func HexDecode(s string) ([]byte, error) { return hex.DecodeString(s) }

// Mean returns the arithmetic mean of vals, or 0 for an empty slice.
func Mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// Median returns the median of vals. The input is copied, so the caller's slice
// is left unmodified. It returns 0 for an empty slice.
func Median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := make([]float64, len(vals))
	copy(cp, vals)
	insertionSort(cp)
	n := len(cp)
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
}

// Variance returns the population variance of vals, or 0 for an empty slice.
func Variance(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	m := Mean(vals)
	var s float64
	for _, v := range vals {
		d := v - m
		s += d * d
	}
	return s / float64(len(vals))
}

// StdDev returns the population standard deviation of vals, the square root of
// [Variance].
func StdDev(vals []float64) float64 { return math.Sqrt(Variance(vals)) }

// insertionSort sorts a small float slice ascending in place. The hashers only
// sort blocks of a few dozen to a few thousand values, so this keeps the
// package free of any import beyond the numeric primitives.
func insertionSort(a []float64) {
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] > v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}
