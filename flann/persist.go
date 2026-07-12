package flann

import (
	"bytes"
	"encoding/gob"
	"io"
)

// Save writes an index (or any gob-serializable value) to w using encoding/gob.
// It is a thin convenience wrapper over gob.NewEncoder(w).Encode(v); pass the
// index by pointer so its GobEncode method is used, e.g.
// flann.Save(f, forest). Indices that implement gob serialization in this
// package — [KDForestIndex] and [AutotunedIndex] — round-trip through Save and
// [Load] to a structure that answers queries identically to the original.
func Save(w io.Writer, v any) error {
	return gob.NewEncoder(w).Encode(v)
}

// Load reads a value previously written by [Save] from r into v, which must be a
// pointer, e.g. flann.Load(f, &forest). For the package's gob-serializing
// indices the loaded value is fully reconstructed and ready to search.
func Load(r io.Reader, v any) error {
	return gob.NewDecoder(r).Decode(v)
}

// kdForestState is the on-the-wire form of a KDForestIndex. Only the dataset and
// the build parameters are stored; the trees are rebuilt deterministically on
// load from the same seed, so the reconstructed forest is byte-for-byte
// equivalent while the serialized form stays compact.
type kdForestState struct {
	Data      [][]float64
	LeafSize  int
	NumTrees  int
	Seed      int64
	MaxChecks int
}

// GobEncode implements gob.GobEncoder, letting a [KDForestIndex] be persisted
// with [Save] or any encoding/gob encoder.
func (f *KDForestIndex) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	err := gob.NewEncoder(&buf).Encode(kdForestState{
		Data:      f.data,
		LeafSize:  f.leafSize,
		NumTrees:  f.numTrees,
		Seed:      f.seed,
		MaxChecks: f.MaxChecks,
	})
	return buf.Bytes(), err
}

// GobDecode implements gob.GobDecoder, reconstructing a [KDForestIndex] — data,
// parameters and all randomized trees — from bytes produced by GobEncode.
func (f *KDForestIndex) GobDecode(data []byte) error {
	var s kdForestState
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&s); err != nil {
		return err
	}
	f.data = s.Data
	f.dim = validateFloatData(s.Data, "KDForestIndex.GobDecode")
	f.leafSize = s.LeafSize
	if f.leafSize <= 0 {
		f.leafSize = defaultKDLeafSize
	}
	f.numTrees = s.NumTrees
	f.seed = s.Seed
	f.MaxChecks = s.MaxChecks
	f.buildAll()
	return nil
}

// autotunedState is the on-the-wire form of an AutotunedIndex: the dataset,
// target precision and seed suffice to re-run tuning deterministically and
// recover the identical configuration.
type autotunedState struct {
	Data   [][]float64
	Target float64
	Seed   int64
}

// GobEncode implements gob.GobEncoder for [AutotunedIndex].
func (a *AutotunedIndex) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	err := gob.NewEncoder(&buf).Encode(autotunedState{
		Data:   a.data,
		Target: a.target,
		Seed:   a.seedVal,
	})
	return buf.Bytes(), err
}

// GobDecode implements gob.GobDecoder for [AutotunedIndex], rebuilding the index
// and re-running autotuning so the loaded index selects the same configuration.
func (a *AutotunedIndex) GobDecode(data []byte) error {
	var s autotunedState
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&s); err != nil {
		return err
	}
	rebuilt := NewAutotunedIndex(s.Data, s.Target, s.Seed)
	*a = *rebuilt
	return nil
}
