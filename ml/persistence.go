package ml

import (
	"bytes"
	"encoding/gob"
	"io"
	"os"
)

// This file provides model persistence built on encoding/gob. The generic
// helpers Save/Load (and their io and byte variants) work with any value; the
// package's own models — [RTrees], [Boost], [ANNMLP], [GaussianMixture] and
// [KernelSVM] — carry unexported state, so each implements gob.GobEncoder and
// gob.GobDecoder via the serialisable snapshot structs below, letting them be
// round-tripped losslessly.

// SaveFile gob-encodes model and writes it to the named file, truncating any
// existing contents.
func SaveFile(path string, model any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := Save(f, model); err != nil {
		return err
	}
	return f.Close()
}

// LoadFile reads a gob-encoded model from the named file into model, which must
// be a non-nil pointer of the matching type.
func LoadFile(path string, model any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return Load(f, model)
}

// Save gob-encodes model to w.
func Save(w io.Writer, model any) error {
	return gob.NewEncoder(w).Encode(model)
}

// Load gob-decodes a model from r into model, which must be a non-nil pointer.
func Load(r io.Reader, model any) error {
	return gob.NewDecoder(r).Decode(model)
}

// gobEncode marshals v (a snapshot struct) to a gob byte slice.
func gobEncode(v any) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// gobDecode unmarshals a gob byte slice produced by gobEncode into v.
func gobDecode(data []byte, v any) error {
	return gob.NewDecoder(bytes.NewReader(data)).Decode(v)
}

// buildIndex returns the label→position map for a sorted class slice.
func buildIndex(classes []int) map[int]int {
	index := make(map[int]int, len(classes))
	for i, c := range classes {
		index[c] = i
	}
	return index
}

// --- decision-tree snapshots (shared by RTrees) ---

// gobNode is the exported, serialisable mirror of a treeNode.
type gobNode struct {
	Leaf       bool
	Prediction int
	Feature    int
	Threshold  float64
	Left       *gobNode
	Right      *gobNode
}

func toGobNode(n *treeNode) *gobNode {
	if n == nil {
		return nil
	}
	return &gobNode{
		Leaf:       n.leaf,
		Prediction: n.prediction,
		Feature:    n.feature,
		Threshold:  n.threshold,
		Left:       toGobNode(n.left),
		Right:      toGobNode(n.right),
	}
}

func fromGobNode(g *gobNode) *treeNode {
	if g == nil {
		return nil
	}
	return &treeNode{
		leaf:       g.Leaf,
		prediction: g.Prediction,
		feature:    g.Feature,
		threshold:  g.Threshold,
		left:       fromGobNode(g.Left),
		right:      fromGobNode(g.Right),
	}
}

// --- RTrees ---

type rtreesState struct {
	NTrees, MaxDepth, MinSamplesSplit, MaxFeatures int
	Seed                                           int64
	Classes                                        []int
	Dim                                            int
	OOBError                                       float64
	Trees                                          []*gobNode
	Trained                                        bool
}

// GobEncode implements gob.GobEncoder.
func (m *RTrees) GobEncode() ([]byte, error) {
	st := rtreesState{
		NTrees: m.NTrees, MaxDepth: m.MaxDepth, MinSamplesSplit: m.MinSamplesSplit,
		MaxFeatures: m.MaxFeatures, Seed: m.Seed, Classes: m.classes, Dim: m.dim,
		OOBError: m.oobError, Trained: m.trained,
	}
	st.Trees = make([]*gobNode, len(m.trees))
	for i, t := range m.trees {
		st.Trees[i] = toGobNode(t)
	}
	return gobEncode(st)
}

// GobDecode implements gob.GobDecoder.
func (m *RTrees) GobDecode(data []byte) error {
	var st rtreesState
	if err := gobDecode(data, &st); err != nil {
		return err
	}
	m.NTrees, m.MaxDepth, m.MinSamplesSplit, m.MaxFeatures = st.NTrees, st.MaxDepth, st.MinSamplesSplit, st.MaxFeatures
	m.Seed, m.classes, m.dim, m.oobError, m.trained = st.Seed, st.Classes, st.Dim, st.OOBError, st.Trained
	m.index = buildIndex(st.Classes)
	m.trees = make([]*treeNode, len(st.Trees))
	for i, g := range st.Trees {
		m.trees[i] = fromGobNode(g)
	}
	return nil
}

// --- Boost ---

type boostState struct {
	NEstimators int
	Stumps      []stump
	Alphas      []float64
	Classes     []int
	Dim         int
	Trained     bool
}

// GobEncode implements gob.GobEncoder.
func (m *Boost) GobEncode() ([]byte, error) {
	return gobEncode(boostState{
		NEstimators: m.NEstimators, Stumps: m.stumps, Alphas: m.alphas,
		Classes: m.classes, Dim: m.dim, Trained: m.trained,
	})
}

// GobDecode implements gob.GobDecoder.
func (m *Boost) GobDecode(data []byte) error {
	var st boostState
	if err := gobDecode(data, &st); err != nil {
		return err
	}
	m.NEstimators, m.stumps, m.alphas = st.NEstimators, st.Stumps, st.Alphas
	m.classes, m.dim, m.trained = st.Classes, st.Dim, st.Trained
	m.index = buildIndex(st.Classes)
	return nil
}

// --- ANNMLP ---

type annmlpState struct {
	HiddenLayers []int
	Activation   Activation
	LearningRate float64
	Epochs       int
	Seed         int64
	LayerSizes   []int
	Weights      [][][]float64
	Biases       [][]float64
	Mean, Std    []float64
	Classes      []int
	Dim          int
	Trained      bool
}

// GobEncode implements gob.GobEncoder.
func (m *ANNMLP) GobEncode() ([]byte, error) {
	st := annmlpState{
		HiddenLayers: m.HiddenLayers, Activation: m.Activation, LearningRate: m.LearningRate,
		Epochs: m.Epochs, Seed: m.Seed, LayerSizes: m.layerSizes, Weights: m.weights,
		Biases: m.biases, Classes: m.classes, Dim: m.dim, Trained: m.trained,
	}
	if m.scaler != nil {
		st.Mean, st.Std = m.scaler.mean, m.scaler.std
	}
	return gobEncode(st)
}

// GobDecode implements gob.GobDecoder.
func (m *ANNMLP) GobDecode(data []byte) error {
	var st annmlpState
	if err := gobDecode(data, &st); err != nil {
		return err
	}
	m.HiddenLayers, m.Activation, m.LearningRate = st.HiddenLayers, st.Activation, st.LearningRate
	m.Epochs, m.Seed, m.layerSizes = st.Epochs, st.Seed, st.LayerSizes
	m.weights, m.biases, m.classes, m.dim, m.trained = st.Weights, st.Biases, st.Classes, st.Dim, st.Trained
	if st.Mean != nil {
		m.scaler = &scaler{mean: st.Mean, std: st.Std}
	}
	return nil
}

// --- GaussianMixture ---

type gmmState struct {
	K             int
	MaxIter       int
	Tol           float64
	Seed          int64
	Weights       []float64
	Means         [][]float64
	Variances     [][]float64
	Dim           int
	LogLikelihood float64
	Trained       bool
}

// GobEncode implements gob.GobEncoder.
func (m *GaussianMixture) GobEncode() ([]byte, error) {
	return gobEncode(gmmState{
		K: m.K, MaxIter: m.MaxIter, Tol: m.Tol, Seed: m.Seed, Weights: m.weights,
		Means: m.means, Variances: m.variances, Dim: m.dim,
		LogLikelihood: m.logLikelihood, Trained: m.trained,
	})
}

// GobDecode implements gob.GobDecoder.
func (m *GaussianMixture) GobDecode(data []byte) error {
	var st gmmState
	if err := gobDecode(data, &st); err != nil {
		return err
	}
	m.K, m.MaxIter, m.Tol, m.Seed = st.K, st.MaxIter, st.Tol, st.Seed
	m.weights, m.means, m.variances = st.Weights, st.Means, st.Variances
	m.dim, m.logLikelihood, m.trained = st.Dim, st.LogLikelihood, st.Trained
	return nil
}

// --- KernelSVM ---

type kernelSVMState struct {
	Kernel    KernelType
	Gamma     float64
	Degree    int
	Coef0     float64
	Lambda    float64
	Epochs    int
	Seed      int64
	Mean, Std []float64
	SV        [][]float64
	Coef      [][]float64
	Classes   []int
	Dim       int
	GammaRes  float64
	DegreeRes int
	Trained   bool
}

// GobEncode implements gob.GobEncoder.
func (m *KernelSVM) GobEncode() ([]byte, error) {
	st := kernelSVMState{
		Kernel: m.Kernel, Gamma: m.Gamma, Degree: m.Degree, Coef0: m.Coef0,
		Lambda: m.Lambda, Epochs: m.Epochs, Seed: m.Seed, SV: m.sv, Coef: m.coef,
		Classes: m.classes, Dim: m.dim, GammaRes: m.gamma, DegreeRes: m.degree,
		Trained: m.trained,
	}
	if m.scaler != nil {
		st.Mean, st.Std = m.scaler.mean, m.scaler.std
	}
	return gobEncode(st)
}

// GobDecode implements gob.GobDecoder.
func (m *KernelSVM) GobDecode(data []byte) error {
	var st kernelSVMState
	if err := gobDecode(data, &st); err != nil {
		return err
	}
	m.Kernel, m.Gamma, m.Degree, m.Coef0 = st.Kernel, st.Gamma, st.Degree, st.Coef0
	m.Lambda, m.Epochs, m.Seed = st.Lambda, st.Epochs, st.Seed
	m.sv, m.coef, m.classes, m.dim = st.SV, st.Coef, st.Classes, st.Dim
	m.gamma, m.degree, m.trained = st.GammaRes, st.DegreeRes, st.Trained
	if st.Mean != nil {
		m.scaler = &scaler{mean: st.Mean, std: st.Std}
	}
	return nil
}
