package optflow

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// floTag is the sanity-check value ("PIEH" reinterpreted as a little-endian
// float32) that opens every Middlebury .flo file.
const floTag float32 = 202021.25

// WriteFlow serialises a FlowField to w in the Middlebury .flo format: the ASCII
// tag "PIEH" (the float32 202021.25), then the width and height as little-endian
// int32, then Rows*Cols interleaved (u, v) pairs as little-endian float32, in
// row-major order. This is the exact on-disk layout produced by the reference
// Middlebury tools and read back by [ReadFlow].
//
// Because the format stores single-precision floats, values are narrowed from
// the field's float64 samples to float32 on write. flow must be non-empty. Any
// write error from w is returned.
func WriteFlow(w io.Writer, flow *FlowField) error {
	if flow == nil || flow.Rows <= 0 || flow.Cols <= 0 {
		return fmt.Errorf("optflow: WriteFlow requires a non-empty flow field")
	}
	bw := bufio.NewWriter(w)
	if err := binary.Write(bw, binary.LittleEndian, floTag); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, int32(flow.Cols)); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, int32(flow.Rows)); err != nil {
		return err
	}
	// Emit samples through a reusable buffer to avoid per-value reflection cost.
	buf := make([]float32, flow.Cols*2)
	for y := 0; y < flow.Rows; y++ {
		for x := 0; x < flow.Cols; x++ {
			i := (y*flow.Cols + x) * 2
			buf[x*2] = float32(flow.Data[i])
			buf[x*2+1] = float32(flow.Data[i+1])
		}
		if err := binary.Write(bw, binary.LittleEndian, buf); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// ReadFlow deserialises a Middlebury .flo stream written by [WriteFlow] (or any
// conforming tool) from r and returns the reconstructed FlowField. It validates
// the "PIEH" tag and the dimension header, then reads Rows*Cols interleaved
// (u, v) float32 pairs, widening them to the field's float64 samples.
//
// An error is returned for a bad tag, non-positive or implausibly large
// dimensions, or a truncated stream.
func ReadFlow(r io.Reader) (*FlowField, error) {
	br := bufio.NewReader(r)
	var tag float32
	if err := binary.Read(br, binary.LittleEndian, &tag); err != nil {
		return nil, fmt.Errorf("optflow: ReadFlow reading tag: %w", err)
	}
	if tag != floTag {
		return nil, fmt.Errorf("optflow: ReadFlow bad magic tag %v, want %v", tag, floTag)
	}
	var w, h int32
	if err := binary.Read(br, binary.LittleEndian, &w); err != nil {
		return nil, fmt.Errorf("optflow: ReadFlow reading width: %w", err)
	}
	if err := binary.Read(br, binary.LittleEndian, &h); err != nil {
		return nil, fmt.Errorf("optflow: ReadFlow reading height: %w", err)
	}
	if w <= 0 || h <= 0 || w > 1<<20 || h > 1<<20 {
		return nil, fmt.Errorf("optflow: ReadFlow implausible dimensions %dx%d", w, h)
	}
	flow := NewFlowField(int(h), int(w))
	buf := make([]float32, int(w)*2)
	for y := 0; y < int(h); y++ {
		if err := binary.Read(br, binary.LittleEndian, buf); err != nil {
			return nil, fmt.Errorf("optflow: ReadFlow reading row %d: %w", y, err)
		}
		for x := 0; x < int(w); x++ {
			i := (y*int(w) + x) * 2
			flow.Data[i] = float64(buf[x*2])
			flow.Data[i+1] = float64(buf[x*2+1])
		}
	}
	return flow, nil
}

// WriteOpticalFlow writes a FlowField to the named file in the Middlebury .flo
// format (see [WriteFlow]). The file is created or truncated. It is the file
// counterpart of OpenCV's cv::optflow::writeOpticalFlow.
//
// flow must be non-empty. The file is always closed; a close error is reported
// if no earlier write error occurred.
func WriteOpticalFlow(path string, flow *FlowField) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()
	return WriteFlow(f, flow)
}

// ReadOpticalFlow reads a Middlebury .flo file (see [ReadFlow]) from the named
// path and returns the reconstructed FlowField. It is the file counterpart of
// OpenCV's cv::optflow::readOpticalFlow.
func ReadOpticalFlow(path string) (*FlowField, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadFlow(f)
}
