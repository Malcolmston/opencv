package cudacore_test

import (
	"testing"

	"github.com/malcolmston/opencv/cudacore"
)

func TestStreamNoOps(t *testing.T) {
	s := cudacore.NewStream()
	s.WaitForCompletion()
	if !s.QueryIfComplete() {
		t.Fatalf("no-op stream should always be complete")
	}
}

func TestEventNoOps(t *testing.T) {
	s := cudacore.NewStream()
	start := cudacore.NewEvent()
	end := cudacore.NewEvent()

	start.Record(s)
	cudacore.StreamWaitEvent(s, start)
	end.Record(s)
	end.WaitForCompletion()

	if !start.QueryIfComplete() || !end.QueryIfComplete() {
		t.Fatalf("no-op events should always be complete")
	}
	if ms := cudacore.ElapsedTime(start, end); ms != 0 {
		t.Fatalf("ElapsedTime = %v, want 0", ms)
	}
}
