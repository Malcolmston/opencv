package dnn_superres

import "errors"

// errIterations is returned when a refinement count is below 1.
var errIterations = errors.New("dnn_superres: iterations must be >= 1")
