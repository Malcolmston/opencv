package cudaimgproc

import cv "github.com/malcolmston/opencv"

// TemplateMatching is a CPU-backed template matcher, mirroring
// cv::cuda::TemplateMatching. Create one with [CreateTemplateMatching] and run
// it with [TemplateMatching.Match].
type TemplateMatching struct {
	mode cv.TemplateMatchMode
}

// CreateTemplateMatching returns a [TemplateMatching] configured with the given
// similarity mode, mirroring cuda::createTemplateMatching. The supported modes
// are [cv.TmSqdiff], [cv.TmSqdiffNormed], [cv.TmCcoeff] and [cv.TmCcoeffNormed].
func CreateTemplateMatching(mode cv.TemplateMatchMode) *TemplateMatching {
	return &TemplateMatching{mode: mode}
}

// Match slides templ over src and returns a [cv.FloatMat] of similarity scores
// of shape (src.Rows-templ.Rows+1)×(src.Cols-templ.Cols+1); use [cv.MinMaxLoc]
// to locate the best match. Both inputs must have the same channel count and
// templ must fit inside src. The trailing Stream argument is accepted and
// ignored.
func (t *TemplateMatching) Match(src, templ GpuMat, streams ...Stream) *cv.FloatMat {
	_ = firstStream(streams)
	s := src.requireHost("TemplateMatching.Match")
	tm := templ.requireHost("TemplateMatching.Match")
	return cv.MatchTemplate(s, tm, t.mode)
}
