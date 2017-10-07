package ggeom

import (
	"fmt"
	"math"
	"os"
	"testing"

	svg "github.com/ajstarks/svgo/float"
)

func debugDrawLineStrips(canvas *svg.SVG, strips [][]Vec2, scale float64, formats []string) {
	var minx, miny, maxx, maxy float64 = strips[0][0].ApproxX(), float64(strips[0][0].ApproxY()), float64(strips[0][0].ApproxX()), float64(strips[0][0].ApproxY())
	for _, s := range strips {
		for _, p := range s {
			if p.ApproxX() < minx {
				minx = p.ApproxX()
			} else if p.ApproxX() > maxx {
				maxx = p.ApproxX()
			}
			if p.ApproxY() < miny {
				miny = p.ApproxY()
			} else if p.ApproxY() > maxy {
				maxy = p.ApproxY()
			}
		}
	}

	width := maxx - minx
	height := maxy - miny
	arrowLen := (width + height) * 0.02
	width += arrowLen * 2
	height += arrowLen * 2
	width *= scale
	height *= scale

	canvas.Start(width, height)

	tx := func(x float64) float64 { return (x - minx + arrowLen) * scale }
	ty := func(y float64) float64 { return height - ((y - miny + arrowLen) * scale) }

	for si, s := range strips {
		xs := make([]float64, 0)
		ys := make([]float64, 0)
		format := formats[si]

		for i := 0; i <= len(s); i++ {
			p := s[i%len(s)]

			xs = append(xs, tx(p.ApproxX()))
			ys = append(ys, ty(p.ApproxY()))
		}

		canvas.Polyline(xs, ys, format)
	}

	for si, s := range strips {
		format := formats[si]

		sp := s[0]
		canvas.Square(tx(sp.ApproxX()-(arrowLen/2)), ty(sp.ApproxY()-(arrowLen/4)), arrowLen*scale*0.5, "fill: black")

		for i := 1; i < len(s); i++ {
			p1 := s[i-1]
			p2 := s[i%len(s)]
			d := p2.Sub(p1)

			const sinv = 0.309
			const cosv = -0.951
			dx := d.ApproxX()
			dy := d.ApproxY()
			d1 := ApproxVec2(dx*cosv-dy*sinv, dx*sinv+dy*cosv).ApproxScale(arrowLen)
			d2 := ApproxVec2(dx*cosv+dy*sinv, dx*-sinv+dy*cosv).ApproxScale(arrowLen)
			h1 := p2.Add(d1)
			h2 := p2.Add(d2)
			canvas.Line(tx(p2.ApproxX()), ty(p2.ApproxY()), tx(h1.ApproxX()), ty(h1.ApproxY()), format)
			canvas.Line(tx(p2.ApproxX()), ty(p2.ApproxY()), tx(h2.ApproxX()), ty(h2.ApproxY()), format)
		}
	}

	canvas.End()
}

func TestIsBetweenAnticlockwise(t *testing.T) {
	falseCases := SofSofVec2([][][]float64{
		{{-1, 1}, {0, 1}, {1, 1}},
		{{-1, 0}, {0, 1}, {1, 0}},
		{{-0.01, 999}, {0.0001, 1023}, {0.01, 1000}},
	})
	trueCases := SofSofVec2([][][]float64{
		{{-30, -20}, {-1, -10}, {2, -40}},
		{{-1000, 0}, {0, -1000}, {1000, 0}},
		{{-0.01, -999}, {0.0001, -1000}, {0.01, -1000}},
	})
	trueIrreversibleCases := SofSofVec2([][][]float64{
		{{0, 1}, {0, 1}, {0, 1}},
		{{0, 1}, {0, 2}, {0, 3}},
	})

	for _, c := range falseCases {
		if IsBetweenAnticlockwise(c[0], c[1], c[2]) {
			t.Error()
		}
		if !IsBetweenAnticlockwise(c[2], c[1], c[0]) {
			t.Error()
		}
	}

	for _, c := range trueCases {
		if !IsBetweenAnticlockwise(c[0], c[1], c[2]) {
			t.Error()
		}
		if IsBetweenAnticlockwise(c[2], c[1], c[0]) {
			t.Error()
		}
	}

	for _, c := range trueIrreversibleCases {
		if !IsBetweenAnticlockwise(c[0], c[1], c[2]) {
			t.Error()
		}
	}
}

func TestFastSegmentsDontIntersect(t *testing.T) {
	trueTests := SofSofVec2([][][]float64{
		// Two parallel non-colinear vertical lines
		{{-1, 1}, {-1, 0}, {1, 1}, {1, 0}},
	})

	falseTests := SofSofVec2([][][]float64{
		// Two parallel non-colinear vertical lines with too-big coords
		{{-1, math.MaxFloat64}, {-1, 0}, {1, math.MaxFloat64}, {1, 0}},
		// Two parallel non-colinear diagonal lines with overlapping bounding rects
		{{-1, -2}, {1, 2}, {-1.1, -2}, {0.9, 2}},
	})

	for _, vs := range trueTests {
		if !FastSegmentsDontIntersect(&vs[0], &vs[1], &vs[2], &vs[3]) {
			t.Error()
		}
	}
	for _, vs := range falseTests {
		if FastSegmentsDontIntersect(&vs[0], &vs[1], &vs[2], &vs[3]) {
			t.Error()
		}
	}
}

func TestSegmentsIntersect(t *testing.T) {
	falseTests := SofSofVec2([][][]float64{
		// Two parallel non-colinear vertical lines
		{{-1, 1}, {-1, 0}, {1, 1}, {1, 0}},
		// Two parallel non-colinear vertical lines with big coords
		{{-1, math.MaxFloat64}, {-1, 0}, {1, math.MaxFloat64}, {1, 0}},
		// Two parallel non-colinear diagonal lines with overlapping bounding rects
		{{-1, -2}, {1, 2}, {-1.1, -2}, {0.9, 2}},
	})

	trueTests := SofSofVec2([][][]float64{
		// A cross.
		{{-10, -10}, {10, 10}, {-10, 10}, {10, -10}},
		// As above but with segments pointing the other way.
		{{10, 10}, {-10, -10}, {10, -10}, {-10, 10}},
		// Only barely intersect.
		{{-5, 0}, {5, math.SmallestNonzeroFloat64}, {-5, math.SmallestNonzeroFloat64}, {5, 0}},
		// A T
		{{-100, 5}, {100, 5}, {1, 5}, {1, -0.001}},
		// Colinear with overlap
		{{-100, 5}, {100, 5}, {90, 5}, {101, 5}},
		// Colinear with overlap
		{{5, -100}, {5, 100}, {5, 90}, {5, 101}},
		// Colinear with overlap
		{{-1, -1}, {1, 1}, {0.5, 0.5}, {2, 2}},
		// Colinear joned at tip.
		{{-100, 5}, {100, 5}, {100, 5}, {101, 5}},
	})

	for _, vs := range trueTests {
		if !SegmentsIntersect(&vs[0], &vs[1], &vs[2], &vs[3]) {
			t.Error()
		}
	}
	for _, vs := range falseTests {
		if SegmentsIntersect(&vs[0], &vs[1], &vs[2], &vs[3]) {
			t.Error()
		}
	}
}

func TestConvolve(t *testing.T) {
	//p1 := Polygon2{verts: []Vec2{{10, 10}, {-10, 10}, {-10, -10}, {10, -10}}}

	p1 := Polygon2{verts: []Vec2{{r(10), r(10)}, {r(-10), r(10)}, {r(-10), r(-10)}, {r(10), r(-10)}, {r(0), r(-5)}}}
	p2 := Polygon2{verts: []Vec2{{r(0), r(2)}, {r(-1), r(0)}, {r(1), r(0)}}}

	cs := GetConvolutionCycle(&p1, &p2)

	fmt.Printf("P1: %+v\n\nP2: %+v\n\nConv: %+v\n", p1, p2, cs)

	svgout, _ := os.Create("out.svg")
	canvas := svg.New(svgout)
	debugDrawLineStrips(canvas, [][]Vec2{p1.verts, cs}, 20, []string{"stroke: black; stroke-width: 4; fill: none", "stroke: red; stroke-width: 1; fill: none"})
}
