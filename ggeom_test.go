package ggeom

import (
	"fmt"
	"math"
	"os"
	"testing"

	svg "github.com/ajstarks/svgo/float"
)

const SvgWidth = 800
const SvgHeight = 800

func debugDrawLineStrips(canvas *svg.SVG, strips [][]Vec2, formats []string) {
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
	ws := SvgWidth / width
	hs := SvgHeight / height
	scale := math.Min(ws, hs)

	width *= scale
	height *= scale

	canvas.Start(width, height)

	tx := func(x float64) float64 { return (x - minx + arrowLen) * scale }
	ty := func(y float64) float64 { return height - ((y - miny + arrowLen) * scale) }

	// origin marker
	canvas.Square(tx(-arrowLen/4), ty(arrowLen/4), arrowLen*scale*0.5, "fill: green")

	for si, s := range strips {
		xs := make([]float64, 0)
		ys := make([]float64, 0)
		fmti := si
		if fmti >= len(formats) {
			fmti = len(formats) - 1
		}
		format := formats[fmti]

		for i := 0; len(s) > 0 && i <= len(s); i++ {
			ii := i
			if ii >= len(s) {
				ii = 0
			}

			p := s[ii]

			xs = append(xs, tx(p.ApproxX()))
			ys = append(ys, ty(p.ApproxY()))
		}

		if len(xs) > 0 {
			canvas.Polyline(xs, ys, format)
		}
	}

	for si, s := range strips {
		if len(s) == 0 {
			continue
		}

		fmti := si
		if fmti >= len(formats) {
			fmti = len(formats) - 1
		}
		format := formats[fmti]

		sp := s[0]
		canvas.Square(tx(sp.ApproxX()-(arrowLen/8)), ty(sp.ApproxY()+(arrowLen/8)), arrowLen*scale*0.25, format)

		for i := 1; i <= len(s); i++ {
			p1 := s[i-1]
			p2 := s[i%len(s)]
			d := p2.Sub(p1)
			if d.x.Sign() == 0 && d.y.Sign() == 0 {
				continue
			}

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
		{{1, 0}, {1, 0}, {0, 1}},
		{{-1.2, 0}, {0, 1}, {0, -1.2}},
		{{0, -1.2}, {0, 1}, {1.2, 0}},
	})
	trueCases := SofSofVec2([][][]float64{
		{{0, -4}, {1.2, 0}, {4, 0}},
		{{0, -1.2}, {4, 0}, {1.2, 0}},
		{{4, 0}, {0, 1.2}, {0, 1}},
		{{1.2, 0}, {0, 1}, {0, 1.2}},
		{{0, 1}, {-1.2, 0}, {-3, 0}},
		{{0, 1.2}, {-3, 0}, {-1.2, 0}},
		{{-3, 0}, {0, -1.2}, {0, 1}},
		{{1, 0}, {0, 1}, {0, 1}},
		{{-30, -20}, {-1, -10}, {2, -40}},
		{{-1000, 0}, {0, -1000}, {1000, 0}},
		{{-0.01, -999}, {0.0001, -1000}, {0.01, -1000}},
		{{1, 0}, {0, 0.5}, {1, -1}},
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

func TestNondegenerateSegmentIntersection(t *testing.T) {
	tests := SofSofVec2([][][]float64{
		{{-1, -1}, {1, 1}, {-1, 1}, {1, -1}, {0, 0}},            // A cross centered on zero
		{{-1, 0}, {1, 2}, {-1, 2}, {1, 0}, {0, 1}},              // The case above translated up one unit
		{{-1, 2}, {1, 0}, {-1, 0}, {1, 2}, {0, 1}},              // The case above with points swapped.
		{{-1, 0}, {1, 0}, {-0.5, 10}, {-0.5, -10}, {-0.5, 0}},   // Vertical line intersecting horizontal line
		{{-1, -1}, {1, -1}, {-0.5, 9}, {-0.5, -11}, {-0.5, -1}}, // The case above translated down one unit
		{{-0.5, -10}, {-0.5, 0}, {-1, 0}, {1, 0}, {-0.5, 0}},    // Horizontal line intersecting with vertical line
	})

	for _, tst := range tests {
		p := NondegenerateSegmentIntersection(&tst[0], &tst[1], &tst[2], &tst[3])
		//fmt.Printf("INT: %f %f\n", p.ApproxX(), p.ApproxY())
		if !p.Eq(&tst[4]) {
			t.Error()
		}
	}
}

func TestGetSegmentIntersectionInfo(t *testing.T) {
	const NONE = -999
	const NOT_UNIQUE = -9999

	var VNONE Vec2
	VNONE.x.SetFloat64(NONE)
	VNONE.y.SetFloat64(NONE)

	var VNOT_UNIQUE Vec2
	VNOT_UNIQUE.x.SetFloat64(NOT_UNIQUE)
	VNOT_UNIQUE.y.SetFloat64(NOT_UNIQUE)

	tests := SofSofVec2([][][]float64{
		{{-1, -1}, {1, 1}, {-1, 1}, {1, -1}, {0, 0}},                     // A cross centered on zero
		{{-1, 0}, {1, 2}, {-1, 2}, {1, 0}, {0, 1}},                       // The case above translated up one unit
		{{-1, 2}, {1, 0}, {-1, 0}, {1, 2}, {0, 1}},                       // The case above with points swapped.
		{{-1, 0}, {1, 0}, {-0.5, 10}, {-0.5, -10}, {-0.5, 0}},            // Vertical line intersecting horizontal line
		{{-1, -1}, {1, -1}, {-0.5, 9}, {-0.5, -11}, {-0.5, -1}},          // The case above translated down one unit
		{{-0.5, -10}, {-0.5, 0}, {-1, 0}, {1, 0}, {-0.5, 0}},             // Horizontal line intersecting with vertical line
		{{-1, 1}, {1, 1}, {-0.5, 1}, {0.5, 1}, {NOT_UNIQUE, NOT_UNIQUE}}, // One horizontal line that completely overlaps another
		{{1, -1}, {1, 1}, {1, -0.5}, {1, 0.5}, {NOT_UNIQUE, NOT_UNIQUE}}, // One vertical line that completely overlaps another
		{{-1, -2}, {1, 2}, {1, 2}, {2, 4}, {1, 2}},                       // Two adjacent diagonal lines
		{{1, 2}, {-1, -2}, {-1, -2}, {-2, -4}, {-1, -2}},                 // Two adjacent diagonal lines
		{{1, 2}, {5, 2}, {5, 2}, {7, 2}, {5, 2}},                         // Two adjacent horizontal lines
		{{2, 1}, {2, 5}, {2, 5}, {2, 7}, {2, 5}},                         // Two adjacent vertical lines
		{{-1, -1}, {2, 2}, {-11, -12}, {1, -2}, {NONE, NONE}},            // Non-intersecting non-parallel
		{{-2, -4}, {2, 4}, {-3, -5}, {1, 3}, {NONE, NONE}},               // Non-intersecting parallel
		{{3, 0}, {-2, 2}, {2, -2}, {2, 2}, {2, -2}},                      // A random case of interest
	})

	for i, tst := range tests {
		info := GetSegmentIntersectionInfo(&tst[0], &tst[1], &tst[2], &tst[3])
		if info.intersect {
			if info.unique {
				if !info.p.Eq(&tst[4]) {
					t.Error()
				}
			} else {
				if !tst[4].Eq(&VNOT_UNIQUE) {
					t.Error()
				}
			}
			fmt.Printf("Intersection point computed for test %v: (%v,%v)\n", i, info.p.ApproxX(), info.p.ApproxY())
		} else {
			if !tst[4].Eq(&VNONE) {
				t.Error()
			}
		}
	}
}

const EPSILON = 0.000001

func TestSegmentLoopIntersections(t *testing.T) {
	tests := SofSofVec2([][][]float64{
		{{-3, 4}, {-1, -2}, {2, 1}, {-3, 1}}, // polygon
		{{-2, 1}}, // intersection points
		/////
		{{0, 1}, {-1, 1}, {-1, -1}, {1, -1}, {1, 0.5}, {-2, 0.5}, {-2, -2}, {-0.5, -2}, {-0.5, 2}},
		{{-1, 0.5}, {-0.5, -1}, {-0.5, 0.5}, {-0.5, 1}},
		/////
		{{0, 1}, {-1.01, 1}, {-1.02, -1}, {1.03, -1}, {1.04, 0.5}, {-2.05, 0.5}, {-2.06, -2}, {-0.57, -2}, {-0.58, 2}},
		{{-1.0125, 0.5}, {-0.5774999999999999, 1}, {-0.5725, -1}, {-0.5762499999999999, 0.5}},
		/////
		{{-5, 0}, {-4, -1}, {-2, 1}, {0, -1}, {4, 1}, {6, -1}},
		{{-3.1666666666666665, -0.16666666666666666}, {-0.6, -0.4}, {0.9230769230769231, -0.5384615384615384}},
		/////
		{{-2, 2}, {2, -2}, {2, 2}, {-2, -2}},
		{{0, 0}},
		/////
		{{-2, 2}, {2, -2}, {2, 2}, {-2, -2}, {-2, 0}, {3, 0}},
		{{0, 0}, {0, 0}, {0, 0}, {2, 0}, {2, -2}},
	})

	for i := 0; i < len(tests); i += 2 {
		ps := tests[i]

		svgout, err := os.Create(fmt.Sprintf("testoutputs/TestSegmentLoopIntersections_figure_%v.svg", i/2))
		if err != nil {
			fmt.Errorf("Error opening SVG: %v\n", err)
		}
		canvas := svg.New(svgout)
		debugDrawLineStrips(canvas, [][]Vec2{ps}, []string{"stroke: black; stroke-width: 4; fill: none"})
		svgout.Close()

		its1 := tests[i+1]
		its2 := SegmentLoopIntersections(ps)

		fmt.Printf("TestSegmentLoopIntersections test %v\n", i/2)
		fmt.Printf("  Expected intersections: ")
		for i, it := range its1 {
			if i != 0 {
				fmt.Printf(";  ")
			}
			xf, _ := it.x.Float64()
			yf, _ := it.y.Float64()
			fmt.Printf("%v, %v ", xf, yf)
		}
		fmt.Printf("\n  Computed intersections:")
		for _, it := range its2 {
			xf, _ := it.p.x.Float64()
			yf, _ := it.p.y.Float64()
			fmt.Printf("\n  %v, %v [%v,%v] of (%v,%v) -> (%v,%v)  with  (%v,%v) -> (%v,%v)\n", xf, yf, it.seg1, it.seg2, ps[it.seg1].ApproxX(), ps[it.seg1].ApproxY(), ps[(it.seg1+1)%len(ps)].ApproxX(), ps[(it.seg1+1)%len(ps)].ApproxY(), ps[it.seg2].ApproxX(), ps[it.seg2].ApproxY(), ps[(it.seg2+1)%len(ps)].ApproxX(), ps[(it.seg2+1)%len(ps)].ApproxY())
		}
		fmt.Printf("\n")

		used := make(map[int]bool)
		for _, i1 := range its1 {
			found := false
			for i, i2 := range its2 {
				if used[i] {
					continue
				}

				if i1.SlowEqEpsilon(&i2.p, EPSILON) {
					found = true
					used[i] = true
					break
				}
			}
			if !found {
				t.Error()
			}
		}
		used = make(map[int]bool)
		for _, i1 := range its2 {
			found := false
			for i, i2 := range its1 {
				if used[i] {
					continue
				}

				if i2.SlowEqEpsilon(&i1.p, EPSILON) {
					found = true
					used[i] = true
					break
				}
			}
			if !found {
				t.Error()
			}
		}
	}
}

/*func TestConvolve(t *testing.T) {
	tests := SofSofVec2([][][]float64{
		{{10, 10}, {-10, 10}, {-10, -10}, {10, -10}},                                  // p
		{{0, 2}, {-1, 0}, {1, 0}},                                                     // q
		{{-11, -10}, {11, -10}, {11, 10}, {10, 12}, {-10, 12}, {-11, 10}, {-11, -10}}, // convolution cycle
		/////
		{{10, 10}, {-10, 10}, {-10, -10}, {10, -10}, {0, -5}},
		{{0, 2}, {-1, 0}, {1, 0}},
		{{-11, -10}, {11, -10}, {10, -8}, {0, -3}, {-1, -5}, {1, -5}, {11, 10}, {10, 12}, {-10, 12}, {-11, 10}, {-11, -10}},
		/////
		{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
		{{0.6, 0.6}, {-0.6, 0.6}, {-0.6, -0.6}, {0.6, -0.6}},
		{{-0.6, -0.6}, {4.6, -0.6}, {4.6, 1.6}, {0.4, 1.6}, {0.4, 0.4}, {1.6, 0.4}, {1.6, 2.6}, {0.4, 2.6}, {0.4, 1.4}, {1.6, 1.4}, {2.6, 2.4}, {2.6, 3.6}, {1.4, 3.6}, {1.4, 2.4}, {3.6, 2.4}, {3.6, 3.6}, {2.4, 3.6}, {2.4, 2.4}, {3.4, 1.4}, {4.6, 1.4}, {4.6, 3.6}, {3.6, 4.6}, {-0.6, 4.6}, {-0.6, -0.6}},
		/////
		{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
		{{0.3, 0.3}, {0, 0.2}, {-0.3, 0.3}, {-0.3, -0.3}, {0.3, -0.3}},
		{{-0.6, -0.6}, {4.6, -0.6}, {4.6, 1.6}, {0.4, 1.6}, {0.4, 0.4}, {1.6, 0.4}, {1.6, 2.6}, {0.4, 2.6}, {0.4, 1.4}, {1.6, 1.4}, {2.6, 2.4}, {2.6, 3.6}, {1.4, 3.6}, {1.4, 2.4}, {3.6, 2.4}, {3.6, 3.6}, {2.4, 3.6}, {2.4, 2.4}, {3.4, 1.4}, {4.6, 1.4}, {4.6, 3.6}, {3.6, 4.6}, {-0.6, 4.6}, {-0.6, -0.6}},
		/////
		{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
		{{0.3, 0.3}, {0, 0.2}, {-0.3, 0.3}, {-0.2, 0}, {-0.3, -0.3}, {0, -0.2}, {0.3, -0.3}},
		{{-0.6, -0.6}, {4.6, -0.6}, {4.6, 1.6}, {0.4, 1.6}, {0.4, 0.4}, {1.6, 0.4}, {1.6, 2.6}, {0.4, 2.6}, {0.4, 1.4}, {1.6, 1.4}, {2.6, 2.4}, {2.6, 3.6}, {1.4, 3.6}, {1.4, 2.4}, {3.6, 2.4}, {3.6, 3.6}, {2.4, 3.6}, {2.4, 2.4}, {3.4, 1.4}, {4.6, 1.4}, {4.6, 3.6}, {3.6, 4.6}, {-0.6, 4.6}, {-0.6, -0.6}},
		/////
		{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
		{{0.6, 0.6}, {0, 0.4}, {-0.6, 0.6}, {-0.4, 0}, {-0.6, -0.6}, {0, -0.4}, {0.6, -0.6}},
		{{-0.6, -0.6}, {4.6, -0.6}, {4.6, 1.6}, {0.4, 1.6}, {0.4, 0.4}, {1.6, 0.4}, {1.6, 2.6}, {0.4, 2.6}, {0.4, 1.4}, {1.6, 1.4}, {2.6, 2.4}, {2.6, 3.6}, {1.4, 3.6}, {1.4, 2.4}, {3.6, 2.4}, {3.6, 3.6}, {2.4, 3.6}, {2.4, 2.4}, {3.4, 1.4}, {4.6, 1.4}, {4.6, 3.6}, {3.6, 4.6}, {-0.6, 4.6}, {-0.6, -0.6}},
	})

	for i := 0; i < len(tests); i += 3 {
		p := Polygon2{verts: tests[i]}
		q := Polygon2{verts: tests[i+1]}
		expected := tests[i+2]

		cs := GetConvolutionCycle(&p, &q)
		fmt.Printf("The expected (left) vs. computed conv cycle for test %v:\n", i/3)
		l := len(cs)
		if len(expected) > l {
			l = len(expected)
		}

		itns := SegmentLoopIntersections(cs)

		for i := 0; i < l; i++ {
			fmt.Printf("    ")
			if i < len(expected) {
				fmt.Printf("%+2.2f, %+2.2f\t", expected[i].ApproxX(), expected[i].ApproxY())
			} else {
				fmt.Printf("_____\t")
			}
			if i < len(cs) {
				fmt.Printf("%+2.2f, %+2.2f\t", cs[i].ApproxX(), cs[i].ApproxY())
			} else {
				fmt.Printf("_____\t")
			}
			fmt.Printf("\n")

			if i >= len(cs) || i >= len(expected) || !expected[i].SlowEqEpsilon(&cs[i], EPSILON) {
				t.Fail()
			}
		}
		fmt.Printf("\n")

		fmt.Printf("Intersections\n")
		for _, itn := range itns {
			fmt.Printf("    (%v,%v) -> (%v,%v)  with  (%v,%v) -> (%v,%v)  at (%v,%v)\n", cs[itn.seg1].ApproxX(), cs[itn.seg1].ApproxY(), cs[(itn.seg1+1)%len(cs)].ApproxX(), cs[(itn.seg1+1)%len(cs)].ApproxY(), cs[itn.seg2].ApproxX(), cs[itn.seg2].ApproxY(), cs[(itn.seg2+1)%len(cs)].ApproxX(), cs[(itn.seg2+1)%len(cs)].ApproxY(), itn.p.ApproxX(), itn.p.ApproxY())
		}

		crosses := make([][]Vec2, 0)
		for _, itn := range itns {
			crosses = append(crosses, SofVec2([][]float64{{itn.p.ApproxX(), itn.p.ApproxY()}, {itn.p.ApproxX(), itn.p.ApproxY()}}))
		}

		svgout, _ := os.Create(fmt.Sprintf("testoutputs/TestConvolve_figure_%v.svg", i/3))
		canvas := svg.New(svgout)
		strips := [][]Vec2{p.verts, cs}
		strips = append(strips, crosses...)
		debugDrawLineStrips(canvas, strips, []string{"stroke: black; stroke-width: 4; fill: none", "stroke: red; fill: red; stroke-width: 1; fill: none", "stroke: green; fill: green; stroke-width: 1"})
	}
}*/
