package ggeom

import (
	"fmt"
	"os"
	"testing"

	svg "github.com/ajstarks/svgo/float"
)

func debugDrawLineStrips(canvas *svg.SVG, strips [][]Vec2, scale float64, formats []string) {
	var minx, miny, maxx, maxy float64 = float64(strips[0][0].x), float64(strips[0][0].y), float64(strips[0][0].x), float64(strips[0][0].y)
	for _, s := range strips {
		for _, p := range s {
			if float64(p.x) < minx {
				minx = float64(p.x)
			} else if float64(p.x) > maxx {
				maxx = float64(p.x)
			}
			if float64(p.y) < miny {
				miny = float64(p.y)
			} else if float64(p.y) > maxy {
				maxy = float64(p.y)
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

	tx := func(x Scalar) float64 { return (float64(x) - minx + arrowLen) * scale }
	ty := func(y Scalar) float64 { return height - ((float64(y) - miny + arrowLen) * scale) }

	for si, s := range strips {
		xs := make([]float64, 0)
		ys := make([]float64, 0)
		format := formats[si]

		for i := 0; i <= len(s); i++ {
			p := s[i%len(s)]

			xs = append(xs, tx(p.x))
			ys = append(ys, ty(p.y))
		}

		canvas.Polyline(xs, ys, format)
	}

	for si, s := range strips {
		format := formats[si]

		sp := s[0]
		canvas.Square(tx(sp.x-Scalar(arrowLen/2)), ty(sp.y-Scalar(arrowLen/4)), arrowLen*scale*0.5, "fill: black")

		for i := 1; i < len(s); i++ {
			p1 := s[i-1]
			p2 := s[i%len(s)]
			d := p2.Sub(p1)

			const sinv = Scalar(0.309)
			const cosv = Scalar(-0.951)
			d1 := Vec2{d.x*cosv - d.y*sinv, d.x*sinv + d.y*cosv}.Scale(Scalar(arrowLen))
			d2 := Vec2{d.x*cosv + d.y*sinv, d.x*-sinv + d.y*cosv}.Scale(Scalar(arrowLen))
			h1 := p2.Add(d1)
			h2 := p2.Add(d2)
			canvas.Line(tx(p2.x), ty(p2.y), tx(h1.x), ty(h1.y), format)
			canvas.Line(tx(p2.x), ty(p2.y), tx(h2.x), ty(h2.y), format)
		}
	}

	canvas.End()
}

func TestIsBetweenAnticlockwise(t *testing.T) {
	false_cases := [][]Vec2{
		{{-1, 1}, {0, 1}, {1, 1}},
		{{-1, 0}, {0, 1}, {1, 0}},
		{{-0.01, 999}, {0.0001, 10}, {0.01, 1000}},
	}
	true_cases := [][]Vec2{
		{{-30, -20}, {-1, -10}, {2, -40}},
		{{-1000, 0}, {0, -1000}, {1000, 0}},
		{{-0.01, -999}, {0.0001, -10}, {0.01, -1000}},
	}
	true_irreversible_cases := [][]Vec2{
		{{0, 1}, {0, 1}, {0, 1}},
		{{0, 1}, {0, 2}, {0, 3}},
	}

	for _, c := range false_cases {
		if IsBetweenAnticlockwise(c[0], c[1], c[2]) {
			t.Error()
		}
		if !IsBetweenAnticlockwise(c[2], c[1], c[0]) {
			t.Error()
		}
	}

	for _, c := range true_cases {
		if !IsBetweenAnticlockwise(c[0], c[1], c[2]) {
			t.Error()
		}
		if IsBetweenAnticlockwise(c[2], c[1], c[0]) {
			t.Error()
		}
	}

	for _, c := range true_irreversible_cases {
		if !IsBetweenAnticlockwise(c[0], c[1], c[2]) {
			t.Error()
		}
	}
}

func TestConvolve(t *testing.T) {
	//p1 := Polygon2{verts: []Vec2{{10, 10}, {-10, 10}, {-10, -10}, {10, -10}}}
	p1 := Polygon2{verts: []Vec2{{10, 10}, {-10, 10}, {-10, -10}, {10, -10}, {0, -5}}}
	p2 := Polygon2{verts: []Vec2{{0, 2}, {-1, 0}, {1, 0}}}

	cs := GetConvolutionCycle(&p1, &p2)

	fmt.Printf("P1: %+v\n\nP2: %+v\n\nConv: %+v\n", p1, p2, cs)

	svgout, _ := os.Create("out.svg")
	canvas := svg.New(svgout)
	debugDrawLineStrips(canvas, [][]Vec2{p1.verts, cs}, 20, []string{"stroke: black; stroke-width: 4; fill: none", "stroke: red; stroke-width: 1; fill: none"})
}
