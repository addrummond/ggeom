package ggeom

import (
	"fmt"
	"math/big"
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

func r(i float64) big.Rat {
	var r big.Rat
	r.SetFloat64(i)
	return r
}

func TestIsBetweenAnticlockwise(t *testing.T) {
	false_cases := [][]Vec2{
		{{r(-1), r(1)}, {r(0), r(1)}, {r(1), r(1)}},
		{{r(-1), r(0)}, {r(0), r(1)}, {r(1), r(0)}},
		{{r(-0.01), r(999)}, {r(0.0001), r(10)}, {r(0.01), r(1000)}},
	}
	true_cases := [][]Vec2{
		{{r(-30), r(-20)}, {r(-1), r(-10)}, {r(2), r(-40)}},
		{{r(-1000), r(0)}, {r(0), r(-1000)}, {r(1000), r(0)}},
		{{r(-0.01), r(-999)}, {r(0.0001), r(-10)}, {r(0.01), r(-1000)}},
	}
	true_irreversible_cases := [][]Vec2{
		{{r(0), r(1)}, {r(0), r(1)}, {r(0), r(1)}},
		{{r(0), r(1)}, {r(0), r(2)}, {r(0), r(3)}},
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

	p1 := Polygon2{verts: []Vec2{{r(10), r(10)}, {r(-10), r(10)}, {r(-10), r(-10)}, {r(10), r(-10)}, {r(0), r(-5)}}}
	p2 := Polygon2{verts: []Vec2{{r(0), r(2)}, {r(-1), r(0)}, {r(1), r(0)}}}

	cs := GetConvolutionCycle(&p1, &p2)

	fmt.Printf("P1: %+v\n\nP2: %+v\n\nConv: %+v\n", p1, p2, cs)

	svgout, _ := os.Create("out.svg")
	canvas := svg.New(svgout)
	debugDrawLineStrips(canvas, [][]Vec2{p1.verts, cs}, 20, []string{"stroke: black; stroke-width: 4; fill: none", "stroke: red; stroke-width: 1; fill: none"})
}
