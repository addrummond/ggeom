package ggeom

import (
	"fmt"
	"math"
	"os"
)

type Scalar float64

type Vec2 struct {
	x Scalar
	y Scalar
}

type Segment2 struct {
	from Vec2
	to   Vec2
}

type Polygon2 struct {
	verts []Vec2
}

func (a Vec2) Add(b Vec2) Vec2 {
	return Vec2{x: a.x + b.x, y: a.y + b.y}
}
func (a Vec2) Sub(b Vec2) Vec2 {
	return Vec2{x: a.x - b.x, y: a.y - b.y}
}

func (a Vec2) Dot(b Vec2) Scalar {
	return a.x*b.x + a.y*b.y
}

func (a Vec2) Det(b Vec2) Scalar {
	return a.x*b.y - a.y*b.x
}

func (a Vec2) Length() Scalar {
	return Scalar(math.Sqrt(float64(a.x*a.x) + float64(a.y*a.y)))
}

func (a Vec2) Scale(v Scalar) Vec2 {
	l := Scalar(math.Sqrt(float64(a.x*a.x) + float64(a.y*a.y)))
	return Vec2{(a.x * v) / l, (a.y * v) / l}
}

// Uses y,x ordering
func (p *Polygon2) IndexOfBottommost() int {
	var minx, miny Scalar = p.verts[0].x, p.verts[0].y
	var mini int
	for i, vert := range p.verts[1:] {
		if vert.y < miny {
			mini = i + 1
			miny = vert.y
			minx = vert.x
		} else if vert.y == miny && vert.x < minx {
			mini = i + 1
			minx = vert.x
		}
	}
	return mini
}

// True iff b is reached before c going anticlockwise from a
func IsBetweenAnticlockwise(a Vec2, b Vec2, c Vec2) bool {
	// See AndyG's answer to https://stackoverflow.com/questions/13640931/how-to-determine-if-a-vector-is-between-two-other-vectors
	return a.Det(b) >= 0 && b.Det(c) >= 0
}

type Label struct {
	p1a, p1b, p2a, p2b int
}

func GetConvolutionCycle(p *Polygon2, q *Polygon2) ([]Vec2, map[Label]bool) {
	return getConvolutionCycleGivenStartingVertices(p, p.IndexOfBottommost(), q, q.IndexOfBottommost())
}

func getConvolutionCycleGivenStartingVertices(p *Polygon2, i0 int, q *Polygon2, j0 int) ([]Vec2, map[Label]bool) {
	points := make([]Vec2, 0, len(p.verts)+len(q.verts))
	usedLabels := make(map[Label]bool)
	i := i0
	j := j0
	s := p.verts[i].Add(q.verts[j])

	points = append(points, s)

	for {
		ip1 := (i + 1) % len(p.verts)
		im1 := (i - 1)
		if im1 < 0 {
			im1 += len(p.verts)
		}
		jp1 := (j + 1) % len(q.verts)
		jm1 := (j - 1)
		if jm1 < 0 {
			jm1 += len(q.verts)
		}

		piTOpiplus1 := p.verts[ip1].Sub(p.verts[i])
		qjminus1TOqj := q.verts[j].Sub(q.verts[jm1])
		qjTOqjplus1 := q.verts[jp1].Sub(q.verts[j])
		piminus1TOpi := p.verts[i].Sub(p.verts[im1])
		incp := IsBetweenAnticlockwise(qjminus1TOqj, piTOpiplus1, qjTOqjplus1)
		incq := IsBetweenAnticlockwise(piminus1TOpi, qjTOqjplus1, piTOpiplus1)

		if !(incp || incq) {
			fmt.Fprintf(os.Stderr, "No segment could be added in a loop iteration in 'computeConvolutionCycle'; this is a bug.\n%+v <- %+v <- %+v\n%+v <- %+v <- %+v\n", qjminus1TOqj, piTOpiplus1, qjTOqjplus1, piminus1TOpi, qjTOqjplus1, piTOpiplus1)
			panic("Panicked due to bug in 'computeConvolutionCycle'")
		}

		if incp {
			t := p.verts[ip1].Add(q.verts[j])
			points = append(points, t)
			usedLabels[Label{p1a: i, p1b: i + 1, p2a: j, p2b: -1}] = true
			s = t
			i = ip1
		}
		if incq {
			t := p.verts[i].Add(q.verts[jp1])
			points = append(points, t)
			usedLabels[Label{p1a: i, p1b: -1, p2a: j, p2b: j + 1}] = true
			s = t
			j = jp1
		}

		if i == i0 && j == j0 {
			break
		}
	}

	return points, usedLabels
}
