package ggeom

import (
	"math"
	"math/big"
)

type Scalar = big.Rat

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

func (a Vec2) ApproxX() float64 {
	v, _ := a.x.Float64()
	return v
}
func (a Vec2) ApproxY() float64 {
	v, _ := a.y.Float64()
	return v
}

func ApproxVec2(x, y float64) Vec2 {
	var xr, yr big.Rat
	xr.SetFloat64(x)
	yr.SetFloat64(y)
	return Vec2{xr, yr}
}

func (a Vec2) Eq(b Vec2) bool {
	return a.x.Cmp(&b.x) == 0 && a.y.Cmp(&b.y) == 0
}

func (a Vec2) Add(b Vec2) Vec2 {
	var newx, newy Scalar
	newx.Add(&a.x, &b.x)
	newy.Add(&a.y, &b.y)
	return Vec2{x: newx, y: newy}
}
func (a Vec2) Sub(b Vec2) Vec2 {
	var newx, newy Scalar
	newx.Sub(&a.x, &b.x)
	newy.Sub(&a.y, &b.y)
	return Vec2{x: newx, y: newy}
}

func (a Vec2) Dot(b Vec2) Scalar {
	var v1, v2 Scalar
	v1.Mul(&a.x, &b.x)
	v2.Mul(&a.y, &b.y)
	v1.Add(&v1, &v2)
	return v1
}

func (a Vec2) Det(b Vec2) Scalar {
	var v1, v2 Scalar
	v1.Mul(&a.x, &b.y)
	v2.Mul(&a.y, &b.x)
	v1.Sub(&v1, &v2)
	return v1
}

func (a Vec2) ApproxLength() float64 {
	x, _ := a.x.Float64()
	y, _ := a.y.Float64()
	return math.Sqrt(x + y*y)
}

func (a Vec2) ApproxScale(v float64) Vec2 {
	x, _ := a.x.Float64()
	y, _ := a.y.Float64()
	l := math.Sqrt(x*x + y*y)
	var nx, ny big.Rat
	nx.SetFloat64((x * v) / l)
	ny.SetFloat64((y * v) / l)
	return Vec2{nx, ny}
}

// Uses y,x ordering
func (p *Polygon2) IndexOfBottommost() int {
	var minx, miny Scalar = p.verts[0].x, p.verts[0].y
	var mini int
	for i, vert := range p.verts[1:] {
		if vert.y.Cmp(&miny) < 0 {
			mini = i + 1
			miny = vert.y
			minx = vert.x
		} else if vert.y.Cmp(&miny) == 0 && vert.x.Cmp(&minx) < 0 {
			mini = i + 1
			minx = vert.x
		}
	}
	return mini
}

// True iff b is reached before c going anticlockwise from a
func IsBetweenAnticlockwise(a Vec2, b Vec2, c Vec2) bool {
	// See AndyG's answer to https://stackoverflow.com/questions/13640931/how-to-determine-if-a-vector-is-between-two-other-vectors
	ab := a.Det(b)
	bc := b.Det(c)
	return (&ab).Sign() >= 0 && (&bc).Sign() >= 0
}

func ACIsReflex(p1, p2, p3 Vec2) bool {
	v := p1.Sub(p2).Det(p3.Sub(p2))
	return (&v).Sign() > 0
}

func NReflexVerts(p *Polygon2) int {
	rp := 0
	for i := 0; i < len(p.verts)-1; i++ {
		p1 := p.verts[i]
		p2 := p.verts[i+1]
		p3 := p.verts[(i+2)%len(p.verts)]
		if ACIsReflex(p1, p2, p3) {
			rp++
		}
	}
	return rp
}

func GetReflexVertIndices(p *Polygon2) []int {
	vs := make([]int, 0, len(p.verts)/2)
	for i := 0; i < len(p.verts)-1; i++ {
		p1 := p.verts[i]
		p2 := p.verts[i+1]
		p3 := p.verts[(i+2)%len(p.verts)]
		if ACIsReflex(p1, p2, p3) {
			vs = append(vs, i+1)
		}
	}
	return vs
}

type label struct {
	a, b, c int
}

// Follows the algorithm described on p. 4 of
//     Ron Wein. Exact and Efficient Construction of Planar Minkowski Sums using the Convolution Method.
func GetConvolutionCycle(p *Polygon2, q *Polygon2) []Vec2 {
	// Get the number of reflex vertices for each polygon.
	rp := GetReflexVertIndices(p)
	rq := GetReflexVertIndices(q)

	nrm := len(rq) * len(p.verts)
	mrn := len(rp) * len(q.verts)

	labs := make(map[label]bool)

	if nrm > mrn {
		return getConvolutionCycle(labs, p, p.IndexOfBottommost(), q, q.IndexOfBottommost(), rq)
	} else {
		return getConvolutionCycle(labs, q, q.IndexOfBottommost(), p, p.IndexOfBottommost(), rp)
	}
}

func getConvolutionCycle(labs map[label]bool, p *Polygon2, pstart int, q *Polygon2, qstart int, rq []int) []Vec2 {
	cs := make([]Vec2, 0, len(p.verts)+len(q.verts))
	appendSingleConvolutionCycle(labs, cs, p, pstart, q, qstart)

	for j := 0; j < len(rq); j++ {
		var q1i int
		if j == 0 {
			q1i = len(rq) - 1
		} else {
			q1i = j - 1
		}
		q1 := q.verts[rq[q1i]]
		q2 := q.verts[rq[j]]
		q3 := q.verts[rq[(j+1)%len(rq)]]
		qseg1 := q2.Sub(q1)
		qseg2 := q3.Sub(q2)

		for i := 0; i < len(p.verts); i++ {
			p1 := p.verts[i]
			p2 := p.verts[(i+1)%len(p.verts)]
			pseg := p2.Sub(p1)

			if IsBetweenAnticlockwise(qseg1, pseg, qseg2) && !labs[label{i, i + 1, j}] {
				pstart = i
				qstart = rq[j]
				cs = appendSingleConvolutionCycle(labs, cs, p, i, q, rq[j])
			}
		}
	}

	return cs
}

func appendSingleConvolutionCycle(labs map[label]bool, points []Vec2, p *Polygon2, i0 int, q *Polygon2, j0 int) []Vec2 {
	i := i0
	j := j0
	s := p.verts[i].Add(q.verts[j])

	if len(points) == 0 || !s.Eq(points[len(points)-1]) {
		points = append(points, s)
	}

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
			break
		}

		var t Vec2
		if incp {
			t = p.verts[ip1].Add(q.verts[j])
			labs[label{i, i + 1, j}] = true
			s = t
			i = ip1
		}
		if incq {
			t = p.verts[i].Add(q.verts[jp1])
			s = t
			j = jp1
		}

		points = append(points, t)

		if i == i0 && j == j0 {
			break
		}
	}

	return points
}
