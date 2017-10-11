package ggeom

import (
	"fmt"
	"math"
	"math/big"

	"github.com/addrummond/ggeom/redblacktree"
	"github.com/emirpasic/gods/trees/binaryheap"
)

var _ = fmt.Printf

type Scalar = big.Rat

type Vec2 struct {
	x Scalar
	y Scalar
}

func r(i float64) Scalar {
	var r Scalar
	r.SetFloat64(i)
	return r
}

func SofVec2(s [][]float64) []Vec2 {
	v2s := make([]Vec2, 0, len(s))
	for _, v := range s {
		v2s = append(v2s, Vec2{r(v[0]), r(v[1])})
	}
	return v2s
}

func SofSofVec2(s [][][]float64) [][]Vec2 {
	v2s := make([][]Vec2, 0, len(s))
	for _, vs := range s {
		rvs := make([]Vec2, 0)
		for _, v := range vs {
			rvs = append(rvs, Vec2{r(v[0]), r(v[1])})
		}
		v2s = append(v2s, rvs)
	}

	return v2s
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
	var xr, yr Scalar
	xr.SetFloat64(x)
	yr.SetFloat64(y)
	return Vec2{xr, yr}
}

func (a *Vec2) Eq(b *Vec2) bool {
	return a.x.Cmp(&b.x) == 0 && a.y.Cmp(&b.y) == 0
}

// Use for testing purposes only.
func (a *Vec2) SlowEqEpsilon(b *Vec2, epsilon float64) bool {
	var e Scalar
	e.SetFloat64(epsilon)
	var v Scalar
	v.Sub(&a.x, &b.x)
	v.Abs(&v)
	if v.Cmp(&e) > 0 {
		return false
	}
	v.Sub(&a.y, &b.y)
	v.Abs(&v)
	return v.Cmp(&e) <= 0
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
	var nx, ny Scalar
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

// -1 for counterclockwise, 0 for colinear, 1 for clockwise
func ClockOrientation(p1 Vec2, p2 Vec2, p3 Vec2) int {
	// http://www.geeksforgeeks.org/orientation-3-ordered-points/
	var v1, v2, v3 Scalar
	v1.Sub(&p2.y, &p1.y)
	v2.Sub(&p3.x, &p2.x)
	v1.Mul(&v1, &v2)
	v2.Sub(&p2.x, &p1.x)
	v3.Sub(&p3.y, &p2.y)
	v2.Mul(&v2, &v3)
	v2.Sub(&v1, &v2)

	return v2.Sign()
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

// Follows the algorithm described in
//     Ron Wein. 2006. Exact and Efficient Construction of Planar Minkowski Sums using the Convolution Method. European Symposium on Algorithms, LNCS 4168, pp. 829-840.
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

	if len(points) == 0 || !(&s).Eq(&points[len(points)-1]) {
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

// Anything that's bigger/smaller than the maximum positive/negative integer that can be exactly
// represented in a float (whose predecesor can also be exactly represented)
// we treat as too big for approximate calculations using float64 to be
// possible. This is too conservative, but it's convenient for testing
// because it's easy to initialize big.Rats that are bigger than this
// from a float64 value. In practice, very few coordinates will have values
// bigger than this.
var maxv *Scalar = big.NewRat(4503599627370496, 1)

func inRange(v *Scalar) bool {
	var abs Scalar
	abs.Abs(v)
	return abs.Cmp(maxv) <= 0
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr
func onSegment(p, q, r *Vec2) bool {
	var maxpxrx, minpxrx, maxpyry, minpyry *Scalar
	if p.x.Cmp(&r.x) >= 0 {
		maxpxrx = &p.x
		minpxrx = &r.x
	} else {
		maxpxrx = &r.x
		minpxrx = &p.x
	}

	if p.y.Cmp(&r.y) >= 0 {
		maxpyry = &p.y
		minpyry = &r.y
	} else {
		maxpyry = &r.y
		minpyry = &p.y
	}

	if q.x.Cmp(maxpxrx) <= 0 && q.x.Cmp(minpxrx) >= 0 &&
		q.y.Cmp(maxpyry) <= 0 && q.y.Cmp(minpyry) >= 0 {
		return true
	}
	return false
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
func orientation(p, q, r *Vec2) int {
	var qySubPy Scalar
	qySubPy.Sub(&q.y, &p.y)
	var rxSubQx Scalar
	rxSubQx.Sub(&r.x, &q.x)
	var fst Scalar
	fst.Mul(&qySubPy, &rxSubQx)

	var qxSubPx Scalar
	qxSubPx.Sub(&q.x, &p.x)
	var rySubQy Scalar
	rySubQy.Sub(&r.y, &q.y)
	var snd Scalar
	snd.Mul(&qxSubPx, &rySubQy)

	// Presumably, comparison is going to be a bit cheaper than subtraction
	// with a big.Rat, so rather than doing a subtraction and looking at
	// the sign, as in the original example code, we compare directly.
	c := fst.Cmp(&snd)
	if c == 0 {
		return 0 // colinear
	} else if c == 1 {
		return 1 // clockwise
	} else {
		return 2 // anticlockwise
	}
}

// If the segments do not intersect or intersect in the non-degenerate case,
// the second member of the return value is nil. Otherwise, it is set to
// an arbitrarily chosen member of the subset of {p1,p2,q1,q2} that lies
// along the intersection.
func segmentsIntersectNoJoinCheck(p1, p2, q1, q2 *Vec2) (bool, *Vec2) {
	if FastSegmentsDontIntersect(p1, p2, q1, q2) {
		return false, nil // the segments definitely don't intersect; won't be a degenerate case
	}

	// See https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
	// and http://www.cdn.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
	// and http://www.dcs.gla.ac.uk/~pat/52233/slides/Geometry1x1.pdf
	// and http://jeffe.cs.illinois.edu/teaching/373/notes/x06-sweepline.pdf

	// Find the four orientations needed for general and special case
	o1 := orientation(p1, q1, q2)
	o2 := orientation(p2, q1, q2)
	o3 := orientation(p1, p2, q1)
	o4 := orientation(p1, p2, q2)

	if o1 != o2 && o3 != o4 {
		return true, nil // the segments intersect; not degenerate
	} else if o1 == 0 && onSegment(q1, p1, q2) {
		return true, p1
	} else if o2 == 0 && onSegment(q1, p2, q2) {
		return true, q2
	} else if o3 == 0 && onSegment(p1, q1, p2) {
		return true, p1
	} else if o4 == 0 && onSegment(p1, q2, p2) {
		return true, q2
	} else {
		return false, nil
	}
}

// Returns true and the intersection point if two segments
// share a point and do not otherwise overlap.
func segmentsAdjacent(p1, p2, q1, q2 *Vec2) (bool, Vec2) {
	var r Vec2

	if p1.Eq(q1) {
		if !onSegment(p1, q2, p2) && !onSegment(q1, p2, q2) {
			r = *p1
			return true, r
		} else {
			return false, r
		}
	} else if p1.Eq(q2) {
		if !onSegment(q2, p2, q1) && !onSegment(p1, q1, p2) {
			r = *p1
			return true, r
		} else {
			return false, r
		}
	} else if p2.Eq(q1) {
		if !onSegment(p2, q2, p1) && !onSegment(q1, p1, q2) {
			r = *p2
			return true, r
		} else {
			return false, r
		}
	} else if p2.Eq(q2) {
		if !onSegment(p2, q1, p1) && !onSegment(q2, p1, q1) {
			r = *p2
			return true, r
		} else {
			return false, r
		}
	} else {
		return false, r
	}
}

// Returns (intersects or not, point on overlap iff degenerate intersection). Degenerate
// intersections are those where no single intersection point is defined.
func SegmentsIntersect(p1, p2, q1, q2 *Vec2) (bool, *Vec2) {
	// Instances where one segment joins the end of another are likely to be quite
	// common when dealing with polygons, so we should be able to save a fair
	// bit of big.Rat arithmetic by checking for this case.
	adj, _ := segmentsAdjacent(p1, p2, q1, q2)
	if adj {
		return true, nil // the segments intersect; not degenerate
	}

	return segmentsIntersectNoJoinCheck(p1, p2, q1, q2)
}

// Returns a boolean indicating whether the segments intersect,
// a boolean indicating whether there is a unique intersection point,
// and the intersection point itself (set to (0,0) if
// the segments don't intersect, or a point arbitrarily chosen
// from the subset of {p1,p2,q1,q2} that lies along the intersection.
func SegmentIntersection(p1, p2, q1, q2 *Vec2) (bool, bool, Vec2) {
	adj, pt := segmentsAdjacent(p1, p2, q1, q2)
	if adj {
		return true, true, pt
	}

	var v Vec2
	intersect, degeneratePt := segmentsIntersectNoJoinCheck(p1, p2, q1, q2)
	if intersect {
		if degeneratePt != nil {
			return true, false, *degeneratePt
		} else {
			return true, true, NondegenerateSegmentIntersection(p1, p2, q1, q2)
		}
	} else {
		return false, false, v
	}
}

var pinf = math.Inf(1)
var ninf = math.Inf(-1)

// FastSegmentsDontIntersect tests whether it is possible to quickly determine that the segments
// do not intersect using inexact arithmetic. We do a simple
// check that rejects segment pairs that have no x or y
// overlap. The good thing about this test is that it's easy
// to get the reasoning about floating point precision
// correct, since the test can be done using comparisons
// withouot any arithmetic.
func FastSegmentsDontIntersect(s1a, s1b, s2a, s2b *Vec2) bool {
	if inRange(&s1a.x) && inRange(&s1a.y) && inRange(&s1b.x) && inRange(&s1b.y) && inRange(&s2a.x) && inRange(&s2a.y) && inRange(&s2b.x) && inRange(&s2b.y) {
		f1ax := s1a.ApproxX()
		f1ay := s1a.ApproxY()
		f1bx := s1b.ApproxX()
		f1by := s1b.ApproxY()

		f2ax := s2a.ApproxX()
		f2ay := s2a.ApproxY()
		f2bx := s2b.ApproxX()
		f2by := s2b.ApproxY()

		// Assuming that math.Rat is doing it's job correctly,
		// we can be sure that the true value of each coordinate lies between
		// the next lowest and next highest float64.
		f1axl := math.Nextafter(f1ax, ninf)
		f1ayl := math.Nextafter(f1ay, ninf)
		f1bxl := math.Nextafter(f1bx, ninf)
		f1byl := math.Nextafter(f1by, ninf)

		f1axh := math.Nextafter(f1ax, pinf)
		f1ayh := math.Nextafter(f1ay, pinf)
		f1bxh := math.Nextafter(f1bx, pinf)
		f1byh := math.Nextafter(f1by, pinf)

		f2axl := math.Nextafter(f2ax, ninf)
		f2ayl := math.Nextafter(f2ay, ninf)
		f2bxl := math.Nextafter(f2bx, ninf)
		f2byl := math.Nextafter(f2by, ninf)

		f2axh := math.Nextafter(f2ax, pinf)
		f2ayh := math.Nextafter(f2ay, pinf)
		f2bxh := math.Nextafter(f2bx, pinf)
		f2byh := math.Nextafter(f2by, pinf)

		s2rightofs1 := f2axl > f1axh && f2axl > f1bxh && f2bxl > f1axh && f2bxl > f1bxh
		s1rightofs2 := f2axh < f1axl && f2axh < f1bxl && f2bxh < f1axl && f2bxh < f1bxl

		if s1rightofs2 || s2rightofs1 {
			return true
		}

		s2aboves1 := f2ayl > f1ayh && f2ayl > f1byh && f2byl > f1ayh && f2byl > f1byh
		s1aboves2 := f2ayh < f1ayl && f2ayh < f1byl && f1byh < f1ayl && f2byh < f1byl

		if s1aboves2 || s2aboves1 {
			return true
		}
	}

	return false
}

// Assumes that segments intersect at a single point and are not parallel.
func NondegenerateSegmentIntersection(s1a, s1b, s2a, s2b *Vec2) Vec2 {
	var tmp, w Scalar
	var x, y Scalar
	var xset, yset bool

	var m Scalar
	w.Sub(&s1b.x, &s1a.x)
	if w.Sign() == 0 {
		// The line is vertical. Thus we know that the x value
		// of the intersection point will be the x value of the
		// points on the line.
		x = s1a.x
		xset = true
	} else {
		tmp.Sub(&s1b.y, &s1a.y)
		if tmp.Sign() == 0 {
			// The line is horizontal.
			y = s1b.y
			yset = true
		} else {
			m.Mul(&tmp, w.Inv(&w))
		}
	}

	var n Scalar
	w.Sub(&s2b.x, &s2a.x)
	if w.Sign() == 0 {
		// The line is vertical.
		x = s2a.x
		xset = true
	} else {
		tmp.Sub(&s2b.y, &s2a.y)
		if tmp.Sign() == 0 {
			// The line is horizontal.
			y = s2b.y
			yset = true
		} else {
			n.Mul(&tmp, w.Inv(&w))
		}
	}

	var c Scalar
	tmp.Mul(&m, &s1a.x)
	c.Sub(&s1a.y, &tmp)

	var d Scalar
	tmp.Mul(&n, &s2a.x)
	d.Sub(&s2a.y, &tmp)

	// We know that m - n is nonzero because the lines aren't parallel.
	if !xset {
		x.Sub(&d, &c)
		if x.Sign() != 0 { // save some unnecessary arithmetic
			tmp.Sub(&m, &n)
			tmp.Inv(&tmp)
			x.Mul(&x, &tmp)
		}
	}

	if !yset {
		tmp.Mul(&m, &d)
		y.Mul(&n, &c)
		y.Sub(&y, &tmp)
		if y.Sign() != 0 { // save some unnecessary arithmetic
			tmp.Sub(&n, &m)
			tmp.Inv(&tmp)
			y.Mul(&y, &tmp)
		}
	}

	return Vec2{x, y}
}

// Returns min(s1.y, s2.y) if line is vertical. Assumes
// that the line has a value for the given x coordinate.
func SegmentYValueAtX(sa, sb *Vec2, x *Scalar) Scalar {
	var tmp, w Scalar

	var m Scalar
	w.Sub(&sb.x, &sa.x)
	if w.Sign() == 0 {
		// The line is vertical. Return the smallest y value.
		if sa.y.Cmp(&sb.y) <= 0 {
			return sa.y
		} else {
			return sb.y
		}
	} else {
		tmp.Sub(&sb.y, &sa.y)
		if tmp.Sign() == 0 {
			// The line is horizontal.
			return sa.y
		} else {
			m.Mul(&tmp, w.Inv(&w))
		}
	}

	var c Scalar
	tmp.Mul(&m, &sa.x)
	c.Sub(&sa.y, &tmp)

	var y Scalar
	y.Mul(&m, x)
	y.Add(&y, &c)

	return y
}

const (
	start = 0
	cross = 1
	end   = 2
)

type bentleyEvent struct {
	kind    int
	i       int
	i2      int
	left    *Vec2
	right   *Vec2
	deleted bool
}

func bentleyEventPs(i int, points []Vec2) (*Vec2, *Vec2) {
	p1 := &(points[i])
	p2 := &(points[(i+1)%len(points)])

	if p1.x.Cmp(&p2.x) <= 0 {
		return p1, p2
	} else {
		return p2, p1
	}
}

func sameOrAdjacent(s1, s2, l int) bool {
	d := s1 - s2
	dd := d * d
	return dd <= 1 || d == l-1 || d == -(l-1)
}

type Intersection struct {
	seg1 int
	seg2 int
	p    Vec2
}

// An implementation of the Bentley Ottmann algorithm for the case where
// the input segments are connected in a loop. The loop is implicitly closed
// by segment from last point to first point. This returns all intersections
// except for the points in the original input (which could all be considered
// intersection points).
func SegmentLoopIntersections(points []Vec2) []Intersection {
	// Some useful pseudocode at https://www.hackerearth.com/practice/math/geometry/line-intersection-using-bentley-ottmann-algorithm/tutorial/
	// http://jeffe.cs.illinois.edu/teaching/373/notes/x06-sweepline.pdf
	// https://github.com/ideasman42/isect_segments-bentley_ottmann/blob/master/poly_point_isect.py

	hcmp := func(a, b interface{}) int {
		aa, bb := a.(*bentleyEvent), b.(*bentleyEvent)

		x1, x2 := &aa.left.x, &bb.left.x
		if aa.kind != start {
			x1 = &aa.right.x
		}
		if bb.kind != start {
			x2 = &bb.right.x
		}

		c := x1.Cmp(x2)
		if c != 0 {
			return c
		} else {
			// 'start' < 'cross' < 'end'
			return aa.kind - bb.kind
		}
	}
	events := binaryheap.NewWith(hcmp)
	for i := 0; i < len(points); i++ {
		left, right := bentleyEventPs(i, points)
		e1 := bentleyEvent{
			kind:  start,
			i:     i,
			left:  left,
			right: right,
		}
		events.Push(&e1)
		e2 := bentleyEvent{
			kind:  end,
			i:     i,
			left:  left,
			right: right,
		}
		events.Push(&e2)
	}

	type tkey struct {
		segi int
		x    *Scalar
		y    *Scalar
	}
	tcmp := func(a, b interface{}) int {
		aa, bb := a.(tkey), b.(tkey)

		c := aa.y.Cmp(bb.y)
		if c == 0 {
			c = aa.x.Cmp(bb.x)
			if c != 0 {
				return c
			} else {
				return aa.segi - bb.segi
			}
		} else {
			return c
		}
	}
	tree := redblacktree.NewWith(tcmp)
	segToKey := make(map[int]tkey)

	intersections := make([]Intersection, 0)
	intersectionPoints := redblacktree.NewWith(func(a, b interface{}) int {
		aa, bb := a.(*Vec2), b.(*Vec2)
		c := aa.y.Cmp(&bb.y)
		if c != 0 {
			return c
		} else {
			return aa.x.Cmp(&bb.x)
		}
	})

	count := 0
	for e, notEmpty := events.Pop(); notEmpty; e, notEmpty = events.Pop() {
		if count > 100 {
			break
		}
		count++

		event := e.(*bentleyEvent)
		if event.deleted {
			continue
		}

		fmt.Printf("Event: kind=%v [x=%v], seg1=%v, seg2=%v\n", event.kind, &event.left.x, event.i, event.i2)

		vals := tree.Values()
		keys := tree.Keys()
		fmt.Printf("The tree before\n")
		for i := 0; i < len(vals); i++ {
			k := keys[i].(tkey)
			v := vals[i].(int)
			fmt.Printf("    k=(%v,%v) -> %v\n", k.segi, k.y, v)
		}
		fmt.Printf("\n")

		if event.kind == start {
			p1 := &points[event.i]
			p2 := &points[(event.i+1)%len(points)]
			y := SegmentYValueAtX(p1, p2, &p1.x)
			//fmt.Printf("y value of segment (%v,%v) -> (%v,%v) at x=%v is %v\n", &p1.x, &p1.y, &p2.x, &p2.y, &p1.x, &y)

			tk := tkey{event.i, &p1.x, &y}
			it1, replaced := tree.PutAndGetIterator(tk, event.i)
			if replaced {
				panic("Unexpected replacement")
			}
			segToKey[event.i] = tk
			it2 := it1

			var psp1, psp2, nsp1, nsp2 *Vec2
			var prevI, nextI int
			for it1.Prev() {
				prevI = it1.Value().(int)

				if !sameOrAdjacent(event.i, prevI, len(points)) {
					psp1 = &points[prevI]
					psp2 = &points[(prevI+1)%len(points)]
					p1 := &points[event.i]
					p2 := &points[(event.i+1)%len(points)]
					intersect, _, intersectionPoint := SegmentIntersection(psp1, psp2, p1, p2)
					if intersect {
						events.Push(&bentleyEvent{
							kind:  cross,
							i:     prevI,
							i2:    event.i,
							left:  &intersectionPoint,
							right: &intersectionPoint,
						})
					}

					break
				}
			}
			for it2.Next() {
				nextI = it2.Value().(int)

				if !sameOrAdjacent(nextI, event.i, len(points)) {
					nsp1 = &points[nextI]
					nsp2 = &points[(nextI+1)%len(points)]
					p1 := &points[event.i]
					p2 := &points[(event.i+1)%len(points)]
					intersect, _, intersectionPoint := SegmentIntersection(nsp1, nsp2, p1, p2)
					if intersect {
						events.Push(&bentleyEvent{
							kind:  cross,
							i:     nextI,
							i2:    event.i,
							left:  &intersectionPoint,
							right: &intersectionPoint,
						})
					}

					break
				}
			}
		} else if event.kind == end {
			it, f := tree.GetIterator(segToKey[event.i])
			if !f {
				panic(fmt.Sprintf("Internal error [1] in 'SegmentLoopIntersections': could not find key with seg index %v\n", event.i))
			}

			// Because we're dealing with a strip of lines segements, we don't need to do
			// anything at an end point except remove the relevant entry from the tree,
			// as there'll be another corresponding start point where intersection checks
			// can be made.

			tree.RemoveAt(it)
		} else if event.kind == cross {
			itn := Intersection{event.i, event.i2, *event.left}
			intersections = append(intersections, itn)

			_, replaced := intersectionPoints.PutAndGetIterator(event.left, true)
			if !replaced {
				si := event.i
				ti := event.i2

				if si == ti {
					panic("Internal error [2] in 'SegementLoopIteration'")
				}

				sKey, tKey := segToKey[si], segToKey[ti]
				if tcmp(sKey, tKey) > 0 {
					si, ti = ti, si
					sKey, tKey = tKey, sKey
				}

				sIt, sItExists := tree.GetIterator(segToKey[si])
				tIt, tItExists := tree.GetIterator(segToKey[ti])

				if !(sItExists && tItExists) {
					panic("Internal error [3] in 'SegmentLoopIntersections'")
				}

				fmt.Printf("Intersection of %v and %v\n", si, ti)

				if tree.Size() > 2 {
					if tcmp(tKey, sKey) == 0 {
						panic("Bad comparison before swap")
					}

					tree.SwapAt(sIt, tIt)
					sIt, tIt = tIt, sIt
					segToKey[si], segToKey[ti] = tKey, sKey

					s1 := &points[si]
					s2 := &points[(si+1)%len(points)]
					t1 := &points[ti]
					t2 := &points[(ti+1)%len(points)]

					vals := tree.Values()
					keys := tree.Keys()
					fmt.Printf("Modified tree\n")
					for i := 0; i < len(vals); i++ {
						k := keys[i].(tkey)
						v := vals[i].(int)
						fmt.Printf("    k=(%v,%v) -> %v\n", k.segi, k.y, v)
					}
					fmt.Printf("\n")

					var u, r int
					var u1, u2, r1, r2 *Vec2

					for sIt.Next() {
						u = sIt.Value().(int)
						if !sameOrAdjacent(u, si, len(points)) {
							u1 = &points[u]
							u2 = &points[(u+1)%len(points)]

							intersect, _, intersectionPoint := SegmentIntersection(s1, s2, u1, u2)
							if intersect {
								fmt.Printf("Inserting A %v x %v\n", si, u)
								events.Push(&bentleyEvent{
									kind:  cross,
									i:     si,
									i2:    u,
									left:  &intersectionPoint,
									right: &intersectionPoint,
								})
							}

							break
						}
					}

					for tIt.Prev() {
						r = tIt.Value().(int)
						if !sameOrAdjacent(r, ti, len(points)) {
							r1 = &points[r]
							r2 = &points[(r+1)%len(points)]

							if u1 != nil && (ti == u && r == si) || (ti == si && r == u) {
								fmt.Printf("SKEY %+v, TKEY %+v  [%v, %v]\n", &sKey, &tKey, ti, r)
								panic("SHOULD NOT HAPPEN")
							}

							intersect, _, intersectionPoint := SegmentIntersection(t1, t2, r1, r2)
							if intersect {
								fmt.Printf("Inserting B %v x %v\n", ti, r)
								events.Push(&bentleyEvent{
									kind:  cross,
									i:     ti,
									i2:    r,
									left:  &intersectionPoint,
									right: &intersectionPoint,
								})
							}

							break
						}
					}
				}
			}
		}
	}

	return intersections
}
