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

func (a *Vec2) ApproxX() float64 {
	v, _ := a.x.Float64()
	return v
}
func (a *Vec2) ApproxY() float64 {
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

// SlowEqEpsilon tests whether two vectors are equal within
// the margin specified by epsilon (inclusive, checked indepently
// for x and y coordinates). This function is not very efficient
// and should be used for testing only.
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

func (a *Vec2) Copy() *Vec2 {
	var x, y Scalar
	x.Set(&a.x)
	y.Set(&a.y)
	return &Vec2{x, y}
}

func (a *Vec2) Add(b *Vec2, c *Vec2) {
	a.x.Add(&b.x, &c.x)
	a.y.Add(&b.y, &c.y)
}

func (a *Vec2) Sub(b *Vec2, c *Vec2) {
	a.x.Sub(&b.x, &c.x)
	a.y.Sub(&b.y, &c.y)
}

func (a *Vec2) Dot(b *Vec2) *Scalar {
	var v1, v2 Scalar
	v1.Mul(&a.x, &b.x)
	v2.Mul(&a.y, &b.y)
	v1.Add(&v1, &v2)
	return &v1
}

// Det computes the z value of the 3d cross product with z=0 for the input vectors.
func (a *Vec2) Det(b *Vec2) *Scalar {
	var v1, v2 Scalar
	v1.Mul(&a.x, &b.y)
	v2.Mul(&a.y, &b.x)
	v1.Sub(&v1, &v2)
	return &v1
}

// ApproxLength performs an approximate calculation of the length of the vector
// using float64 arithmetic.
func (a *Vec2) ApproxLength() float64 {
	x, _ := a.x.Float64()
	y, _ := a.y.Float64()
	return math.Sqrt(x + y*y)
}

// ApproxScale approximately scales the vector using float64 arithmetic.
func (a Vec2) ApproxScale(v float64) Vec2 {
	x, _ := a.x.Float64()
	y, _ := a.y.Float64()
	l := math.Sqrt(x*x + y*y)
	var nx, ny Scalar
	nx.SetFloat64((x * v) / l)
	ny.SetFloat64((y * v) / l)
	return Vec2{nx, ny}
}

// IsBetweenAnticlockwise returns true iff b is reached before c going anticlockwise from a.
func IsBetweenAnticlockwise(a *Vec2, b *Vec2, c *Vec2) bool {
	ba := b.Det(a)
	ac := a.Det(c)
	cb := c.Det(b)
	sba, sac, scb := ba.Sign(), ac.Sign(), cb.Sign()

	if sba >= 0 {
		return sac < 0 && scb <= 0
	} else {
		return sac < 0 || scb <= 0
	}
}

// ACIsReflex returns true if the vertex p2 in a polygon,
// preceded by p1 and followed by p3 (anticlockwise order)
// is a reflex vertex.
func ACIsReflex(p1, p2, p3 *Vec2) bool {
	var s1, s2 Vec2
	s1.Sub(p1, p2)
	s2.Sub(p3, p2)
	v := s1.Det(&s2)
	return v.Sign() > 0
}

// NReflexVerts returns the number of reflex vertices that the given polygon has.
func NReflexVerts(p *Polygon2) int {
	rp := 0
	for i := 0; i < len(p.verts)-1; i++ {
		p1 := &p.verts[i]
		p2 := &p.verts[i+1]
		p3 := &p.verts[(i+2)%len(p.verts)]
		if ACIsReflex(p1, p2, p3) {
			rp++
		}
	}
	return rp
}

// GetReflexVertIndices returns the indices of a polygon's reflex vertices.
func GetReflexVertIndices(p *Polygon2) []int {
	vs := make([]int, 0, len(p.verts)/2)
	for i := 0; i < len(p.verts)-1; i++ {
		p1 := &p.verts[i]
		p2 := &p.verts[i+1]
		p3 := &p.verts[(i+2)%len(p.verts)]
		if ACIsReflex(p1, p2, p3) {
			vs = append(vs, i+1)
		}
	}
	return vs
}

type label struct {
	a, b, c, d int
}

func (p *Polygon2) indexOfBottommost() int {
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

// GetConvolutionCycle returns the sequence of vertices that forms
// the convolution cycle for the polyons p and q.
func GetConvolutionCycle(p *Polygon2, q *Polygon2) []Vec2 {
	// Roughly follows the algorithm described in
	//     Ron Wein. 2006. Exact and Efficient Construction of Planar Minkowski Sums using the Convolution Method. European Symposium on Algorithms, LNCS 4168, pp. 829-840.

	// Get the number of reflex vertices for each polygon.
	rp := GetReflexVertIndices(p)
	rq := GetReflexVertIndices(q)

	nrm := len(rq) * len(p.verts)
	mrn := len(rp) * len(q.verts)

	labs := make(map[label]bool)

	if nrm < mrn {
		return getConvolutionCycle(labs, p, p.indexOfBottommost(), q, q.indexOfBottommost(), rq)
	} else {
		return getConvolutionCycle(labs, q, q.indexOfBottommost(), p, p.indexOfBottommost(), rp)
	}
}

func getConvolutionCycle(labs map[label]bool, p *Polygon2, pstart int, q *Polygon2, qstart int, rq []int) []Vec2 {
	cs := make([]Vec2, 0, len(p.verts)+len(q.verts))
	cs = appendSingleConvolutionCycle(labs, cs, p, pstart, q, qstart)

	for rqi := 0; rqi < len(rq); rqi++ {
		j := rq[rqi]
		jm1 := (len(q.verts) + j - 1) % len(q.verts)
		jp1 := (j + 1) % len(q.verts)

		q1 := &q.verts[jm1]
		q2 := &q.verts[j]
		q3 := &q.verts[jp1]

		var qseg1, qseg2 Vec2
		qseg1.Sub(q2, q1)
		qseg2.Sub(q3, q2)

		for i := 0; i < len(p.verts); i++ {
			ip1 := (i + 1) % len(p.verts)
			p1 := &p.verts[i]
			p2 := &p.verts[ip1]
			var pseg Vec2
			pseg.Sub(p2, p1)

			if IsBetweenAnticlockwise(&qseg1, &pseg, &qseg2) && !labs[label{i, ip1, j, -1}] {
				//fmt.Printf("Starting next convolution cycle at %v vert of p at (%v,%v)\n", pstart, p.verts[i].ApproxX(), p.verts[i].ApproxY())
				cs = appendSingleConvolutionCycle(labs, cs, p, i, q, j)
			}
		}
	}

	if len(cs) == 0 {
		panic("Internal error in 'getConvolutionCycle'")
	}
	cs = cs[:len(cs)-1]

	return cs
}

func appendSingleConvolutionCycle(labs map[label]bool, points []Vec2, p *Polygon2, i0 int, q *Polygon2, j0 int) []Vec2 {
	i := i0
	j := j0
	var s Vec2
	s.Add(&p.verts[i], &q.verts[j])

	if len(points) == 0 || !(&s).Eq(&points[len(points)-1]) {
		points = append(points, s)
	}

	for {
		ip1 := (i + 1) % len(p.verts)
		jp1 := (j + 1) % len(q.verts)
		jm1 := (len(q.verts) + j - 1) % len(q.verts)

		var piTOpiplus1, qjminus1TOqj, qjTOqjplus1 Vec2
		piTOpiplus1.Sub(&p.verts[ip1], &p.verts[i])
		qjminus1TOqj.Sub(&q.verts[j], &q.verts[jm1])
		qjTOqjplus1.Sub(&q.verts[jp1], &q.verts[j])
		//fmt.Printf("On P=%v, Q=%v\n", i, j)
		//fmt.Printf("Is between [Q%v,P%v,Q%v] cc=%v (%v,%v),  (%v,%v)  (%v,%v)\n", jm1, i, j, incp, qjminus1TOqj.ApproxX(), qjminus1TOqj.ApproxY(), piTOpiplus1.ApproxX(), piTOpiplus1.ApproxY(), qjTOqjplus1.ApproxX(), qjTOqjplus1.ApproxY())

		// Modified from the Wein paper according to the similar but
		// slightly different pseudocode listing on p.29 of
		// https://pdfs.semanticscholar.org/a2d2/186d8b6481be81eed75857d831b32a2ad940.pdf
		//
		// This appears to be a direct modification of Wein's original, since 'incq'
		// is still defined even though its value is never used.
		//
		// Unlike Wein's original pseudocode, the modified psuedocode seems to work!
		//
		// I'm still using Wein's algorithm for computing the additional convolution
		// cycles required when both of the polygon's are non-convex.

		incp := IsBetweenAnticlockwise(&qjminus1TOqj, &piTOpiplus1, &qjTOqjplus1) && !labs[label{i, ip1, j, -1}]

		var s Vec2
		if incp {
			//fmt.Printf("===> cc=%v (%v,%v),  (%v,%v)  (%v,%v)\n", incp, qjminus1TOqj.ApproxX(), qjminus1TOqj.ApproxY(), piTOpiplus1.ApproxX(), piTOpiplus1.ApproxY(), qjTOqjplus1.ApproxX(), qjTOqjplus1.ApproxY())
			s.Add(&p.verts[ip1], &q.verts[j])
			labs[label{i, ip1, j, -1}] = true
			i = ip1
			points = append(points, s)
		} else {
			//fmt.Printf("===> Q cc=%v (%v,%v),  (%v,%v)  (%v,%v)\n", incq, piminus1TOpi.ApproxX(), piminus1TOpi.ApproxY(), qjTOqjplus1.ApproxX(), qjTOqjplus1.ApproxY(), piTOpiplus1.ApproxX(), piTOpiplus1.ApproxY())
			s.Add(&p.verts[i], &q.verts[jp1])
			labs[label{i, -1, j, jp1}] = true
			j = jp1
			points = append(points, s)
		}

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

// OnSegment takes three colinear points p, q, r and returns true
// iff the point q lies on line segment 'pr
func OnSegment(p, q, r *Vec2) bool {
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

// Orientation finds the  orientation of an ordered triplet (p, q, r).
// It returns
// 0 if p, q and r are colinear
// 1 if they are clockwise
// 2 if they are counterclockwise
func Orientation(p, q, r *Vec2) int {
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
	o1 := Orientation(p1, q1, q2)
	o2 := Orientation(p2, q1, q2)
	o3 := Orientation(p1, p2, q1)
	o4 := Orientation(p1, p2, q2)

	if o1 != o2 && o3 != o4 {
		return true, nil // the segments intersect; not degenerate
	} else if o1 == 0 && OnSegment(q1, p1, q2) {
		return true, p1
	} else if o2 == 0 && OnSegment(q1, p2, q2) {
		return true, p2
	} else if o3 == 0 && OnSegment(p2, q1, p2) {
		return true, q1
	} else if o4 == 0 && OnSegment(p1, q2, p2) {
		return true, q2
	} else {
		return false, nil
	}
}

func segmentsSharePoint(p1, p2, q1, q2 *Vec2) *Vec2 {
	if p1.Eq(q1) || p1.Eq(q2) {
		return p1
	} else if p2.Eq(q1) || p2.Eq(q2) {
		return p2
	} else {
		return nil
	}
}

// Returns true and the intersection point if two segments
// share a point and do not otherwise overlap.
func segmentsAdjacent(p1, p2, q1, q2 *Vec2) (bool, *Vec2) {
	var r Vec2

	if p1.Eq(q1) {
		if !OnSegment(p1, q2, p2) && !OnSegment(q1, p2, q2) {
			return true, p1
		} else {
			return false, &r
		}
	} else if p1.Eq(q2) {
		if !OnSegment(q2, p2, q1) && !OnSegment(p1, q1, p2) {
			return true, p1
		} else {
			return false, &r
		}
	} else if p2.Eq(q1) {
		if !OnSegment(p2, q2, p1) && !OnSegment(q1, p1, q2) {
			return true, p2
		} else {
			return false, &r
		}
	} else if p2.Eq(q2) {
		if !OnSegment(p2, q1, p1) && !OnSegment(q2, p1, q1) {
			return true, p2
		} else {
			return false, &r
		}
	} else {
		return false, &r
	}
}

type SegmentIntersectionInfo struct {
	// true iff the segments intersect, incuding all weird/degenerate intersection cases
	intersect bool
	// true iff the segments intersect and there is a unique intersection point
	unique bool
	// true iff the segments intersect and are adjacent (no overlaping section, intersect at a vertex shared by the two)
	adjacent bool
	// true iff the segments intersect and the intersection point is one of the segment vertices.
	atVertex bool
	// the intersection point if any; set to an arbitrarily chosen point on the intersection if there is no unique intersection point; set to (0,0) if there is no intersection point.
	p *Vec2
}

// SegmentIntersection returns a SegmentIntersectionInfo
// structure, which contains all the information you would typically
// want about the intersection of two segments (if any).
func GetSegmentIntersectionInfo(p1, p2, q1, q2 *Vec2) SegmentIntersectionInfo {
	adj, pt := segmentsAdjacent(p1, p2, q1, q2)
	if adj {
		return SegmentIntersectionInfo{
			intersect: true,
			unique:    true,
			adjacent:  true,
			atVertex:  true,
			p:         pt,
		}
	}

	var v Vec2
	intersect, degeneratePt := segmentsIntersectNoJoinCheck(p1, p2, q1, q2)
	if intersect {
		if degeneratePt != nil {
			return SegmentIntersectionInfo{
				intersect: true,
				unique:    false,
				adjacent:  false,
				atVertex:  degeneratePt.Eq(p1) || degeneratePt.Eq(p2) || degeneratePt.Eq(q1) || degeneratePt.Eq(q2),
				p:         degeneratePt,
			}
		} else {
			pt := NondegenerateSegmentIntersection(p1, p2, q1, q2)
			return SegmentIntersectionInfo{
				intersect: true,
				unique:    true,
				adjacent:  false,
				atVertex:  pt.Eq(p1) || pt.Eq(p2) || pt.Eq(q1) || pt.Eq(q2),
				p:         pt,
			}
		}
	} else {
		return SegmentIntersectionInfo{
			intersect: false,
			unique:    false,
			adjacent:  false,
			p:         &v,
		}
	}
}

// NonFunkySegment intersection returns a boolean indicating whether or
// not the two segments have a non-funky intersection, and a point
// which is the intersection point if they do, or (0,0) otherwise.
//
// An intersection is non-funky iff:
//
//     (i)   There is a unique intersection point,
//     (ii)  the segments are not parallel, and
//     (iii) the intersection point is not one of the segments' vertices
func NonFunkySegmentIntersection(p1, p2, q1, q2 *Vec2) (bool, *Vec2) {
	var v Vec2

	shared := segmentsSharePoint(p1, p2, q1, q2)
	if shared != nil {
		return false, &v
	}

	intersect, degeneratePt := segmentsIntersectNoJoinCheck(p1, p2, q1, q2)
	if intersect && degeneratePt == nil {
		pt := NondegenerateSegmentIntersection(p1, p2, q1, q2)
		if pt.Eq(p1) || pt.Eq(p2) || pt.Eq(q1) || pt.Eq(q2) {
			return false, &v
		}
		return true, pt
	}

	return false, &v
}

func SegmentIntersection(p1, p2, q1, q2 *Vec2) (bool, *Vec2) {
	adj, pt := segmentsAdjacent(p1, p2, q1, q2)
	if adj {
		return true, pt
	}

	intersect, degeneratePt := segmentsIntersectNoJoinCheck(p1, p2, q1, q2)
	if intersect {
		if degeneratePt != nil {
			return false, degeneratePt
		}

		return true, NondegenerateSegmentIntersection(p1, p2, q1, q2)
	}

	var v Vec2
	return false, &v
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

// NondegenerateSegmentIntersection returns the intersection point of
// two segements on the assumption that the segments intersect at a
// single point and are not parallel.
func NondegenerateSegmentIntersection(s1a, s1b, s2a, s2b *Vec2) *Vec2 {
	var x, y, m, n, tmp Scalar

	tmp.Sub(&s1b.x, &s1a.x)
	if tmp.Sign() == 0 {
		// The first line is vertical. Thus we know that the x value
		// of the intersection point will be the x value of the
		// points on the line.
		x.Set(&s1a.x)

		// Second line cannot be vertical as we are handling non-degenerate cases only.
		y.Sub(&s2b.y, &s2a.y)
		n.Sub(&s2b.x, &s2a.x)
		n.Inv(&n)
		n.Mul(&n, &y) // n is now the slope of the second line

		var d Scalar
		d.Mul(&n, &s2a.x)
		d.Sub(&s2a.y, &d)
		y.Mul(&n, &x)
		y.Add(&y, &d)

		return &Vec2{x, y}
	}

	var tmp2 Scalar
	tmp2.Sub(&s1b.y, &s1a.y)
	if tmp2.Sign() == 0 {
		// The line is horizontal.
		y.Set(&s1b.y)
		// m is initialized as 0
	} else {
		m.Mul(&tmp2, tmp.Inv(&tmp))
	}

	tmp.Sub(&s2b.x, &s2a.x)
	if tmp.Sign() == 0 {
		// The line is vertical.
		x.Set(&s2a.x)

		// First line cannot be vertical as we are handling non-degenerate cases only.
		var c Scalar
		c.Mul(&m, &s1a.x)
		c.Sub(&s1a.y, &c)
		y.Mul(&m, &x)
		y.Add(&y, &c)

		return &Vec2{x, y}
	}

	tmp2.Sub(&s2b.y, &s2a.y)
	if tmp2.Sign() == 0 {
		// The line is horizontal.
		y.Set(&s2b.y)
		// n is initialized as 0
	} else {
		n.Mul(&tmp2, tmp.Inv(&tmp))
	}

	// If we get here, neither of the lines is vertical, and the general
	// solution will work.

	var c Scalar
	tmp.Mul(&m, &s1a.x)
	c.Sub(&s1a.y, &tmp)

	var d Scalar
	tmp.Mul(&n, &s2a.x)
	d.Sub(&s2a.y, &tmp)

	// We know that m - n is nonzero because the lines aren't parallel.
	x.Sub(&d, &c)
	if x.Sign() != 0 { // save some unnecessary arithmetic
		tmp.Sub(&m, &n)
		tmp.Inv(&tmp)
		x.Mul(&x, &tmp)
	}

	tmp.Mul(&m, &d)
	y.Mul(&n, &c)
	y.Sub(&y, &tmp)
	if y.Sign() != 0 { // save some unnecessary arithmetic
		tmp.Sub(&n, &m)
		tmp.Inv(&tmp)
		y.Mul(&y, &tmp)
	}

	return &Vec2{x, y}
}

func f(v *Scalar) float64 {
	r, _ := v.Float64()
	return r
}

// SegmentYValueAtX returns min(sa.y, sb.y) if line is vertical.
// It assumes that the line has a value for the given x coordinate.
func SegmentYValueAtX(sa, sb *Vec2, x *Scalar) *Scalar {
	var tmp, w, m Scalar

	w.Sub(&sb.x, &sa.x)
	if w.Sign() == 0 {
		// The line is vertical. Return the smallest y value.
		if sa.y.Cmp(&sb.y) <= 0 {
			return &sa.y
		}
		return &sb.y
	}

	tmp.Sub(&sb.y, &sa.y)
	if tmp.Sign() == 0 {
		// The line is horizontal.
		return &sa.y
	}

	m.Mul(&tmp, w.Inv(&w))

	var c Scalar
	tmp.Mul(&m, &sa.x)
	c.Sub(&sa.y, &tmp)

	var y Scalar
	y.Mul(&m, x)
	y.Add(&y, &c)

	return &y
}

const (
	start    = 0
	vertical = 1
	cross    = 2
	end      = 3
)

type bentleyEvent struct {
	kind     int
	i        int
	i2       int
	left     *Vec2
	right    *Vec2
	deleted  bool
	swapOnly bool
}

func bentleyEventCmp(a, b interface{}) int {
	aa, bb := a.(*bentleyEvent), b.(*bentleyEvent)

	x1, x2, y1, y2 := &aa.left.x, &bb.left.x, &aa.left.y, &bb.left.y
	if aa.kind != start {
		x1 = &aa.right.x
		y1 = &aa.right.y
	}
	if bb.kind != start {
		x2 = &bb.right.x
		y2 = &bb.right.y
	}

	c := x1.Cmp(x2)
	if c != 0 {
		return c
	} else if aa.kind != bb.kind {
		return aa.kind - bb.kind
	} else {
		return y1.Cmp(y2)
	}

	/*c := x1.Cmp(x2)
	if c != 0 {
		return c
	} else if aa.kind == end && bb.kind != end {
		return 1
	} else if aa.kind != end && bb.kind == end {
		return -1
	} else {
		y1, y2 := &aa.left.y, &bb.left.y
		c = y1.Cmp(y2)
		if c != 0 {
			return c
		} else {
			return aa.kind - bb.kind
		}
	}*/
}

type bentleyTreeKey struct {
	segi int
	x    *Scalar
	y    *Scalar
}

func bentleyTreeCmp(a, b interface{}) int {
	aa, bb := a.(bentleyTreeKey), b.(bentleyTreeKey)

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

func debugPrintBentleyTree(tree *redblacktree.Tree, indent string) {
	vals := tree.Values()
	keys := tree.Keys()
	for i := 0; i < len(vals); i++ {
		k := keys[i].(bentleyTreeKey)
		v := vals[i].(int)
		fmt.Printf("%s=(%v,%v) -> %v\n", indent, k.segi, k.y, v)
	}
	fmt.Printf("\n")
}

func bentleyEventPs(i int, points []Vec2) (*Vec2, *Vec2) {
	p1 := &points[i]
	p2 := &points[(i+1)%len(points)]

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

type Intersection struct{ seg1, seg2 int }

func intersection(seg1, seg2 int) Intersection {
	if seg1 < seg2 {
		return Intersection{seg1, seg2}
	} else {
		return Intersection{seg2, seg1}
	}
}

// SegmentLoopIntersections implements the Bentley Ottmann algorithm for the case where
// the input segments are connected in a loop. The loop is implicitly closed
// by segment from last point to first point. The function returns all intersections
// except for the points in the original input (which could all be considered
// intersection points). Points at intersection of n distinct pairs
// of line segments appear n times in the output.
func SegmentLoopIntersections(points []Vec2) map[Intersection]*Vec2 {
	// Some useful pseudocode at https://www.hackerearth.com/practice/math/geometry/line-intersection-using-bentley-ottmann-algorithm/tutorial/
	// http://jeffe.cs.illinois.edu/teaching/373/notes/x06-sweepline.pdf
	// https://github.com/ideasman42/isect_segments-bentley_ottmann/blob/master/poly_point_isect.py

	events := binaryheap.NewWith(bentleyEventCmp)
	for i := 0; i < len(points); i++ {
		left, right := bentleyEventPs(i, points)
		if left.x.Cmp(&right.x) == 0 {
			events.Push(&bentleyEvent{
				kind:  vertical,
				i:     i,
				left:  left,
				right: right,
			})
		} else {
			events.Push(&bentleyEvent{
				kind:  start,
				i:     i,
				left:  left,
				right: right,
			})
			events.Push(&bentleyEvent{
				kind:  end,
				i:     i,
				left:  left,
				right: right,
			})
		}
	}

	tree := redblacktree.NewWith(bentleyTreeCmp)
	segToKey := make(map[int]bentleyTreeKey)

	intersections := make(map[Intersection]*Vec2)
	// Holds intersections which we need to keep track of, but which we
	// don't want to eventually report, because they lie at a segement vertex.
	vertIntersections := make(map[Intersection]bool)

	addIntersection := func(theint Intersection, p *Vec2) {
		s1a, s1b := &points[theint.seg1], &points[(theint.seg1+1)%len(points)]
		s2a, s2b := &points[theint.seg2], &points[(theint.seg2+1)%len(points)]
		if s1a.Eq(p) || s1b.Eq(p) || s2a.Eq(p) || s2b.Eq(p) {
			vertIntersections[theint] = true
		} else {
			intersections[theint] = p
		}
	}

	addCross := func(seg1, seg2 int, p *Vec2) {
		theint := intersection(seg1, seg2)
		exists := intersections[theint] != nil || vertIntersections[theint]
		addIntersection(theint, p)
		events.Push(&bentleyEvent{
			kind:     cross,
			i:        seg1,
			i2:       seg2,
			left:     p,
			right:    p,
			swapOnly: exists,
		})
	}

	for e, notEmpty := events.Pop(); notEmpty; e, notEmpty = events.Pop() {
		event := e.(*bentleyEvent)
		if event.deleted {
			continue
		}

		if event.kind == start {
			p1 := &points[event.i]
			p2 := &points[(event.i+1)%len(points)]
			y := SegmentYValueAtX(p1, p2, &event.left.x)

			tk := bentleyTreeKey{event.i, &event.left.x, y}
			it1, replaced := tree.PutAndGetIterator(tk, event.i)
			if replaced {
				panic("Internal error [1] in 'SegmentLoopIntersections'")
			}
			segToKey[event.i] = tk
			it2 := it1

			for it1.Prev() {
				prevI := it1.Value().(int)

				if !sameOrAdjacent(event.i, prevI, len(points)) {
					psp1 := &points[prevI]
					psp2 := &points[(prevI+1)%len(points)]
					p1 := &points[event.i]
					p2 := &points[(event.i+1)%len(points)]
					intersect, intersectionPoint := SegmentIntersection(psp1, psp2, p1, p2)
					if intersect {
						addCross(prevI, event.i, intersectionPoint)
					}

					break
				}
			}
			for it2.Next() {
				nextI := it2.Value().(int)

				if !sameOrAdjacent(event.i, nextI, len(points)) {
					nsp1 := &points[nextI]
					nsp2 := &points[(nextI+1)%len(points)]
					p1 := &points[event.i]
					p2 := &points[(event.i+1)%len(points)]
					intersect, intersectionPoint := SegmentIntersection(nsp1, nsp2, p1, p2)
					if intersect {
						addCross(nextI, event.i, intersectionPoint)
					}

					break
				}
			}
		} else if event.kind == vertical {
			p1a := &points[event.i]
			p1b := &points[(event.i+1)%len(points)]

			it := tree.Iterator()
			for it.Next() {
				segi := it.Value().(int)
				p2a := &points[segi]
				p2b := &points[(segi+1)%len(points)]

				intersect, ip := SegmentIntersection(p1a, p1b, p2a, p2b)
				if intersect {
					itn := intersection(segi, event.i)
					if !ip.Eq(p1a) && !ip.Eq(p1b) && !ip.Eq(p2a) && !ip.Eq(p2b) {
						intersections[itn] = ip
					} else {
						vertIntersections[itn] = true
					}
				}
			}
		} else if event.kind == end {
			it, f := tree.GetIterator(segToKey[event.i])
			if !f {
				panic(fmt.Sprintf("Internal error [2] in 'SegmentLoopIntersections': could not find key with seg index %v\n", event.i))
			}
			it1 := it
			it2 := it

			var ri, ti int = -1, -1

			for it1.Prev() {
				prevI := it1.Value().(int)

				if !sameOrAdjacent(event.i, prevI, len(points)) {
					ri = prevI
					break
				}
			}
			for it2.Next() {
				nextI := it2.Value().(int)

				if !sameOrAdjacent(event.i, nextI, len(points)) {
					ti = nextI
					break
				}
			}

			if ri != -1 && ti != -1 {
				p1a := &points[ri]
				p1b := &points[(ri+1)%len(points)]
				p2a := &points[ti]
				p2b := &points[(ti+1)%len(points)]

				intersect, intersectionPoint := SegmentIntersection(p1a, p1b, p2a, p2b)
				if intersect {
					addCross(ri, ti, intersectionPoint)
				}
			}

			tree.RemoveAt(it)
		} else if event.kind == cross {
			si := event.i
			ti := event.i2

			if si == ti {
				panic("Internal error [3] in 'SegementLoopIteration'")
			}

			sKey, tKey := segToKey[si], segToKey[ti]
			if bentleyTreeCmp(sKey, tKey) > 0 {
				si, ti = ti, si
				sKey, tKey = tKey, sKey
			}

			sIt, sItExists := tree.GetIterator(segToKey[si])
			tIt, tItExists := tree.GetIterator(segToKey[ti])

			if !(sItExists && tItExists) {
				panic(fmt.Sprintf("Internal error [4] in 'SegmentLoopIntersections' can't find %v or %v", si, ti))
			}

			if tree.Size() > 2 {
				if bentleyTreeCmp(tKey, sKey) == 0 {
					panic("Internal error [5] in 'SegmentLoopIntersections'")
				}

				tree.SwapAt(sIt, tIt)
				sIt, tIt = tIt, sIt
				segToKey[si], segToKey[ti] = tKey, sKey

				if !event.swapOnly {
					s1 := &points[si]
					s2 := &points[(si+1)%len(points)]
					t1 := &points[ti]
					t2 := &points[(ti+1)%len(points)]

					for sIt.Next() {
						u := sIt.Value().(int)
						if !sameOrAdjacent(u, si, len(points)) {
							u1 := &points[u]
							u2 := &points[(u+1)%len(points)]

							intersect, intersectionPoint := SegmentIntersection(s1, s2, u1, u2)
							if intersect {
								addCross(si, u, intersectionPoint)
							}

							break
						}
					}
					for tIt.Prev() {
						r := tIt.Value().(int)
						if !sameOrAdjacent(r, ti, len(points)) {
							r1 := &points[r]
							r2 := &points[(r+1)%len(points)]

							intersect, intersectionPoint := SegmentIntersection(t1, t2, r1, r2)
							if intersect {
								addCross(ti, r, intersectionPoint)
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

func SegmentLoopIntersectionsUsingNaiveAlgo(points []Vec2) map[Intersection]*Vec2 {
	intersections := make(map[Intersection]*Vec2)

	for si := 0; si < len(points); si++ {
		p := &points[si]
		for sj := si + 1; sj < len(points); sj++ {
			if sameOrAdjacent(si, sj, len(points)) {
				continue
			}

			q := &points[sj]

			pb := &points[(si+1)%len(points)]
			qb := &points[(sj+1)%len(points)]
			intersect, ip := SegmentIntersection(p, pb, q, qb)
			if intersect && !ip.Eq(p) && !ip.Eq(pb) && !ip.Eq(q) && !ip.Eq(qb) {
				intersections[intersection(si, sj)] = ip
			}
		}
	}

	return intersections
}

// DCEL represents a doubly-connected edge list.
// See pp. 31-32 of Mark de Berg, Marc van Kreveld, Mark Overmars, Otfried Schwarzkopf. 1998. Computational Geometry: Algorithms and Applications. Second, Revised Edition. Springer.
type DCEL struct {
	Faces []DCELFace
}

// DCELHalfEdge represents a half-edge within a doubly-connected edge list.
type DCELHalfEdge struct {
	Origin       *DCELVertex
	Twin         *DCELHalfEdge
	IncidentFace *DCELFace
}

// DCELVertex represents a vertex within a doubly-connected edge list.
type DCELVertex struct {
	P            *Vec2
	IncidentEdge *DCELHalfEdge
}

// DCELFace represents a face within a doubly connected edge list.
type DCELFace struct {
	OuterComponent  *DCELHalfEdge
	InnerComponents []*DCELHalfEdge
}

/*func DCELFromSegmentLoop(points []Vec2) DCEL {
	itns := SegmentLoopIntersections(points)

	itnWith := make(map[int][]int)
	for k := range itns {
		if itnWith[k.seg1] == nil {
			itnWith[k.seg1] = []int{k.seg2}
		} else {
			itnWith[k.seg1] = append(itnWith[k.seg1], k.seg2)
		}
		if itnWith[k.seg2] == nil {
			itnWith[k.seg2] = []int{k.seg1}
		} else {
			itnWith[k.seg2] = append(itnWith[k.seg2], k.seg1)
		}
	}

	faces := make([]DCELFace, 0)

	type trace struct {
		startSeg int
		face     *DCELFace
	}

	traces := make([]trace, 0)
	startFace := DCELFace{make([]*Vec2, 0)}
	currentTrace := trace{0, &startFace}

	for {
		first := true
		for i := currentTrace.startSeg; i < len(points) && (!first || i != currentTrace.startSeg); i, first = (i+1)%len(points), false {
			ss := itnWith[i]
			if ss == nil {
				currentTrace.face.Vertices = append(currentTrace.face.Vertices, &points[i])
			} else {
				for _, s := range ss {
					newVerts := []*Vec2{itns[intersection(s, i)]}
					newFace := DCELFace{newVerts}
					traces = append(traces, trace{(s + 1) % len(points), &newFace})
				}
			}
		}

		if

		if len(traces) == 0 {
			break
		}
		currentTrace, traces = traces[0], traces[1:]
	}
}*/
