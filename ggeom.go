package ggeom

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"unsafe" // just used to convert pointers to integers (but not back again!)

	"github.com/addrummond/ggeom/redblacktree"
	"github.com/emirpasic/gods/trees/binaryheap"
)

var _ = fmt.Printf

const debug = true

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

// Polygon2 represents a convex or nonconvex 2D polygon without holes
type Polygon2 struct {
	// wound anticlockwise
	verts []Vec2
}

// Polygon2WithHoles represents a convex or nonconvex 2D polygon with holes
type Polygon2WithHoles struct {
	outer *Polygon2
	holes []*Polygon2
}

func (a *Vec2) X() *Scalar {
	return &a.x
}

func (a *Vec2) Y() *Scalar {
	return &a.y
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

// ConvexOutlineOf returns the polygon consisting of the convex vertices of its argument.
// (Assumes that the polygon is wound anticlockwise.)
func ConvexOutlineOf(p *Polygon2) Polygon2 {
	newVerts := make([]Vec2, 0, len(p.verts))

	for i := 0; i < len(p.verts); i++ {
		var d1, d2 Vec2
		d1.Sub(&p.verts[i], &p.verts[(i+len(p.verts)-1)%len(p.verts)])
		d2.Sub(&p.verts[(i+1)%len(p.verts)], &p.verts[i])
		det := d1.Det(&d2)
		if det.Sign() >= 0 { // It's an counterclockwise turn, hence a convex vertex given that the polygon is wound counterclockwise
			newVerts = append(newVerts, *(p.verts[i].Copy()))
		}
	}

	return Polygon2{verts: newVerts}
}

// GetConvolutionCycle returns the sequence of vertices that forms
// the convolution cycle for the polyons p and q.
func GetConvolutionCycle(p *Polygon2, q *Polygon2) []Vec2 {
	// Roughly follows the algorithm described in
	//     Ron Wein. 2006. Exact and Efficient Construction of Planar Minkowski Sums using the Convolution Method. European Symposium on Algorithms, LNCS 4168, pp. 829-840.

	qq := ConvexOutlineOf(q)
	q = &qq

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

	if debug {
		for i := 1; i < len(cs); i++ {
			if cs[i].Eq(&cs[i-1]) {
				panic("Two identical adjacent points in convolution cycle in 'getConvolutionCycle'")
			}
		}
	}

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
		// cycles required when both of the polygons are non-convex.

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
func segmentsIntersectNoAdjacencyCheck(p1, p2, q1, q2 *Vec2) (bool, *Vec2) {
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
	} else if o3 == 0 && OnSegment(p1, q1, p2) {
		return true, q1
	} else if o4 == 0 && OnSegment(p1, q2, p2) {
		return true, q2
	} else {
		return false, nil
	}
}

func segmentsIntersectSpanNoAdjacencyCheck(p1, p2, q1, q2 *Vec2) (bool, *Vec2, *Vec2) {
	if FastSegmentsDontIntersect(p1, p2, q1, q2) {
		return false, nil, nil // the segments definitely don't intersect; won't be a degenerate case
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
		return true, nil, nil // the segments intersect; not degenerate
	} else if o1 == 0 && OnSegment(q1, p1, q2) {
		if OnSegment(p1, q1, p2) {
			return true, p1, q1
		} else if OnSegment(p1, q2, p2) {
			return true, p1, q2
		} else {
			return true, p1, p2
		}
	} else if o2 == 0 && OnSegment(q1, p2, q2) {
		if OnSegment(p1, q1, p2) {
			return true, p2, q1
		} else if OnSegment(p1, q2, p2) {
			return true, p2, q2
		} else {
			return true, p2, p1
		}
	} else if o3 == 0 && OnSegment(p1, q1, p2) {
		if OnSegment(q1, p1, q2) {
			return true, q1, p1
		} else if OnSegment(q1, p2, q2) {
			return true, q1, p2
		} else {
			return true, q1, q2
		}
	} else if o4 == 0 && OnSegment(p1, q2, p2) {
		if OnSegment(q1, p1, q2) {
			return true, q2, p1
		} else if OnSegment(q1, p2, q2) {
			return true, q2, p2
		} else {
			return true, q2, q1
		}
	} else {
		return false, nil, nil
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

// SegmentIntersectionFlagIntersects is set iff the segments intersect, incuding all weird/degenerate intersection cases
const SegmentIntersectionFlagIntersects = 1

// SegmentIntersectionFlagUnique is set iff the segments intersect and there is a unique intersection point
const SegmentIntersectionFlagUnique = 2

// SegmentIntersectionFlagAdjacent is set iff the segments intersect and are adjacent (no overlaping section, intersect at a vertex shared by the two)
const SegmentIntersectionFlagAdjacent = 4

// SegmentIntersectionFlagAtVertex is set iff the segments intersect and the intersection point is one of the segment vertices.
const SegmentIntersectionFlagAtVertex = 8

type SegmentIntersectionInfo struct {
	flags int
	p     *Vec2
}

// SegmentIntersection returns a SegmentIntersectionInfo
// structure, which contains all the information you would typically
// want about the intersection of two segments (if any).
func GetSegmentIntersectionInfo(p1, p2, q1, q2 *Vec2) SegmentIntersectionInfo {
	adj, pt := segmentsAdjacent(p1, p2, q1, q2)
	if adj {
		return SegmentIntersectionInfo{
			flags: SegmentIntersectionFlagIntersects |
				SegmentIntersectionFlagUnique |
				SegmentIntersectionFlagAdjacent |
				SegmentIntersectionFlagAtVertex,
			p: pt,
		}
	}

	var v Vec2
	intersect, degeneratePt := segmentsIntersectNoAdjacencyCheck(p1, p2, q1, q2)
	if intersect {
		if degeneratePt != nil {
			f := SegmentIntersectionFlagIntersects
			if degeneratePt.Eq(p1) || degeneratePt.Eq(p2) || degeneratePt.Eq(q1) || degeneratePt.Eq(q2) {
				f |= SegmentIntersectionFlagAtVertex
			}
			return SegmentIntersectionInfo{
				flags: f,
				p:     degeneratePt,
			}
		}

		pt := NondegenerateSegmentIntersection(p1, p2, q1, q2)
		f := SegmentIntersectionFlagIntersects | SegmentIntersectionFlagUnique
		if pt.Eq(p1) || pt.Eq(p2) || pt.Eq(q1) || pt.Eq(q2) {
			f |= SegmentIntersectionFlagAtVertex
		}
		return SegmentIntersectionInfo{
			flags: f,
			p:     pt,
		}
	}

	return SegmentIntersectionInfo{
		flags: 0,
		p:     &v,
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
//     (iii) the intersection point is not one of the segments' vertices.
func NonFunkySegmentIntersection(p1, p2, q1, q2 *Vec2) (bool, *Vec2) {
	var v Vec2

	shared := segmentsSharePoint(p1, p2, q1, q2)
	if shared != nil {
		return false, &v
	}

	intersect, degeneratePt := segmentsIntersectNoAdjacencyCheck(p1, p2, q1, q2)
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

	intersect, degeneratePt := segmentsIntersectNoAdjacencyCheck(p1, p2, q1, q2)
	if intersect {
		if degeneratePt != nil {
			return true, degeneratePt
		}

		return true, NondegenerateSegmentIntersection(p1, p2, q1, q2)
	}

	return false, nil
}

func SegmentIntersectionSpan(p1, p2, q1, q2 *Vec2) (int, [2]*Vec2) {
	adj, pt := segmentsAdjacent(p1, p2, q1, q2)
	if adj {
		return 1, [2]*Vec2{pt, nil}
	}

	intersect, degeneratePt1, degeneratePt2 := segmentsIntersectSpanNoAdjacencyCheck(p1, p2, q1, q2)
	if intersect {
		if degeneratePt1 != nil {
			return 2, [2]*Vec2{degeneratePt1, degeneratePt2}
			//return 1, [2]*Vec2{degeneratePt1, nil}
		}

		return 1, [2]*Vec2{NondegenerateSegmentIntersection(p1, p2, q1, q2), nil}
	}

	return 0, [2]*Vec2{nil, nil}
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
	start = 1
	cross = 2
	end   = 3
)

type bentleyEvent struct {
	kind     int
	i        int
	i2       int
	left     *Vec2
	right    *Vec2
	swapOnly bool
}

func bentleyEventCmp(a, b interface{}) int {
	aa, bb := a.(*bentleyEvent), b.(*bentleyEvent)

	x1, x2 := &aa.left.x, &bb.left.x
	if aa.kind == end {
		x1 = &aa.right.x
	}
	if bb.kind == end {
		x2 = &bb.right.x
	}

	c := x1.Cmp(x2)
	if c != 0 {
		return c
	} else {
		c = aa.kind - bb.kind
		if c != 0 {
			return c
		}

		y1, y2 := &aa.left.y, &bb.left.y
		return y1.Cmp(y2)
	}
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

func noCheckNeeded(s1, s2, l int, starts []int) bool {
	if sameOrAdjacent(s1, s2, l) {
		return true
	}

	if len(starts) == 0 {
		return false
	}

	// Check it's not one of the bogus segments from one loop to the next.
	for _, s := range starts {
		ss := s + 1
		if s1 == ss || s2 == ss {
			return true
		}
	}

	i := 0
	for i < len(starts) && s1 > starts[i] {
		i++
	}
	return s2 >= starts[i] || (i > 0 && s2 < starts[i-1])
}

type Intersection struct{ seg1, seg2 int }

func intersection(seg1, seg2 int) Intersection {
	if seg1 < seg2 {
		return Intersection{seg1, seg2}
	}

	return Intersection{seg2, seg1}
}

// SegmentLoopIntersections implements the Bentley Ottmann algorithm for the case where
// the input segments are connected in a loop. The loop is implicitly closed
// by segment from last point to first point. The function returns all intersections
// except for the points in the original input (which could all be considered
// intersection points). Points at intersection of n distinct pairs
// of line segments appear n times in the output. The function also returns the total
// number of pairs of lines for which an intersection test was made.
func SegmentLoopIntersections(points []Vec2, starts []int) (map[Intersection]*Vec2, int) {
	// Some useful pseudocode at https://www.hackerearth.com/practice/math/geometry/line-intersection-using-bentley-ottmann-algorithm/tutorial/
	// http://jeffe.cs.illinois.edu/teaching/373/notes/x06-sweepline.pdf
	// https://github.com/ideasman42/isect_segments-bentley_ottmann/blob/master/poly_point_isect.py

	checks := 0

	events := binaryheap.NewWith(bentleyEventCmp)
	for i := 0; i < len(points); i++ {
		left, right := bentleyEventPs(i, points)
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

	tree := redblacktree.NewWith(bentleyTreeCmp)
	segToKey := make(map[int]bentleyTreeKey)

	intersections := make(map[Intersection]*Vec2)
	crosses := make(map[Intersection]bool)

	addIntersection := func(theint Intersection, p *Vec2) {
		intersections[theint] = p
	}

	addCross := func(seg1, seg2 int, p *Vec2) {
		theint := intersection(seg1, seg2)
		addIntersection(theint, p)
		events.Push(&bentleyEvent{
			kind:     cross,
			i:        seg1,
			i2:       seg2,
			left:     p,
			right:    p,
			swapOnly: crosses[theint],
		})
		crosses[theint] = true
	}

	for e, notEmpty := events.Pop(); notEmpty; e, notEmpty = events.Pop() {
		event := e.(*bentleyEvent)

		if event.kind == start {
			tk := bentleyTreeKey{event.i, &event.left.x, &event.left.y}
			it1, replaced := tree.PutAndGetIterator(tk, event.i)
			if replaced {
				panic("Internal error [1] in 'SegmentLoopIntersections'")
			}
			segToKey[event.i] = tk
			it2 := it1

			for it1.Prev() {
				prevI := it1.Value().(int)

				if !noCheckNeeded(event.i, prevI, len(points), starts) {
					psp1 := &points[prevI]
					psp2 := &points[(prevI+1)%len(points)]
					p1 := &points[event.i]
					p2 := &points[(event.i+1)%len(points)]
					nPoints, points := SegmentIntersectionSpan(psp1, psp2, p1, p2)
					checks++
					for i := 0; i < nPoints; i++ {
						addCross(prevI, event.i, points[i])
					}

					if p1.y.Cmp(&event.left.y) != 0 && p2.y.Cmp(&event.left.y) != 0 {
						break
					}
				}
			}
			for it2.Next() {
				nextI := it2.Value().(int)

				if !noCheckNeeded(event.i, nextI, len(points), starts) {
					nsp1 := &points[nextI]
					nsp2 := &points[(nextI+1)%len(points)]
					p1 := &points[event.i]
					p2 := &points[(event.i+1)%len(points)]
					nPoints, points := SegmentIntersectionSpan(nsp1, nsp2, p1, p2)
					checks++
					for i := 0; i < nPoints; i++ {
						addCross(nextI, event.i, points[i])
					}

					if p1.y.Cmp(&event.left.y) != 0 && p2.y.Cmp(&event.left.y) != 0 {
						break
					}
				}
			}
		} else if event.kind == end {
			it, f := tree.GetIterator(segToKey[event.i])
			if !f {
				panic(fmt.Sprintf("Internal error [2] in 'SegmentLoopIntersections': could not find key with seg index %v\n", event.i))
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
						if !noCheckNeeded(u, si, len(points), starts) {
							u1 := &points[u]
							u2 := &points[(u+1)%len(points)]

							nPoints, points := SegmentIntersectionSpan(s1, s2, u1, u2)
							checks++
							for i := 0; i < nPoints; i++ {
								addIntersection(intersection(si, u), points[i])
								//addCross(si, u, points[i])
							}

							break
						}
					}
					for tIt.Prev() {
						r := tIt.Value().(int)
						if !noCheckNeeded(r, ti, len(points), starts) {
							r1 := &points[r]
							r2 := &points[(r+1)%len(points)]

							nPoints, points := SegmentIntersectionSpan(t1, t2, r1, r2)
							checks++
							for i := 0; i < nPoints; i++ {
								addIntersection(intersection(ti, r), points[i])
								//addCross(ti, r, points[i])
							}

							break
						}
					}
				}
			}
		}
	}

	fmt.Printf("Used %v checks compared to %v for naive algo\n", checks, (len(points)*len(points))/2)

	return intersections, checks
}

// SegmentLoopIntersectionsUsingNaiveAlgo compares every pair of segments
// in a given segment loop to find all intersections. It returns the intersections
// together with the total number of intersection tests that were made.
func SegmentLoopIntersectionsUsingNaiveAlgo(points []Vec2) (map[Intersection]*Vec2, int) {
	intersections := make(map[Intersection]*Vec2)

	checks := 0
	for si := 0; si < len(points); si++ {
		p := &points[si]
		for sj := si + 1; sj < len(points); sj++ {
			if sameOrAdjacent(si, sj, len(points)) {
				continue
			}

			q := &points[sj]

			pb := &points[(si+1)%len(points)]
			qb := &points[(sj+1)%len(points)]
			nPoints, points := SegmentIntersectionSpan(p, pb, q, qb)
			checks++
			for i := 0; i < nPoints; i++ {
				intersections[intersection(si, sj)] = points[i]
			}
		}
	}

	return intersections, checks
}

// EL represents a singly-linked edge list.
type EL struct {
	Faces []ELFace
}

// ELHalfEdge represents a half-edge within a doubly-connected edge list.
type ELHalfEdge struct {
	Origin       *ELVertex
	Twin         *ELHalfEdge
	IncidentFace *ELFace
	Next         *ELHalfEdge
}

// ELVertex represents a vertex within a doubly-connected edge list.
type ELVertex struct {
	P             *Vec2
	IncidentEdges []*ELHalfEdge
}

func vertexIndex(base, v *ELVertex) int {
	return int(uintptr(unsafe.Pointer(v))-uintptr(unsafe.Pointer(base))) / int(unsafe.Sizeof(ELVertex{}))
}

// ELFace represents a face within a doubly connected edge list.
type ELFace struct {
	OuterComponent  *ELHalfEdge
	InnerComponents []*ELHalfEdge
}

type intersectionWith struct {
	segi int
	p    *Vec2
}
type intersectionWithByXy struct {
	is []intersectionWith
	xd int // x direction (if -1, sorted with segments with smaller x values first)
	yd int // y direction (if -1, sorted with segments with smaller y values first)
}

func (is intersectionWithByXy) Len() int {
	return len(is.is)
}
func (is intersectionWithByXy) Swap(i, j int) {
	is.is[i], is.is[j] = is.is[j], is.is[i]
}
func (is intersectionWithByXy) Less(i, j int) bool {
	c := is.is[i].p.x.Cmp(&is.is[j].p.x)
	if c != 0 {
		if is.xd <= 0 {
			return c < 0
		}
		return c > 0
	}

	c = is.is[i].p.y.Cmp(&is.is[j].p.y)
	if is.yd <= 0 {
		return c < 0
	}
	return c > 0
}

func sameDirection(p1, p2, q1, q2 *Vec2) bool {
	pxd := p1.x.Cmp(&p2.x)
	pyd := p1.y.Cmp(&p2.y)
	qxd := q1.x.Cmp(&q2.x)
	qyd := q1.x.Cmp(&q2.y)

	return pxd == qxd && pyd == qyd
}

// sometimes useful for debugging 'HalfEdgesFromSegmentLoop'
func checkVertDuplicates(verts []ELVertex, callsite int) {
	fmt.Printf("Added [%v] (%v, %v)\n", callsite, verts[len(verts)-1].P.ApproxX(), verts[len(verts)-1].P.ApproxY())
	for i := 0; i < len(verts)-2; i++ {
		if verts[i].P.Eq(verts[len(verts)-1].P) {
			panic("Duplicate added!")
		}
	}
}

type vertexKey struct {
	p *Vec2
}

func vertexNodeCmp(a, b interface{}) int {
	aa := a.(vertexKey)
	bb := b.(vertexKey)
	yc := aa.p.Y().Cmp(bb.p.Y())
	if yc == 0 {
		return aa.p.X().Cmp(bb.p.X())
	}
	return yc
}

func HalfEdgesFromSegmentLoop(points []Vec2) (halfEdges []ELHalfEdge, vertices []ELVertex) {
	itns, _ := SegmentLoopIntersections(points, []int{})

	itnWith := make(map[int][]intersectionWith)
	for k, p := range itns {
		if itnWith[k.seg1] == nil {
			itnWith[k.seg1] = []intersectionWith{{k.seg2, p}}
		} else {
			itnWith[k.seg1] = append(itnWith[k.seg1], intersectionWith{k.seg2, p})
		}
		if itnWith[k.seg2] == nil {
			itnWith[k.seg2] = []intersectionWith{{k.seg1, p}}
		} else {
			itnWith[k.seg2] = append(itnWith[k.seg2], intersectionWith{k.seg1, p})
		}
	}

	// vertexTree -> *ELVertex
	vertexTree := redblacktree.NewWith(vertexNodeCmp)

	// In the worst case, each intersection splits two segments in two, and thus increases
	// the number of segments by three. We also have two half edges for every edge.
	maxNHalfEdges := (len(points) + (len(itns) * 3)) * 2
	halfEdges = make([]ELHalfEdge, 0, maxNHalfEdges)
	vertices = make([]ELVertex, 0, maxNHalfEdges)

	var prev *ELHalfEdge
	for segi := range points {
		p1 := &points[segi]
		p2 := &points[(segi+1)%len(points)]

		itns := itnWith[segi]
		// Sort the intersections by the position on the current segment.
		sort.Sort(intersectionWithByXy{itns, p1.x.Cmp(&p2.x), p1.y.Cmp(&p2.y)})

		var lastVert *ELVertex
		for i := -1; i < len(itns); i++ {
			var p *Vec2
			if i == -1 {
				p = p1
			} else {
				p = itns[i].p
			}

			if p.Eq(p2) {
				break
			}

			it, _ := vertexTree.PutIfNotExists(vertexKey{p}, func() interface{} {
				vertices = append(vertices, ELVertex{p, make([]*ELHalfEdge, 0, 2)})
				if len(vertices) > maxNHalfEdges {
					panic("Maximum length of 'vertices' exceeded in 'HalfEdgesFromSegmentLoop' [1]")
				}
				return &vertices[len(vertices)-1]
			})

			itnVert := it.Value().(*ELVertex)

			if itnVert == lastVert {
				continue
			}

			halfEdges = append(halfEdges, ELHalfEdge{
				Origin: itnVert,
				Next:   nil,
			})
			if len(halfEdges) > maxNHalfEdges {
				panic("Maximum length of 'halfEdges' exceeded in 'HalfEdgesFromSegmentLoop' [0]")
			}
			he := &halfEdges[len(halfEdges)-1]

			itnVert.IncidentEdges = append(itnVert.IncidentEdges, he)

			if prev != nil {
				prev.Next = he
			}

			prev = he
			lastVert = itnVert
		}
	}

	if len(halfEdges) < 2 {
		panic("Fewer half edges than expected in 'HalfEdgesFromSegmentLoop'")
	}

	// Tie the knot.
	last := &halfEdges[len(halfEdges)-1]
	first := &halfEdges[0]
	first.Origin.IncidentEdges = append(first.Origin.IncidentEdges, last)
	halfEdges[len(halfEdges)-1].Next = first

	if debug {
		for i, v1 := range vertices {
			for j, v2 := range vertices {
				if i != j && v1.P.Eq(v2.P) {
					panic(fmt.Sprintf("Unexpected vertex point equality in 'HalfEdgesFromSegmentLoop': point=(%v,%v) %v,%v", v1.P.ApproxX(), v2.P.ApproxY(), i, j))
				}
			}
		}
	}

	return halfEdges, vertices
}

func traceFrom(prev []*ELVertex, base, v *ELVertex, visitCount []int8) [][]*ELVertex {
	if visitCount[vertexIndex(base, v)] > 0 {
		return [][]*ELVertex{}
	}

	visitCount[vertexIndex(base, v)]++

	if len(prev) == 0 {
		for _, ie := range v.IncidentEdges {
			r := traceFrom([]*ELVertex{v}, base, ie.Next.Origin, visitCount)
			if len(r) > 0 {
				return r
			}
		}
	} else {
		prevV := prev[len(prev)-1]

		var bestNextV *ELVertex
		var bestCp *Scalar
		for _, ie := range v.IncidentEdges {
			nextV := ie.Next.Origin

			if nextV == prevV || nextV == v {
				continue
			}

			if visitCount[vertexIndex(base, nextV)] > 0 {
				// Is this a cycle?
				for i := len(prev) - 1; i >= 0; i-- {
					pv := prev[i]

					if pv == nextV {
						theCycle := make([]*ELVertex, len(prev)-i)
						copy(theCycle, prev[i:len(prev)])
						theCycle = append(theCycle, v)
						for _, v := range theCycle {
							visitCount[vertexIndex(base, v)] = math.MaxInt8
						}
						return [][]*ELVertex{theCycle}
					}
				}

				continue
			}

			var lastDir, newDir Vec2
			lastDir.Sub(v.P, prevV.P)
			newDir.Sub(nextV.P, v.P)

			cp := lastDir.Det(&newDir)

			if cp.Sign() <= 0 { // clockwise turn
				if bestNextV == nil {
					bestNextV = nextV
					bestCp = cp
				} else if cp.Cmp(bestCp) < 0 {
					bestNextV = nextV
					bestCp = cp
				}
			}
		}

		if bestNextV != nil {
			newPrev := make([]*ELVertex, len(prev), len(prev)+1)
			copy(newPrev, prev)
			newPrev = append(newPrev, v)
			r := traceFrom(newPrev, base, bestNextV, visitCount)
			if len(r) > 0 {
				return r
			}
		}
	}

	visitCount[vertexIndex(base, v)]--

	return [][]*ELVertex{}
}

// See https://pdfs.semanticscholar.org/a2d2/186d8b6481be81eed75857d831b32a2ad940.pdf
// p. 16 section 3.1.1
// for some indication that this algorithm that I made up is on the right track
func traceInnies(vertices []ELVertex, outline []*ELVertex) [][]*ELVertex {
	traces := make([][]*ELVertex, 0)
	visitCount := make([]int8, len(vertices), len(vertices))

	for _, v := range outline {
		visitCount[vertexIndex(&vertices[0], v)] = math.MaxInt8
	}

	for i := range vertices {
		if visitCount[i] > 0 {
			continue
		}

		traces = append(traces, traceFrom([]*ELVertex{}, &vertices[0], &vertices[i], visitCount)...)

		visitCount[i] = math.MaxInt8
	}

	return traces
}

// Implements arrays of booleans using bit arrays.
func makeBoolArray(size int) []uint64 {
	return make([]uint64, (size/65)+1)
}
func indexBoolArray(a []uint64, i int) bool {
	ii := uint(i)
	return (a[ii/64]>>(ii%64))&1 != 0
}
func setBoolArray(a []uint64, i int, v bool) {
	ii := uint(i)
	a[i/64] |= 1 << (ii % 64)
}
func clearBoolArray(a []uint64) {
	for i := 0; i < len(a); i++ {
		a[i] = 0
	}
}

// traceOutline gets the outline of a convolution from the list
// of vertices in the EL. This turns out to be quite simple.
// All we have to do is take the "most clockwise" turn available at
// every intersection. We know that no hole will ever share an
// edge with the outline. Thus, by excluding edges on the outline
// from our subsequent search for cycles, we can reduce the size
// of the graph and speed up the computation somewhat. (Since any
// cycles that we miss will be non-hole cycles which we can
// safely ignore.)
// Assumes that vertex indices start at zero.
func traceOutline(vertices []ELVertex) []*ELVertex {
	currentVertex := &vertices[0]
	var prevVertex *ELVertex

	trace := make([]*ELVertex, 0)
	visited := makeBoolArray(len(vertices))

	for i := 0; prevVertex == nil || currentVertex != &vertices[0]; i++ {
		fmt.Printf("LOOP WITH %v\n", vertexIndex(&vertices[0], currentVertex))

		if i > len(vertices) {
			panic("Too many loop iterations in 'TraceOutine'")
		}

		trace = append(trace, currentVertex)

		if prevVertex == nil {
			prevVertex = currentVertex
			currentVertex = currentVertex.IncidentEdges[0].Next.Origin
		} else if len(currentVertex.IncidentEdges) == 1 {
			for _, e := range currentVertex.IncidentEdges {
				if e.Origin == currentVertex {
					prevVertex = currentVertex
					currentVertex = e.Next.Origin
					break
				}
			}
		} else {
			var currentDirectionVec Vec2
			currentDirectionVec.Sub(currentVertex.P, prevVertex.P)
			currentDirection := &currentDirectionVec

			// 'best' will end up being set to the vertex corresponding to the
			// most clockwise possible turn.

			var bestCp *Scalar
			var best *ELVertex

			if i != 0 {
				clearBoolArray(visited)
			}

			for j := 0; j < len(currentVertex.IncidentEdges); j++ {
				ie := currentVertex.IncidentEdges[j]
				v := ie.Next.Origin
				if v == currentVertex || indexBoolArray(visited, vertexIndex(&vertices[0], v)) {
					continue
				}
				setBoolArray(visited, vertexIndex(&vertices[0], v), true)

				var d Vec2
				d.Sub(v.P, currentVertex.P)
				det := currentDirection.Det(&d)
				if bestCp == nil || det.Cmp(bestCp) < 0 {
					bestCp = det
					best = v
				}
			}

			if best == nil {
				panic("Unexpected nil value for 'best' in 'traceOutline'")
			}

			prevVertex = currentVertex
			currentVertex = best
		}
	}

	return trace
}

type intermediateSort struct {
	xord, yord int
	points     []*Vec2
}

func (is intermediateSort) Len() int {
	return len(is.points)
}

func (is intermediateSort) Swap(i, j int) {
	is.points[i], is.points[j] = is.points[j], is.points[i]
}

func (is intermediateSort) Less(i, j int) bool {
	if is.xord == 0 && is.yord == 0 {
		panic("'xord' and 'yord' are both zero!")
	}

	if is.xord == 0 {
		return is.points[i].y.Cmp(&is.points[j].y) == is.yord
	}
	if is.yord == 0 {
		return is.points[i].x.Cmp(&is.points[j].x) == is.xord
	}

	c := is.points[i].x.Cmp(&is.points[j].x)
	if c == 0 {
		return is.points[i].y.Cmp(&is.points[j].y) == is.yord
	}
	return c == is.xord
}

func PointInsidePolygon(point *Vec2, polygon *Polygon2) bool {
	// Construct a horizontal segment from the point extending rightward to
	// a point beyond the rightmost vertex of the polygon. If this segment
	// intersects an odd number of the polygon's segments, the point lies
	// inside the polygon.

	crossings := 0
	for i := 0; i < len(polygon.verts); i++ {
		p1 := &polygon.verts[i]
		p2 := &polygon.verts[(i+1)%len(polygon.verts)]
		c := p1.y.Cmp(&p2.y)
		if c == 0 {
			// This segment of the polygon is horizontal, so it's just a question
			// of whether it has the same y coordinate and whether there's horizontal
			// overlap.
			if p1.y.Cmp(&point.y) == 0 && (p1.x.Cmp(&point.x) >= 0 || p2.x.Cmp(&point.x) >=0) {
				crossings++
			}
		} else {
			if p1.y.Cmp(&point.y) == c && p2.y.Cmp(&point.y) == -c {
				// The polygon segment crosses the horizontal line vertically.
				// If both points overlap horizontally with the horizontal segment,
				// the we know immediately that there's an intersection.
				c1 := p1.x.Cmp(&point.x)
				c2 := p2.x.Cmp(&point.x)
				if c1 >= 0 && c2 >= 0 {
					crossings++
				} else if c1 >=0 || c2 >= 0 {
					// This is the more complicated case. The segment crosses the horizontal
					// line vertically, but we have to figure out whether or not it's too far
					// to the left to touch it. This requires doing some actual arithmetic,
					// which hopefully won't happen too often.
					var ydiff, xdiff, yd Scalar
					ydiff.Sub(&p1.y, &p2.y)
					xdiff.Sub(&p2.x, &p1.x)
					yd.Sub(&p1.x, &point.x)
					ydiff.Inv(&ydiff)
					yd.Mul(&yd, &ydiff)
					yd.Mul(&yd, &xdiff)
					yd.Add(&p1.x, &xdiff)
					if yd.Cmp(&point.x) >= 0 {
						crossings++
					}
				}
			}
		}
	}

	return (crossings % 2) == 1
}

/*
func WeilerAtherton(polygon1, polygon2 *Polygon2) {
	// Shallow copying Vec2s is ok if we don't modify them.
	n := len(polygon1.verts) + len(polygon2.verts)
	combined := make([]Vec2, n, n)
	copy(combined, polygon1.verts)
	copy(combined[len(polygon1.verts):len(combined)], polygon2.verts)

	intersections, _ := SegmentLoopIntersections(combined, []int{len(polygon1.verts)})
	intermediates := make([][]*Vec2, n, n)

	for itn, v := range intersections {
		intermediates[itn.seg1] = append(intermediates[itn.seg1], v)
		intermediates[itn.seg2] = append(intermediates[itn.seg2], v)
	}

	for i, im := range intermediates {
		v1 := &combined[i]
		v2 := &combined[(i+1)%len(combined)]
		sort.Sort(intermediateSort{v1.x.Cmp(&v2.x), v1.y.Cmp(&v2.y), im})
	}

}
*/

func ElementaryCircuits(vertices []ELVertex) [][]*ELVertex {
	outline := traceOutline(vertices)
	innies := traceInnies(vertices, outline)
	r := make([][]*ELVertex, 0, len(innies)+1)
	return append(append(r, outline), innies...)
}
