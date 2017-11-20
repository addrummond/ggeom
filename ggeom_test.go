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
	if len(strips) == 0 {
		return
	}

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
			p1 := &s[i-1]
			p2 := &s[i%len(s)]
			var d Vec2
			d.Sub(p2, p1)
			if d.x.Sign() == 0 && d.y.Sign() == 0 {
				continue
			}

			const sinv = 0.309
			const cosv = -0.951
			dx := d.ApproxX()
			dy := d.ApproxY()
			d1 := ApproxVec2(dx*cosv-dy*sinv, dx*sinv+dy*cosv).ApproxScale(arrowLen)
			d2 := ApproxVec2(dx*cosv+dy*sinv, dx*-sinv+dy*cosv).ApproxScale(arrowLen)
			var h1, h2 Vec2
			h1.Add(p2, &d1)
			h2.Add(p2, &d2)
			canvas.Line(tx(p2.ApproxX()), ty(p2.ApproxY()), tx(h1.ApproxX()), ty(h1.ApproxY()), format)
			canvas.Line(tx(p2.ApproxX()), ty(p2.ApproxY()), tx(h2.ApproxX()), ty(h2.ApproxY()), format)
		}
	}

	canvas.End()
}

func debugHalfEdgeGraphToHtmlAnimation(start *DCELVertex, width int, height int) string {
	var o string
	w := float64(width)
	h := float64(height)

	var minx, maxx, miny, maxy = math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
	followed := make(map[*DCELHalfEdge]bool)
	var findBounds func(vert *DCELVertex)
	findBounds = func(vert *DCELVertex) {
		x := vert.P.ApproxX()
		y := vert.P.ApproxY()

		if x < minx {
			minx = x
		}
		if x > maxx {
			maxx = x
		}
		if y < miny {
			miny = y
		}
		if y > maxy {
			maxy = y
		}

		for _, edge := range vert.IncidentEdges {
			if !followed[edge] && edge.Forward && edge.Origin.P.Eq(vert.P) {
				followed[edge] = true
				findBounds(edge.Twin.Origin)
			}
		}
	}
	findBounds(start)

	maxx *= 1.1
	minx *= 1.1
	maxy *= 1.1
	miny *= 1.1

	tx := func(x float64) float64 {
		return ((x - minx) / (maxx - minx)) * w
	}
	ty := func(y float64) float64 {
		return h - (((y - miny) / (maxy - miny)) * h)
	}

	var vindices string
	var coords string
	var lines string
	var incident string
	var incidentCoords string
	followed = make(map[*DCELHalfEdge]bool)

	var traverse func(vert, from *DCELVertex)
	traverse = func(vert, from *DCELVertex) {
		if from != nil {
			if len(lines) > 0 {
				vindices += ", "
				coords += ", "
				lines += ", "
				incident += ", "
				incidentCoords += ", "
			}
			vindices += fmt.Sprintf("%v", from.Index)
			coords += fmt.Sprintf("[[%v,%v],[%v,%v]]", from.P.ApproxX(), from.P.ApproxY(), vert.P.ApproxX(), vert.P.ApproxY())
			lines += fmt.Sprintf("[[%v,%v],[%v,%v]]", tx(from.P.ApproxX()), ty(from.P.ApproxY()), tx(vert.P.ApproxX()), ty(vert.P.ApproxY()))
			incident += "["
			incidentCoords += "["
			count := 0
			for _, ie := range from.IncidentEdges {
				if ie.Forward {
					if count != 0 {
						incident += ", "
						incidentCoords += ", "
					}
					incident += fmt.Sprintf("[[%v,%v],[%v,%v]]", tx(ie.Origin.P.ApproxX()), ty(ie.Origin.P.ApproxY()), tx(ie.Twin.Origin.P.ApproxX()), ty(ie.Twin.Origin.P.ApproxY()))
					incidentCoords += fmt.Sprintf("[[%v,%v],[%v,%v]]", ie.Origin.P.ApproxX(), ie.Origin.P.ApproxY(), ie.Twin.Origin.P.ApproxX(), ie.Twin.Origin.P.ApproxY())
					count++
				}
			}
			incident += "]"
			incidentCoords += "]"
		}
		for _, edge := range vert.IncidentEdges {
			if !followed[edge] && edge.Forward && edge.Origin.P.Eq(vert.P) {
				followed[edge] = true
				traverse(edge.Twin.Origin, vert)
			}
		}
	}

	traverse(start, nil)

	cw := fmt.Sprintf("%v", w)
	ch := fmt.Sprintf("%v", h)

	o += `<html>
<body>
<p id='info' style='height: 1.5em'>
<p>Go to vertex <input id='gotovertex' type='text' width='3'>
<p>
<canvas id='canvas' width='` + cw + `' height='` + ch + `'></canvas>
<script>
var info = document.getElementById('info');

var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var vindices = [` + vindices + `]
var coords = [` + coords + `];
var lines = [` + lines + `];
var incident = [` + incident + `];
var incidentCoords = [` + incidentCoords + `];
var i = 0;

function draw() {
    for (var j = 0; j < lines.length; ++j) {
		ctx.lineWidth = 6;		
        if (j < i)
            ctx.strokeStyle = '#aaaaaa';
        else if (j == i)
			continue;
		else if (j > i)
		    ctx.strokeStyle = '#eeeeee';
        ctx.beginPath();
        ctx.moveTo(lines[j][0][0], lines[j][0][1]);
        ctx.lineTo(lines[j][1][0], lines[j][1][1]);
		ctx.stroke();

		ctx.lineWidth = 1;
		ctx.strokeStyle = '#000000';
		var sinv = 0.309
		var cosv = -0.951
		var dx = lines[j][1][0] - lines[j][0][0]
		var dy = lines[j][1][1] - lines[j][0][1]
		var a1x = dx*cosv-dy*sinv
		var a1y = dx*sinv+dy*cosv
		var a2x = dx*cosv+dy*sinv
		var a2y = dx*-sinv+dy*cosv
		var l = Math.sqrt(dx*dx + dy*dy)/20
		a1x /= l
		a1y /= l
		a2x /= l
		a2y /= l
		ctx.beginPath();
		ctx.moveTo(lines[j][1][0] + a1x, lines[j][1][1] + a1y);
		ctx.lineTo(lines[j][1][0], lines[j][1][1])
		ctx.lineTo(lines[j][1][0] + a2x, lines[j][1][1] + a2y);
		ctx.stroke();
	}

	ctx.strokeStyle = '#000000';
	ctx.beginPath();
	ctx.moveTo(lines[i][0][0], lines[i][0][1]);
	ctx.lineTo(lines[i][1][0], lines[i][1][1]);
	ctx.stroke();
	ctx.fillStyle = '#000000';	
	ctx.beginPath();
	ctx.arc(lines[i][0][0], lines[i	][0][1], 8, 0, 2*Math.PI, false);
	ctx.fill();
	
	var incidentEdges = incident[i];
	ctx.lineWidth = 2;
	for (var j = 0; j < incidentEdges.length; ++j) {
		ctx.strokeStyle = ['#ee0000', '#00ee00', '#0000ee', '#eeee00', '#0000ee', '#ee00ee'][j % incidentEdges.length]
		ctx.beginPath()
		ctx.moveTo(incidentEdges[j][0][0], incidentEdges[j][0][1]);
		ctx.lineTo(incidentEdges[j][1][0], incidentEdges[j][1][1]);
		ctx.stroke();
	}

	var incidentEdgesCoords = incidentCoords[i];
	var isegs = "";
	for (var j = 0; j < incidentEdges.length; ++j) {
		if (j != 0)
			isegs += "  ";
		isegs += "(" + incidentEdgesCoords[j][0][0] + "," + incidentEdgesCoords[j][0][1] + " -> " + incidentEdgesCoords[j][1][0] + "," + incidentEdgesCoords[j][1][1] + ")";
	}
    info.innerHTML = "";
    info.appendChild(document.createTextNode(vindices[i] + ": from (" + coords[i][0][0] + "," + coords[i][0][1] + ") to (" + coords[i][1][0] + "," + coords[i][1][1] + "); incident = " + isegs));
}

document.onkeydown = function (e) {
    if (e.keyCode == 37) { // left
        e.preventDefault();
        if (i > 0) {
            --i;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            draw();
        }
    } else if (e.keyCode == 39) { // right
        e.preventDefault();
        if (i < lines.length - 1) {
            ++i;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            draw();
        }
    }
}

var goToVertex = document.getElementById("gotovertex");
goToVertex.onchange = function (e) {
	e.preventDefault();
	var v = parseInt(goToVertex.value);
	if (! isNaN(v)) {
		var j;
		for (j = 0; j < vindices.length; ++j) {
			if (vindices[j] == v) {
				break;
			}
		}

		if (j < vindices.length && j != i) {
			i = j;
			draw();
		}
	}
}

draw();
</script>
</body>
</html>`

	return o
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
		if IsBetweenAnticlockwise(&c[0], &c[1], &c[2]) {
			t.Error()
		}
		if !IsBetweenAnticlockwise(&c[2], &c[1], &c[0]) {
			t.Error()
		}
	}

	for _, c := range trueCases {
		if !IsBetweenAnticlockwise(&c[0], &c[1], &c[2]) {
			t.Error()
		}
		if IsBetweenAnticlockwise(&c[2], &c[1], &c[0]) {
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
		{{3, 0}, {-2, 2}, {2, -2}, {2, 2}, {2, 0.4}},            // A particular case that was being computed incorrectly
	})

	for _, tst := range tests {
		p := NondegenerateSegmentIntersection(&tst[0], &tst[1], &tst[2], &tst[3])
		//fmt.Printf("INT: %f %f\n", p.ApproxX(), p.ApproxY())
		if !p.SlowEqEpsilon(&tst[4], EPSILON) {
			t.Error()
		}
	}
}

// Just checks some particular cases that were buggy at some point.
func TestSegmentIntersection(t *testing.T) {
	intersectingTests := SofSofVec2([][][]float64{
		{{1, 1}, {4, 1}, {-1, -1}, {3, 1}, {3, 1}},
	})

	for i, tst := range intersectingTests {
		intersects, p := SegmentIntersection(&tst[0], &tst[1], &tst[2], &tst[3])
		if !intersects {
			t.Errorf("Intersecting test %v: expected intersection, none found", i)
		}
		if !p.Eq(&tst[4]) {
			t.Errorf("Intersecting test %v; incorrect intersection point computed", i)
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
		{{3, 0}, {-2, 2}, {2, -2}, {2, 2}, {2, 0.4}},                     // A particular case that was being computed incorrectly
	})

	for i, tst := range tests {
		info := GetSegmentIntersectionInfo(&tst[0], &tst[1], &tst[2], &tst[3])
		if info.intersect {
			if info.unique {
				if !info.p.SlowEqEpsilon(&tst[4], EPSILON) {
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
		{{0, 0}, {0, 0}, {0, 0}, {2, 0}, {2, 0.4}, {0.8571428571428571, 0.8571428571428571}},
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
		its2, _ := SegmentLoopIntersections(ps)

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
		for segs, p := range its2 {
			xf, _ := p.x.Float64()
			yf, _ := p.y.Float64()
			fmt.Printf("\n  %v, %v [%v,%v] of (%v,%v) -> (%v,%v)  with  (%v,%v) -> (%v,%v)\n", xf, yf, segs.seg1, segs.seg2, ps[segs.seg1].ApproxX(), ps[segs.seg1].ApproxY(), ps[(segs.seg1+1)%len(ps)].ApproxX(), ps[(segs.seg1+1)%len(ps)].ApproxY(), ps[segs.seg2].ApproxX(), ps[segs.seg2].ApproxY(), ps[(segs.seg2+1)%len(ps)].ApproxX(), ps[(segs.seg2+1)%len(ps)].ApproxY())
		}
		fmt.Printf("\n")

		if len(its1) != len(its2) {
			t.Fail()
		}

		used := make(map[int]bool)
		for _, p1 := range its2 {
			found := false
			for i, p2 := range its1 {
				if !used[i] && p1.SlowEqEpsilon(&p2, EPSILON) {
					found = true
					used[i] = true
					break
				}
			}
			if !found {
				t.Fail()
			}
		}
	}
}

var exampleLoops = SofSofVec2([][][]float64{
	{{10, 10}, {-10, 10}, {-10, -10}, {10, -10}},                                 // p
	{{0, 2}, {-1, 0}, {1, 0}},                                                    // q
	{{-11, -10}, {-9, -10}, {11, -10}, {11, 10}, {10, 12}, {-10, 12}, {-11, 10}}, // convolution cycle
	/////
	{{10, 10}, {-10, 10}, {-10, -10}, {10, -10}, {0, -5}},
	{{0, 2}, {-1, 0}, {1, 0}},
	{{-11, -10}, {9, -10}, {11, -10}, {10, -8}, {0, -3}, {-1, -5}, {1, -5}, {11, 10}, {10, 12}, {-10, 12}, {-11, 10}},
	/////
	{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
	{{0.6, 0.6}, {-0.6, 0.6}, {-0.6, -0.6}, {0.6, -0.6}},
	{{-0.6, -0.6}, {3.4, -0.6}, {4.6, -0.6}, {4.6, 0.4}, {4.6, 1.6}, {1.6, 1.6}, {0.4, 1.6}, {0.4, 0.4}, {1.6, 0.4}, {1.6, 1.4}, {2.6, 2.4}, {2.6, 3.6}, {1.4, 3.6}, {1.4, 2.4}, {2.4, 2.4}, {3.4, 1.4}, {4.6, 1.4}, {4.6, 2.4}, {4.6, 3.6}, {3.6, 4.6}, {0.6, 4.6}, {-0.6, 4.6}, {-0.6, 0.6}},
	/////
	{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
	{{0.2, 0.2}, {0, 0.15}, {-0.2, 0.2}, {-0.2, -0.2}, {0.2, -0.2}},
	{{-0.2, -0.2}, {3.8, -0.2}, {4.2, -0.2}, {4.2, 0.8}, {4.2, 1.2}, {1.2, 1.2}, {1, 1.15}, {1, 2.15}, {2, 3.15}, {3, 3.15}, {4, 2.15}, {4, 3.15}, {3, 4.15}, {2.8, 4.2}, {-0.2, 4.2}, {-0.2, 0.2}, {-0.2, -0.2}, {0, 4.15}, {0, 0.15}, {4, 0.15}, {4, 1.15}, {3.8, 1.2}, {0.8, 1.2}, {0.8, 0.8}, {1.2, 0.8}, {1.2, 1.8}, {2.2, 2.8}, {2.2, 3.2}, {2, 3.15}, {1.8, 3.2}, {1.8, 2.8}, {2.8, 2.8}, {3.8, 1.8}, {4.2, 1.8}, {4.2, 2.8}, {4.2, 3.2}, {3.2, 4.2}, {0.2, 4.2}},
	/////
	{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
	{{0.3, 0.3}, {0, 0.2}, {-0.3, 0.3}, {-0.2, 0}, {-0.3, -0.3}, {0, -0.2}, {0.3, -0.3}},
	{{-0.3, -0.3}, {3.7, -0.3}, {4, -0.2}, {4, 0.8}, {1, 0.8}, {1.3, 0.7}, {1.3, 1.3}, {1, 1.2}, {1, 2.2}, {0.7, 2.3}, {0.8, 2}, {0.7, 1.7}, {1, 1.8}, {1.3, 1.7}, {2.3, 2.7}, {2.3, 3.3}, {2, 3.2}, {1.7, 3.3}, {1.8, 3}, {1.7, 2.7}, {2.7, 2.7}, {3, 2.8}, {4, 1.8}, {4.3, 1.7}, {4.3, 2.3}, {4.3, 3.3}, {3.3, 4.3}, {0.3, 4.3}, {0, 4.2}, {0, 0.2}, {4, 0.2}, {4, 1.2}, {3.7, 1.3}, {0.7, 1.3}, {0.8, 1}, {0.7, 0.7}, {1, 0.8}, {1, 1.8}, {2, 2.8}, {2.3, 2.7}, {3.3, 2.7}, {3.3, 3.3}, {3, 3.2}, {2.7, 3.3}, {2.8, 3}, {2.7, 2.7}, {3.7, 1.7}, {4, 1.8}, {4, 2.8}, {3, 3.8}, {0, 3.8}, {0, -0.2}, {0.3, -0.3}, {4.3, -0.3}, {4.3, 0.3}, {4.3, 1.3}, {1.3, 1.3}, {1.3, 2.3}, {1, 2.2}, {2, 3.2}, {3, 3.2}, {4, 2.2}, {4, 3.2}, {3, 4.2}, {2.7, 4.3}, {-0.3, 4.3}, {-0.3, 0.3}, {-0.2, 0}, {3.8, 0}, {3.8, 1}, {0.8, 1}, {0.8, 2}, {1.8, 3}, {2.8, 3}, {3.8, 2}, {3.8, 3}, {2.8, 4}, {-0.2, 4}, {-0.3, 3.7}},
	/////
	{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {4, 3}},
	{{0.6, 0.6}, {0, 0.4}, {-0.6, 0.6}, {-0.4, 0}, {-0.6, -0.6}, {0, -0.4}, {0.6, -0.6}},
	{{-0.6, -0.6}, {3.4, -0.6}, {4, -0.4}, {4, 0.6}, {1, 0.6}, {1.6, 0.4}, {1.6, 1.6}, {1, 1.4}, {1, 2.4}, {0.4, 2.6}, {0.6, 2}, {0.4, 1.4}, {1, 1.6}, {1.6, 1.4}, {2.6, 2.4}, {2.6, 3.6}, {2, 3.4}, {1.4, 3.6}, {1.6, 3}, {1.4, 2.4}, {2.4, 2.4}, {3, 2.6}, {4, 1.6}, {4.6, 1.4}, {4.6, 2.6}, {4.6, 3.6}, {3.6, 4.6}, {0.6, 4.6}, {0, 4.4}, {0, 0.4}, {4, 0.4}, {4, 1.4}, {3.4, 1.6}, {0.4, 1.6}, {0.6, 1}, {0.4, 0.4}, {1, 0.6}, {1, 1.6}, {2, 2.6}, {2.6, 2.4}, {3.6, 2.4}, {3.6, 3.6}, {3, 3.4}, {2.4, 3.6}, {2.6, 3}, {2.4, 2.4}, {3.4, 1.4}, {4, 1.6}, {4, 2.6}, {3, 3.6}, {0, 3.6}, {0, -0.4}, {0.6, -0.6}, {4.6, -0.6}, {4.6, 0.6}, {4.6, 1.6}, {1.6, 1.6}, {1.6, 2.6}, {1, 2.4}, {2, 3.4}, {3, 3.4}, {4, 2.4}, {4, 3.4}, {3, 4.4}, {2.4, 4.6}, {-0.6, 4.6}, {-0.6, 0.6}, {-0.4, 0}, {3.6, 0}, {3.6, 1}, {0.6, 1}, {0.6, 2}, {1.6, 3}, {2.6, 3}, {3.6, 2}, {3.6, 3}, {2.6, 4}, {-0.4, 4}, {-0.6, 3.4}},
	/////
	{{0.6, 0.6}, {0, 0.4}, {-0.6, 0.6}, {-0.4, 0}, {-0.6, -0.6}, {0, -0.4}, {0.6, -0.6}},
	{{0.06, 0.06}, {0, 0.04}, {-0.06, 0.06}, {-0.04, 0}, {-0.06, -0.06}, {0, -0.04}, {0.06, -0.06}},
	{{-0.6599999999999999, -0.6599999999999999}, {-0.06, -0.46}, {0, -0.44}, {0.06, -0.46}, {0.06, -0.34}, {0, -0.36000000000000004}, {-0.06, -0.34}, {-0.04, -0.4}, {-0.06, -0.46}, {0.54, -0.6599999999999999}, {0.6, -0.64}, {0.6, 0.5599999999999999}, {0, 0.36000000000000004}, {0.06, 0.34}, {0.06, 0.46}, {0, 0.44}, {-0.06, 0.46}, {-0.04, 0.4}, {-0.06, 0.34}, {0, 0.36000000000000004}, {-0.6, 0.5599999999999999}, {-0.4, -0.04}, {-0.34, -0.06}, {-0.34, 0.06}, {-0.4, 0.04}, {-0.46, 0.06}, {-0.44, 0}, {-0.46, -0.06}, {-0.4, -0.04}, {-0.6, -0.64}, {-0.54, -0.6599999999999999}, {0.06, -0.46}, {0.6599999999999999, -0.6599999999999999}, {0.6599999999999999, -0.54}, {0.6599999999999999, 0.6599999999999999}, {0.06, 0.46}, {-0.54, 0.6599999999999999}, {-0.6, 0.64}, {-0.4, 0.04}, {-0.6, -0.5599999999999999}, {0, -0.36000000000000004}, {0.6, -0.5599999999999999}, {0.6, 0.64}, {0.54, 0.6599999999999999}, {-0.06, 0.46}, {-0.6599999999999999, 0.6599999999999999}, {-0.46, 0.06}, {-0.6599999999999999, -0.54}, {-0.64, -0.6}, {-0.04, -0.4}, {0.5599999999999999, -0.6}, {0.5599999999999999, 0.6}, {-0.04, 0.4}, {-0.64, 0.6}, {-0.6599999999999999, 0.54}, {-0.46, -0.06}},
	/////
	{{3, 4}, {0, 4}, {0, 0}, {4, 0}, {4.5, 1.75}, {1, 1}, {1, 2}, {2, 3}, {3, 3}, {4, 2}, {5, 3}},
	{{0.3, 0.4}, {0, 0.4}, {0, 0}, {0.4, 0}, {0.45, 0.175}, {0.1, 0.1}, {0.1, 0.2}, {0.2, 0.3}, {0.3, 0.3}, {0.4, 0.2}, {0.5, 0.3}},
	{{0, 0}, {0.4, 0}, {4.4, 0}, {4.9, 1.75}, {4.95, 1.925}, {4.6, 1.85}, {4.6, 1.95}, {1.1, 1.2}, {1.2, 1.3}, {1.3, 1.3}, {1.4, 1.2}, {1.5, 1.3}, {1.5, 2.3}, {1.3, 2.4}, {1, 2.4}, {1, 2}, {1.4, 2}, {2.4, 3}, {2.45, 3.175}, {2.1, 3.1}, {2.1, 3.2}, {3.1, 3.2}, {3.2, 3.3}, {4.2, 2.3}, {4.3, 2.3}, {4.4, 2.2}, {4.5, 2.3}, {5.5, 3.3}, {5.3, 3.4}, {3.3, 4.4}, {3, 4.4}, {0, 4.4}, {0, 4}, {0, 0}, {1, 1.4}, {1, 1}, {1.4, 1}, {1.45, 1.175}, {1.45, 2.175}, {1.1, 2.1}, {2.1, 3.1}, {3.1, 3.1}, {3.1, 3.2}, {4.1, 2.2}, {4.2, 2.3}, {5.2, 3.3}, {3.2, 4.3}, {0.2, 4.3}, {0.2, 0.3}, {0.3, 0.3}, {0.4, 0.2}, {4.4, 0.2}, {4.5, 0.3}, {5, 2.05}, {4.8, 2.15}, {4.5, 2.15}, {1, 1.4}, {1.1, 1.1}, {1.1, 1.2}, {1.1, 2.2}, {1.2, 2.3}, {1.3, 2.3}, {1.4, 2.2}, {1.5, 2.3}, {2.5, 3.3}, {2.3, 3.4}, {2, 3.4}, {2, 3}, {2.4, 3}, {3.4, 3}, {3.45, 3.175}, {3.1, 3.1}, {4.1, 2.1}, {5.1, 3.1}, {5.1, 3.2}, {3.1, 4.2}, {0.1, 4.2}, {0.1, 0.2}, {4.1, 0.2}, {4.2, 0.3}, {4.7, 2.05}, {1.2, 1.3}, {1.2, 2.3}, {2.2, 3.3}, {2.3, 3.3}, {2.4, 3.2}, {3.4, 3.2}, {3.5, 3.3}, {3.3, 3.4}, {3, 3.4}, {3, 3}, {4, 2}, {4.4, 2}, {5.4, 3}, {5.45, 3.175}, {3.45, 4.175}, {0.45, 4.175}, {0.1, 4.1}, {0.1, 0.1}, {4.1, 0.1}, {4.6, 1.85}},
})

func TestElementaryCircuits(t *testing.T) {
	lines := []string{"stroke: black; stroke-width: 12; fill: none", "stroke: red; fill: red; stroke-width: 10; fill: none", "stroke: green; fill: none; stroke-width: 8", "stroke: blue; fill: none; stroke-width: 6", "stroke: yellow; fill: none; stroke-width: 4", "stroke: purple; fill: none; stroke-width: 2", "stroke: orange; fill: none; stroke-width: 1"}

	//for i := 0; i < len(exampleLoops); i += 3 {
	//for i := 2 * 3; i == 2*3; i++ {
	for i := 4 * 3; i == 4*3; i++ {
		fmt.Printf("\nTest %v\n\n", i/3)

		p := Polygon2{verts: exampleLoops[i]}
		q := Polygon2{verts: exampleLoops[i+1]}
		hedges, vertices := HalfEdgesFromSegmentLoop(GetConvolutionCycle(&p, &q))
		fmt.Printf("Half edges: [%v] %v\n", len(hedges), hedges)
		fmt.Printf("First vert (least) %v,%v  index=%v\n", vertices[0].P.ApproxX(), vertices[0].P.ApproxY(), vertices[0].Index)
		components := tarjan(vertices[5:], []int{})
		fmt.Printf("Components: %v\n", components)
		for _, c := range components {
			fmt.Printf("    Component:\n")
			for _, v := range c {
				fmt.Printf("        %v, %v  (%v)\n", v.P.ApproxX(), v.P.ApproxY(), v.Index)
			}
		}
		circuits := ElementaryCircuits(vertices)
		//outline, _ := traceOutline(vertices)
		//circuits := [][]*DCELVertex{outline}
		fmt.Printf("Number of circuits: %v\n", len(circuits))
		/*fmt.Printf("Circuits:\n")
		for i, c := range circuits {
			fmt.Printf("  Circuit %v\n", i)
			for _, v := range c {
				fmt.Printf("    (%v,%v)\n", v.P.ApproxX(), v.P.ApproxY())
			}
		}*/

		html := debugHalfEdgeGraphToHtmlAnimation(&vertices[0], 800, 800)
		canvasF, _ := os.Create(fmt.Sprintf("testoutputs/TestElementaryCircuits_animate_half_edges_%v.html", i/3))
		canvasF.WriteString(html)
		canvasF.Close()

		svgout, _ := os.Create(fmt.Sprintf("testoutputs/TestElementaryCircuits_components_figure_%v.svg", i/3))
		strips := make([][]Vec2, 0)
		for _, c := range components {
			strip := make([]Vec2, 0)
			for _, v := range c {
				strip = append(strip, *(v.P.Copy()))
			}
			strips = append(strips, strip)
		}
		canvas := svg.New(svgout)
		debugDrawLineStrips(canvas, strips, lines)
		svgout.Close()

		svgout, _ = os.Create(fmt.Sprintf("testoutputs/TestElementaryCircuits_circuits_figure_%v.svg", i/3))
		strips = make([][]Vec2, 0)
		for _, c := range circuits {
			strip := make([]Vec2, 0)
			for _, v := range c {
				strip = append(strip, *(v.P.Copy()))
			}
			strips = append(strips, strip)
		}
		canvas = svg.New(svgout)
		debugDrawLineStrips(canvas, strips, lines)
		svgout.Close()
	}
}

func TestConvolve(t *testing.T) {
	for i := 0; i < len(exampleLoops); i += 3 {
		p := Polygon2{verts: exampleLoops[i]}
		q := Polygon2{verts: exampleLoops[i+1]}
		expected := exampleLoops[i+2]

		cs := GetConvolutionCycle(&p, &q)
		fmt.Printf("The expected (left) vs. computed conv cycle for test %v:\n", i/3)
		l := len(cs)
		if len(expected) > l {
			l = len(expected)
		}

		fmt.Printf("The conv cycle: {")
		for i, pt := range cs {
			if i != 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("{%v,%v}", pt.ApproxX(), pt.ApproxY())
		}
		fmt.Printf("}\n\n")

		itns, checks := SegmentLoopIntersections(cs)
		itnsNaive, checksNaive := SegmentLoopIntersectionsUsingNaiveAlgo(cs)
		_ = itnsNaive
		//itns = itnsNaive

		if checks >= checksNaive {
			t.Errorf("Bad performance in 'SegmentLoopIntersections' -- used as many intersection checks or more than 'SegmentLoopIntersectionsUsingNaiveAlgo' (%v vs. %v)", checks, checksNaive)
		} else {
			fmt.Printf("Bentley Ottman: %v checks; naive algo: %v checks\n", checks, checksNaive)
		}

		if len(itns) != len(itnsNaive) {
			t.Errorf("Naive segment loop intersection algo found more intersections than Bentley Ottman implementation for test %v: %v vs. %v\n", i/3, len(itnsNaive), len(itns))
		}

		fmt.Printf("Expected vs actual values:\n")
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

			if i >= len(cs) {
				t.Errorf("Too many points computed on convolution cycle for test %v: %v expected, got %v\n", i/3, len(expected), len(cs))
			}
			if i >= len(expected) {
				t.Errorf("Too few points computed on convolution cycle for test %v\n: %v expected, got %v\n", i/3, len(expected), len(cs))
			}
			if i < len(cs) && i < len(expected) && !expected[i].SlowEqEpsilon(&cs[i], EPSILON) {
				t.Errorf("Mismatch in conv cycle points: (%v,%v) [expected] vs (%v,%v) [actual]\n", f(&expected[i].x), f(&expected[i].y), f(&cs[i].x), f(&cs[i].y))
			}
		}
		fmt.Printf("\n")

		crosses := make([][]Vec2, 0)
		for _, p := range itnsNaive {
			crosses = append(crosses, SofVec2([][]float64{{p.ApproxX(), p.ApproxY()}, {p.ApproxX(), p.ApproxY()}}))
		}

		svgout, _ := os.Create(fmt.Sprintf("testoutputs/TestConvolve_figure_%v.svg", i/3))
		canvas := svg.New(svgout)
		strips := [][]Vec2{p.verts, cs}
		strips = append(strips, crosses...)
		debugDrawLineStrips(canvas, strips, []string{"stroke: black; stroke-width: 4; fill: none", "stroke: red; fill: red; stroke-width: 1; fill: none", "stroke: green; fill: green; stroke-width: 1"})
	}
}
