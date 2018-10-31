# ggeom

This is an implementation of some 2D computational geometry algorithms in Go.

The focus is on implementing the algorithms required to compute 2D offset polygons via [Ron Wein's algorithm](https://pdfs.semanticscholar.org/b049/3b89b18d785ca81427404ec78d7ce6602ceb.pdf).

Algorithms implemented so far:

* The intersections of a sequence of connected line segments (via the Bentley-Ottmann algorithm).

* A test for whether or not a point lies inside (or on one of the line segments of) a 2D polygon.

* The intersection of two non-self-intersecting 2D polygons without holes (via the Weiler-Atherton algorithm).

* The elementary circuits of the convolution of one 2D polygon with another. (This is the computation at the heart of Wein's algorithm for computing offset polygons.)

This code is released under a two-clause BSD license. CGAL has a more battle-tested [implementation](https://doc.cgal.org/latest/Straight_skeleton_2/index.html) of poylgon offsetting, but it's GPL licensed.

Scalar values are represented as `big.Rat` values. By redefining the `Scalar` typedef in `ggeom.go`, you can use any other type which follows the same API.
