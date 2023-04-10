import Rhino.Geometry as rg

from ghpythonutils.utils.utils import ConstsCollection, LineHelper


class Quadtree:
    COUNTER = 1

    def __init__(self, min_size, bounding_box):
        self.bounding_box = bounding_box
        self.min_size = min_size

        self.crvs = []
        self.children = None
        self.children_geometries = None

    def _subdivide(self):
        pass

    def _is_intersects(self, crv):
        p1 = crv.PointAtStart
        p2 = crv.PointAtEnd

        for p in (p1, p2):
            condition = self.bounding_box.Contains(p)
            is_intersects = condition in (
                rg.PointContainment.Inside,
                rg.PointContainment.Coincident,
            )

            if is_intersects:
                return True

        return False

    def insert(self, crv):
        if self.children:
            for child in self.children:
                if self._is_intersects(crv):
                    child.insert(crv)
        else:
            self.crvs.append(crv)


#            if (
#                len(self.crvs) > self.COUNTER
#                and self.width > self.MIN_SIZE
#                and self.height > self.MIN_SIZE
#            ):
#                self.subdivide()

if __name__ == "__main__":
    plane = rg.Plane.WorldXY

    l = rg.PolylineCurve([rg.Point3d(0, 0, 0), rg.Point3d(1, 1, 1)])
    bounding_box = LineHelper.get_2d_obb_from_line(
        #        rg.PolylineCurve([plane.Origin, plane.Origin + plane.XAxis]),
        l,
        input_curve,
    ).ToNurbsCurve()

    quadtree = Quadtree(10, bounding_box)
    a = bounding_box

    exploded_curve = input_curve.DuplicateSegments()
    for c in exploded_curve:
        quadtree.insert(c)
