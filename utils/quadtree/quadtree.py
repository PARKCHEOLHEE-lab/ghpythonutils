import Rhino.Geometry as rg

from ghpythonutils.utils.utils import ConstsCollection


class Quadtree:
    COUNTER = 1

    def __init__(self, bounding_box, min_size=2):
        self.bounding_box = bounding_box
        self.min_size = min_size

        self.crvs = []
        self.children = None
        self.children_geometries = None

        self._post_init()

    def _post_init(self):
        exploded_bounding_box = self.bounding_box.DuplicateSegments()

        self.width, self.height, _, _ = [
            seg.GetLength() for seg in exploded_bounding_box
        ]

        _, self.plane = exploded_bounding_box[0].FrameAt(ConstsCollection.HALF)

        centroid = rg.AreaMassProperties.Compute(self.bounding_box).Centroid

        self.plane.Origin = centroid

    def _subdivide(self):
        ne_bb = rg.Rectangle3d(self.plane, self.width / 2, self.height / 2)
        ne_bb = ne_bb.ToNurbsCurve()

        nw_bb = rg.Rectangle3d(self.plane, -self.width / 2, self.height / 2)
        nw_bb = nw_bb.ToNurbsCurve()

        se_bb = rg.Rectangle3d(self.plane, self.width / 2, -self.height / 2)
        se_bb = se_bb.ToNurbsCurve()

        sw_bb = rg.Rectangle3d(self.plane, -self.width / 2, -self.height / 2)
        sw_bb = sw_bb.ToNurbsCurve()

        ne = Quadtree(ne_bb)
        nw = Quadtree(nw_bb)
        se = Quadtree(se_bb)
        sw = Quadtree(sw_bb)

        self.children = [ne, nw, se, sw]
        self.children_geometries = [ne_bb, nw_bb, se_bb, sw_bb]

        for crv in self.crvs:
            if self._is_intersects(crv, ne_bb):
                ne.insert(crv)
            if self._is_intersects(crv, nw_bb):
                nw.insert(crv)
            if self._is_intersects(crv, se_bb):
                se.insert(crv)
            if self._is_intersects(crv, sw_bb):
                sw.insert(crv)

    def _is_intersects(self, crv, bb):
        intersects = rg.Intersect.Intersection.CurveCurve(
            crv, bb, ConstsCollection.TOLERANCE, ConstsCollection.TOLERANCE
        )

        return len(intersects) > 0

    def insert(self, crv):
        """Append the geometry to quadtree

        Args:
            crv (Rhino.Geometry.Curve): Target curve to insert
        """

        if self.children:
            for child in self.children:
                if self._is_intersects(crv, self.bounding_box):
                    child.insert(crv)
        else:
            self.crvs.append(crv)

            if len(self.crvs) >= self.COUNTER and (
                self.width > self.min_size or self.height > self.min_size
            ):
                self._subdivide()


def get_all_quadtree_geometries(quadtree):
    """Get all geometries recursively in the quadtree

    Args:
        quadtree (Quadtree): Root quadtree

    Returns:
        List[Rhino.Geometry.Curve]: Flattened geometry list
    """

    all_quadtree_geometries = []

    if quadtree.children:
        for child in quadtree.children:
            geoms = get_all_quadtree_geometries(child)

            if geoms is not None:
                all_quadtree_geometries.extend(geoms)

    elif quadtree.children is None:
        all_quadtree_geometries.append(quadtree.bounding_box)

    else:
        for child_geometry in quadtree.children_geometries:
            all_quadtree_geometries.append(child_geometry)

    return all_quadtree_geometries
