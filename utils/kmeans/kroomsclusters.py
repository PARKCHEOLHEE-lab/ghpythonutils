from utils.kmeans.kmeans import KMeans

class KRoomsCluster(KMeans):
    """
    To use the kmeans module, refer the link below.
    https://github.com/PARKCHEOLHEE-lab/GhPythonUtils
    """
    
    def __init__(self, floor, corridor, target_area, axis=None):
        self.floor = floor
        self.corridor = corridor
        self.target_area = target_area
        self.axis = axis
        
        KMeans.__init__(self)
        
    def _get_longest_segment(self):
        if self.axis is None:
            self.axis = 1  # set axis to longest segment of floor segments
        return
        
    def _get_longest_segment_axis_grid(self):
        return
        
    def _get_estimated_k(self):
        """The K is given floor area divided target area."""
        k = 0
        return k
        
    def _get_shortest_path(self):
        return

if __name__ == "__main__":
    KRoomsCluster(
        floor=None,
        corridor=None,
        target_area=None,
    )
