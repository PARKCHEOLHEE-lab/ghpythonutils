import urllib2
import json
import os

import Rhino.Geometry as rg


class WallGenerator:
    def __init__(self, base_url, path, height):
        self.base_url = base_url
        self.path = path.replace("\\", "/")
        self.url = base_url + path
        
        self.height = height
        
    def generate(self):
        self.wall_coordinates = self._get_wall_coordinates()
        self.wall_polylines = self._get_wall_polylines()
        self.walls = self._get_walls()
    
    def _get_wall_coordinates(self):
        request = urllib2.Request(self.url)
        response = urllib2.urlopen(request)
        
        return json.loads(response.read())["wall_coordinates"]
        
    def _get_wall_polylines(self):
        wall_polylines = []
        for coords in self.wall_coordinates:
            vertices = [rg.Point3d(coord[0], coord[1], 0) for coord in coords]
            wall_polyline = rg.PolylineCurve(vertices)
            wall_polylines.append(wall_polyline)
        
        return wall_polylines
        
    def _get_walls(self):
        walls = []
        
        extrusion_vector = rg.Vector3d(0, 0, self.height)
        for wall_polyline in self.wall_polylines:
            wall = rg.Extrusion.CreateExtrusion(wall_polyline, extrusion_vector)
            walls.append(wall)
            
        return walls



if __name__ == "__main__":
    
    wall_generator = WallGenerator(
        base_url=base_url, path=image_path, height=200
    )
    wall_generator.generate()
    
    wall_polylines = wall_generator.wall_polylines
    walls = wall_generator.walls