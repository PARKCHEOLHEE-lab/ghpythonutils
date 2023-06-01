
import cv2
import numpy as np

from typing import List
from shapely.geometry import Polygon


class P2M:

    # for open-cv
    THRESHOLD_BINARY = 50
    MAXVAL = 255

    THRESHOLD_CANNY_1 = 100
    THRESHOLD_CANNY_2 = 200
    MIN_AREA = 1

    KERNEL_SIZE = (5, 5)

    # for shapely
    TOLERANCE_SIMPLIFY = 0.01
    MIN_AREA_GEOM = 1000
    
    def __init__(self, path: str) -> None:
        self.path = path
        self.wall_image_cleaned = self.get_wall_image_cleaned()
        self.wall_coordinates = self.get_wall_coordinates()
    
    def get_wall_image_cleaned(self) -> np.ndarray:
        """Removes all elements in the image except the walls"""
        
        image_gray = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        
        # remove other elements except walls
        _, image_binary = cv2.threshold(image_gray, self.THRESHOLD_BINARY, self.MAXVAL, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_binary, connectivity=4)

        image_unconnected_removed = np.zeros_like(image_gray)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > self.MIN_AREA:
                image_unconnected_removed[labels == label] = 255

        # Image Erosion and Dilation
        kernel = np.ones(self.KERNEL_SIZE, np.uint8)
        cleaned_image = cv2.morphologyEx(image_unconnected_removed, cv2.MORPH_CLOSE, kernel)
        
        return cleaned_image
    
    def get_wall_coordinates(self) -> List[Polygon]:
        """Converts list of coordinates to list of polygon"""

        contours, _ = cv2.findContours(self.wall_image_cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        wall_coordinates = []
        for ci, contour in enumerate(contours):

            # Skip the image's boundary coordinates. 
            # Very first iteration is boundary of it
            if ci == 0:
                continue

            coords = [list(coord) for coord, *_ in contour]
            wall_geometry = Polygon(coords).simplify(self.TOLERANCE_SIMPLIFY)

            if wall_geometry.area >= self.MIN_AREA_GEOM: 
                wall_coord = list(wall_geometry.boundary.coords)
                wall_coordinates.append(wall_coord)
        
        return wall_coordinates