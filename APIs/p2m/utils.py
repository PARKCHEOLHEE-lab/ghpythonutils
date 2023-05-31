import numpy as np
import cv2

from shapely.geometry import Polygon


class P2M:

    THRESHOLD_BINARY = 50
    MAXVAL = 255

    THRESHOLD_CANNY_1 = 100
    THRESHOLD_CANNY_2 = 200
    MIN_AREA = 1

    KERNEL_SIZE = (5, 5)
    
    @staticmethod
    def get_cleaned_image(path: str) -> np.ndarray:
        """_summary_

        Args:
            path (str): image file path (.jpg or .png)

        Returns:
            np.ndarray: image after clean up 
        """
        
        image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # remove other elements except walls
        _, image_binary = cv2.threshold(image_gray, P2M.THRESHOLD_BINARY, P2M.MAXVAL, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_binary, connectivity=4)

        image_unconnected_removed = np.zeros_like(image_gray)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > P2M.MIN_AREA:
                image_unconnected_removed[labels == label] = 255

        # Image Erosion and Dilation
        kernel = np.ones(P2M.KERNEL_SIZE, np.uint8)
        cleaned_image = cv2.morphologyEx(image_unconnected_removed, cv2.MORPH_CLOSE, kernel)

        # erosion_and_dilation = cv2.morphologyEx(dilation_and_erosion, cv2.MORPH_OPEN, kernel)
        # cv2.imwrite("test.png", cleaned_image)
        
        return cleaned_image
    
    @staticmethod    
    def get_wall_contour(image):
        
        edges = cv2.Canny(image, P2M.THRESHOLD_CANNY_1, P2M.THRESHOLD_CANNY_2)
        
        cv2.imwrite("edges.png", edges)
        
        return


if __name__ == "__main__":
    path = "C:/Users/park1/AppData/Roaming/McNeel/Rhinoceros/7.0/Plug-ins/IronPython (814d908a-e25c-493d-97e9-ee3861957f49)/settings/lib/ghpythonutils/APIs/p2m/assets/plan_0.png"
    # path = "C:/Users/park1/AppData/Roaming/McNeel/Rhinoceros/7.0/Plug-ins/IronPython (814d908a-e25c-493d-97e9-ee3861957f49)/settings/lib/ghpythonutils/APIs/p2m/assets/plan_1.jpg"
    # path = "C:/Users/park1/AppData/Roaming/McNeel/Rhinoceros/7.0/Plug-ins/IronPython (814d908a-e25c-493d-97e9-ee3861957f49)/settings/lib/ghpythonutils/APIs/p2m/assets/plan_2.jpg"

    cleaned_image = P2M.get_cleaned_image(path)
    P2M.get_wall_contour(cleaned_image)