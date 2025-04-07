"""
License plate localization module for detecting and extracting license plates from images
"""
import cv2
import numpy as np
from config import (BLUE, GREEN, RED, COLOR_THRESHOLD, MIN_AREA, 
                   MAX_WIDTH, ANGLE_THRESHOLD, LICENSE_WIDTH, LICENSE_HEIGHT)

class PlateLocator:
    """License plate localization class"""
    
    def __init__(self):
        """Initialize plate locator"""
        pass
    
    def locate(self, image):
        """
        Locate and extract license plate from image
        
        Args:
            image: Input image
            
        Returns:
            tuple: (processed_plate_image, success)
        """
        # Make a copy of the original image
        img_original = image.copy()
        img_height, img_width = image.shape[:2]
        
        # Resize image if needed
        img = self._resize_image(image)
        
        # Preprocess image
        img_processed = self._preprocess_image(img)
        
        # Locate plate candidates
        img_binary, contours = self._find_plate_contours(img_processed)
        
        # Find plate among candidates
        plate_rect = self._select_plate_contour(contours, img)
        if plate_rect is None:
            print("No license plate detected")
            return None, False
        
        # Draw contour on original image for visualization
        box = cv2.boxPoints(plate_rect)
        box = np.int0(box)
        img_with_box = cv2.drawContours(img.copy(), [box], 0, (0, 0, 255), 2)
        
        # Extract and rectify plate
        plate_img = self._extract_plate(img, plate_rect)
        if plate_img is None:
            return None, False
            
        # Further processing for better character segmentation
        processed_plate = self._process_plate(plate_img)
        
        return processed_plate, True
    
    def _resize_image(self, image):
        """Resize image if width exceeds maximum"""
        img_height, img_width = image.shape[:2]
        if img_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / img_width
            return cv2.resize(image, (MAX_WIDTH, int(img_height * resize_rate)), 
                              interpolation=cv2.INTER_AREA)
        return image.copy()
    
    def _preprocess_image(self, image):
        """Apply image preprocessing filters"""
        # Apply Gaussian blur
        img_gaussian = cv2.GaussianBlur(image, (5, 5), 1)
        
        # Apply median filter
        img_median = cv2.medianBlur(img_gaussian, 3)
        
        return img_median
    
    def _find_plate_contours(self, image):
        """Find contours that might be license plates based on color"""
        img_processed = image.copy()
        
        # Split channels
        img_b = cv2.split(img_processed)[0]
        img_g = cv2.split(img_processed)[1]
        img_r = cv2.split(img_processed)[2]
        
        # Filter colors close to license plate blue
        for i in range(img_processed.shape[0]):
            for j in range(img_processed.shape[1]):
                if (abs(int(img_b[i, j]) - BLUE) < COLOR_THRESHOLD and 
                    abs(int(img_g[i, j]) - GREEN) < COLOR_THRESHOLD and 
                    abs(int(img_r[i, j]) - RED) < COLOR_THRESHOLD):
                    img_processed[i, j] = [255, 255, 255]
                else:
                    img_processed[i, j] = [0, 0, 0]
        
        # Morphological operations to enhance plate regions
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_processed, kernel, iterations=5)
        img_erode = cv2.erode(img_dilate, kernel, iterations=5)
        
        # Convert to grayscale for contour detection
        img_gray = cv2.cvtColor(img_erode, cv2.COLOR_RGB2GRAY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, 
                                                 cv2.CHAIN_APPROX_SIMPLE)[1:]
        
        return img_gray, contours
    
    def _select_plate_contour(self, contours, image):
        """
        Select the contour most likely to be a license plate
        
        Args:
            contours: List of contours
            image: Original image
            
        Returns:
            rect: Minimum area rectangle (center, size, angle)
        """
        for cnt in contours:
            # Generate minimum area rectangle
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            
            # Calculate area and filter by minimum size
            area = area_width * area_height
            if area < MIN_AREA:
                continue
                
            # Ensure width is longer than height
            if area_width < area_height:
                area_width, area_height = area_height, area_width
                
            # Calculate width-to-height ratio
            wh_ratio = area_width / area_height
            
            # Filter by aspect ratio (license plates typically have ratio between 2 and 5.5)
            if 2 < wh_ratio < 5.5:
                return rect
                
        return None
    
    def _extract_plate(self, image, rect):
        """
        Extract and rectify license plate from image
        
        Args:
            image: Original image
            rect: Minimum area rectangle (center, size, angle)
            
        Returns:
            numpy.ndarray: Rectified license plate image
        """
        # Get rotated rectangle coordinates
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get rectangle width and height
        width, height = rect[1]
        
        # Ensure width is longer than height
        if width < height:
            width, height = height, width
        
        # Prepare source and destination points for perspective transform
        src_pts = box.astype("float32")
        
        # Sort points to ensure consistent ordering: top-left, top-right, bottom-right, bottom-left
        s = src_pts.sum(axis=1)
        src_pts = np.array([
            src_pts[np.argmin(s)],  # Top-left
            src_pts[np.argmin(src_pts[:, 0] + src_pts[:, 1] - s)],  # Top-right
            src_pts[np.argmax(s)],  # Bottom-right
            src_pts[np.argmax(src_pts[:, 0] - src_pts[:, 1] - s)]   # Bottom-left
        ])
        
        # Set destination points for standard license plate size
        dst_pts = np.array([
            [0, 0],
            [LICENSE_WIDTH - 1, 0],
            [LICENSE_WIDTH - 1, LICENSE_HEIGHT - 1],
            [0, LICENSE_HEIGHT - 1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, M, (LICENSE_WIDTH, LICENSE_HEIGHT))
        
        return warped
    
    def _process_plate(self, plate_img):
        """
        Process extracted plate for better character segmentation
        
        Args:
            plate_img: Extracted license plate image
            
        Returns:
            numpy.ndarray: Processed plate image ready for character segmentation
        """
        # Apply Gaussian blur for noise reduction
        img_gaussian = cv2.GaussianBlur(plate_img, (5, 5), 1)
        
        # Apply median filter for additional noise reduction
        img_median = cv2.medianBlur(img_gaussian, 3)
        
        # Extract plate background color (typically blue) again for better contrast
        img_b = cv2.split(img_median)[0]
        img_g = cv2.split(img_median)[1]
        img_r = cv2.split(img_median)[2]
        
        # Create a mask for the plate background
        for i in range(img_median.shape[0]):
            for j in range(img_median.shape[1]):
                if (abs(int(img_b[i, j]) - BLUE) < COLOR_THRESHOLD and 
                    abs(int(img_g[i, j]) - GREEN) < COLOR_THRESHOLD and 
                    abs(int(img_r[i, j]) - RED) < COLOR_THRESHOLD):
                    img_median[i, j] = [255, 255, 255]
                else:
                    img_median[i, j] = [0, 0, 0]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)
        
        return gray


# For direct execution
if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='License plate detection')
    parser.add_argument('-i', '--image', required=True, help='Path to input image')
    args = parser.parse_args()
    
    # Read image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image {args.image}")
        sys.exit(1)
    
    # Create plate locator
    locator = PlateLocator()
    
    # Locate plate
    plate_img, success = locator.locate(image)
    
    if success:
        # Show result
        cv2.imshow('Original', image)
        cv2.imshow('Plate', plate_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to locate license plate")