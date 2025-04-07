"""
Character segmentation module for license plate images
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import LICENSE_WIDTH, LICENSE_HEIGHT

class CharSegmentation:
    """Character segmentation for license plate images"""
    
    def __init__(self):
        """Initialize character segmentation"""
        self.char_images = []
        self.projection_threshold = 10  # Minimum number of black pixels to consider a column/row valid
    
    def segment(self, plate_image, visualize=False):
        """
        Segment characters from license plate image
        
        Args:
            plate_image: Preprocessed license plate image (binary or grayscale)
            visualize: Whether to visualize the segmentation process
            
        Returns:
            list: List of segmented character images
        """
        # Reset character images list
        self.char_images = []
        
        # Make a copy of the input image
        img = plate_image.copy()
        
        # Ensure image is grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Remove frame and rivets using jump count thresholding
        img_cleaned = self._remove_frame_and_rivets(img)
        
        # Apply vertical projection for character segmentation
        char_regions = self._find_char_regions(img_cleaned)
        
        # Extract individual characters
        for i, (start, end) in enumerate(char_regions):
            # Skip if region is too narrow (likely noise)
            if end - start < 5:
                continue
                
            # Extract character region
            char_img = img_cleaned[:, start:end]
            
            # Apply horizontal projection to trim top and bottom (optional)
            row_sum = np.sum(char_img == 0, axis=1)
            valid_rows = np.where(row_sum > 0)[0]
            
            if len(valid_rows) > 0:
                top = valid_rows[0]
                bottom = valid_rows[-1]
                
                # Ensure minimum height
                if bottom - top < 10:
                    padding = (10 - (bottom - top)) // 2
                    top = max(0, top - padding)
                    bottom = min(char_img.shape[0] - 1, bottom + padding)
                
                char_img = char_img[top:bottom+1, :]
            
            # Resize character to standard size (20x20)
            char_img_resized = cv2.resize(char_img, (20, 20))
            
            # Add to character images list
            self.char_images.append(char_img_resized)
            
            if visualize:
                plt.figure(figsize=(1, 1))
                plt.imshow(char_img_resized, cmap='gray')
                plt.title(f"Char {i+1}")
                plt.axis('off')
                plt.show()
        
        if visualize:
            # Visualize all characters in a row
            if self.char_images:
                fig, axes = plt.subplots(1, len(self.char_images), figsize=(12, 2))
                for i, char_img in enumerate(self.char_images):
                    if len(self.char_images) > 1:
                        axes[i].imshow(char_img, cmap='gray')
                        axes[i].set_title(f"Char {i+1}")
                        axes[i].axis('off')
                    else:
                        axes.imshow(char_img, cmap='gray')
                        axes.set_title(f"Char {i+1}")
                        axes.axis('off')
                plt.tight_layout()
                plt.show()
        
        return self.char_images
    
    def _remove_frame_and_rivets(self, img):
        """
        Remove license plate frame and rivets using jump count analysis
        
        Args:
            img: Grayscale license plate image
            
        Returns:
            numpy.ndarray: Cleaned image
        """
        # Make a copy of the input image
        img_cleaned = img.copy()
        
        # Calculate row-wise and column-wise jump counts
        row_jumps = self._calculate_jump_counts(img, axis=1)  # Horizontal jumps (by row)
        col_jumps = self._calculate_jump_counts(img, axis=0)  # Vertical jumps (by column)
        
        # Find the valid rows (excluding frame and rivets)
        valid_rows = []
        for i in range(len(row_jumps)):
            if row_jumps[i] >= self.projection_threshold:
                valid_rows.append(i)
        
        if valid_rows:
            row_start = min(valid_rows)
            row_end = max(valid_rows)
        else:
            row_start = 0
            row_end = img.shape[0] - 1
            
        # Find the valid columns (excluding frame)
        valid_cols = []
        for i in range(len(col_jumps)):
            if col_jumps[i] >= self.projection_threshold:
                valid_cols.append(i)
        
        if valid_cols:
            col_start = min(valid_cols)
            col_end = max(valid_cols)
        else:
            col_start = 0
            col_end = img.shape[1] - 1
        
        # Set pixels outside the valid region to white
        for i in range(img_cleaned.shape[0]):
            if i < row_start or i > row_end:
                img_cleaned[i, :] = 255
                
        for j in range(img_cleaned.shape[1]):
            if j < col_start or j > col_end:
                img_cleaned[:, j] = 255
        
        return img_cleaned
    
    def _calculate_jump_counts(self, img, axis=1):
        """
        Calculate the number of black-white transitions along an axis
        
        Args:
            img: Grayscale image
            axis: 0 for columns, 1 for rows
            
        Returns:
            list: Jump counts for each row or column
        """
        jump_counts = []
        
        if axis == 1:  # Horizontal (by row)
            for row in range(img.shape[0]):
                jumps = 0
                for col in range(1, img.shape[1]):
                    # Count transitions between black and white pixels
                    if (img[row, col] < 128) != (img[row, col-1] < 128):
                        jumps += 1
                jump_counts.append(jumps)
        else:  # Vertical (by column)
            for col in range(img.shape[1]):
                jumps = 0
                for row in range(1, img.shape[0]):
                    # Count transitions between black and white pixels
                    if (img[row, col] < 128) != (img[row-1, col] < 128):
                        jumps += 1
                jump_counts.append(jumps)
        
        return jump_counts
    
    def _find_char_regions(self, img):
        """
        Find character regions using vertical projection
        
        Args:
            img: Cleaned grayscale image
            
        Returns:
            list: List of (start_col, end_col) tuples for each character
        """
        # Calculate vertical projection (black pixel count by column)
        projection = np.sum(img < 128, axis=0)
        
        # Find character regions
        char_regions = []
        in_char = False
        start_col = 0
        
        for col in range(len(projection)):
            if not in_char and projection[col] > self.projection_threshold:
                # Start of a character
                in_char = True
                start_col = col
            elif in_char and projection[col] <= self.projection_threshold:
                # End of a character
                in_char = False
                # Ignore very narrow regions (noise)
                if col - start_col >= 5:
                    char_regions.append((start_col, col))
        
        # Handle case where last character extends to the end of the image
        if in_char:
            char_regions.append((start_col, len(projection)-1))
        
        # Special handling for Chinese plate format (if needed)
        # Chinese plates typically have format: 京A12345 (Chinese character followed by letter and digits)
        if len(char_regions) >= 7:
            # First character is likely province character (Chinese)
            # Add special handling if needed
            pass
        
        return char_regions
    
    def visualize_projection(self, img):
        """
        Visualize vertical projection for character segmentation
        
        Args:
            img: Grayscale license plate image
        """
        # Calculate vertical projection (black pixel count by column)
        projection = np.sum(img < 128, axis=0)
        
        plt.figure(figsize=(12, 6))
        
        # Plot the original image
        plt.subplot(2, 1, 1)
        plt.imshow(img, cmap='gray')
        plt.title("License Plate")
        plt.axis('off')
        
        # Plot the projection
        plt.subplot(2, 1, 2)
        plt.bar(range(len(projection)), projection, width=1)
        plt.title("Vertical Projection")
        plt.xlabel("Column")
        plt.ylabel("Black Pixel Count")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Draw threshold line
        plt.axhline(y=self.projection_threshold, color='r', linestyle='-', alpha=0.7)
        plt.text(10, self.projection_threshold + 1, 'Threshold', color='r')
        
        # Highlight character regions
        char_regions = self._find_char_regions(img)
        for start, end in char_regions:
            plt.axvspan(start, end, color='green', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# For direct execution
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='License plate character segmentation')
    parser.add_argument('-i', '--image', required=True, help='Path to license plate image')
    args = parser.parse_args()
    
    # Read image
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {args.image}")
        import sys
        sys.exit(1)
    
    # Create segmentation object
    segmenter = CharSegmentation()
    
    # Visualize projection for debugging
    segmenter.visualize_projection(img)
    
    # Segment characters
    char_images = segmenter.segment(img, visualize=True)
    
    print(f"Segmented {len(char_images)} characters")