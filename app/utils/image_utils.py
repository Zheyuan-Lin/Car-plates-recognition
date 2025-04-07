"""
Image processing utilities for license plate recognition
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(name, img, wait=True):
    """
    Display an image in a window
    
    Args:
        name: Window name
        img: Image to display
        wait: Whether to wait for key press
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyWindow(name)


def show_images(images, titles=None, figsize=(15, 10)):
    """
    Display multiple images in subplots
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        figsize: Figure size (width, height)
    """
    n = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n + 1)]
    
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i in range(n):
        if i < len(images):
            if len(images[i].shape) == 3 and images[i].shape[2] == 3:
                # Convert BGR to RGB for proper display in matplotlib
                img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
            else:
                axes[i].imshow(images[i], cmap='gray')
            axes[i].set_title(titles[i])
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def apply_canny(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection
    
    Args:
        image: Input image
        low_threshold: Lower threshold for hysteresis procedure
        high_threshold: Higher threshold for hysteresis procedure
        
    Returns:
        numpy.ndarray: Edge image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


def apply_threshold(image, method='adaptive', block_size=11, c=2, threshold=127, max_val=255):
    """
    Apply thresholding to image
    
    Args:
        image: Input grayscale image
        method: Thresholding method ('adaptive', 'otsu', 'binary')
        block_size: Block size for adaptive thresholding
        c: Constant subtracted from the mean
        threshold: Global threshold value for binary thresholding
        max_val: Maximum value to assign to pixels above threshold
        
    Returns:
        numpy.ndarray: Thresholded image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'adaptive':
        return cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, block_size, c)
    elif method == 'otsu':
        _, thresh = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    else:  # binary
        _, thresh = cv2.threshold(gray, threshold, max_val, cv2.THRESH_BINARY_INV)
        return thresh


def enhance_plate(image):
    """
    Enhance license plate image for better character segmentation
    
    Args:
        image: License plate image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding
    thresh = apply_threshold(bilateral, method='adaptive')
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return morph


def crop_image(image, bbox):
    """
    Crop image based on bounding box
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, width, height)
        
    Returns:
        numpy.ndarray: Cropped image
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def normalize_image(image):
    """
    Normalize image to 0-1 range
    
    Args:
        image: Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    return image.astype(np.float32) / 255.0


def draw_bounding_boxes(image, boxes, labels=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image
    
    Args:
        image: Input image
        boxes: List of bounding boxes (x, y, width, height)
        labels: List of labels for each box
        color: Box color (B, G, R)
        thickness: Line thickness
        
    Returns:
        numpy.ndarray: Image with bounding boxes
    """
    # Make a copy of the input image
    result = image.copy()
    
    for i, box in enumerate(boxes):
        x, y, w, h = box
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label if provided
        if labels is not None and i < len(labels):
            label_text = str(labels[i])
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
            
            # Draw text background
            cv2.rectangle(result, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(result, label_text, (x, y - 5), font, font_scale, (0, 0, 0), 1)
    
    return result