"""
Data utilities for license plate recognition system
"""
import os
import cv2
import numpy as np
import random
from config import CHARS_CLASS, PROVINCES, CHAR_DIR, CHINESE_CHAR_DIR

def create_directory_structure(base_path='data'):
    """
    Create the necessary directory structure for training data
    
    Args:
        base_path: Base directory path
    """
    # Create base directories
    for dir_type in ['train', 'test']:
        train_path = os.path.join(base_path, dir_type)
        os.makedirs(train_path, exist_ok=True)
        
        # Create character directories
        chars_path = os.path.join(train_path, CHAR_DIR)
        os.makedirs(chars_path, exist_ok=True)
        
        for char in CHARS_CLASS:
            os.makedirs(os.path.join(chars_path, char), exist_ok=True)
        
        # Create Chinese character directories
        chinese_path = os.path.join(train_path, CHINESE_CHAR_DIR)
        os.makedirs(chinese_path, exist_ok=True)
        
        # Create only for pinyin directories (odd indices in PROVINCES)
        for i in range(0, len(PROVINCES), 2):
            if i < len(PROVINCES) and PROVINCES[i].startswith('zh_'):
                os.makedirs(os.path.join(chinese_path, PROVINCES[i]), exist_ok=True)
    
    print(f"Directory structure created at {base_path}")


def split_data(src_dir, train_ratio=0.8, shuffle=True):
    """
    Split data into training and testing sets
    
    Args:
        src_dir: Source directory containing all data
        train_ratio: Ratio of data to use for training
        shuffle: Whether to shuffle data before splitting
        
    Returns:
        tuple: (train_files, test_files)
    """
    all_files = []
    
    # Walk through directory structure
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.bmp')):
                all_files.append(os.path.join(root, file))
    
    # Shuffle data if requested
    if shuffle:
        random.shuffle(all_files)
    
    # Split data
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    return train_files, test_files


def augment_data(image, num_augmentations=5):
    """
    Apply data augmentation to an image
    
    Args:
        image: Input image
        num_augmentations: Number of augmented images to generate
        
    Returns:
        list: List of augmented images
    """
    augmented_images = []
    
    for _ in range(num_augmentations):
        # Apply random transformations
        img = image.copy()
        
        # Random rotation (slight)
        angle = random.uniform(-10, 10)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random brightness/contrast adjustment
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-10, 10)    # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # Random blur
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        augmented_images.append(img)
    
    return augmented_images


def prepare_character_data(image):
    """
    Prepare character image for neural network input
    
    Args:
        image: Character image
        
    Returns:
        numpy.ndarray: Prepared image as flattened vector
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize to standard size
    resized = cv2.resize(gray, (20, 20))
    
    # Normalize and flatten
    normalized = resized.flatten() / 255.0
    
    return normalized


def batch_generator(x, y, batch_size=100, shuffle=True):
    """
    Generate mini-batches for training
    
    Args:
        x: Input data
        y: Target labels
        batch_size: Size of mini-batches
        shuffle: Whether to shuffle data before each epoch
        
    Yields:
        tuple: (x_batch, y_batch)
    """
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield x[batch_indices], y[batch_indices]


def convert_to_one_hot(labels, num_classes):
    """
    Convert integer labels to one-hot encoding
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes
        
    Returns:
        numpy.ndarray: One-hot encoded labels
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def load_test_data(test_dir, char_type='chars'):
    """
    Load test data for evaluation
    
    Args:
        test_dir: Test data directory
        char_type: Type of characters to load ('chars' or 'charsChinese')
        
    Returns:
        tuple: (test_images, test_labels)
    """
    test_images = []
    test_labels = []
    
    # Character class mapping
    if char_type == 'chars':
        class_map = {char: i for i, char in enumerate(CHARS_CLASS)}
        char_dir = CHAR_DIR
    else:  # Chinese characters
        # Only use pinyin directories (odd indices in PROVINCES)
        class_map = {}
        for i in range(0, len(PROVINCES), 2):
            if i < len(PROVINCES) and PROVINCES[i].startswith('zh_'):
                class_map[PROVINCES[i]] = i // 2
        char_dir = CHINESE_CHAR_DIR
    
    # Walk through directory structure
    for root, _, files in os.walk(os.path.join(test_dir, char_dir)):
        class_name = os.path.basename(root)
        if class_name in class_map:
            class_idx = class_map[class_name]
            
            for file in files:
                if file.endswith(('.jpg', '.png', '.bmp')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Prepare image
                        processed_img = prepare_character_data(img)
                        test_images.append(processed_img)
                        test_labels.append(class_idx)
    
    return np.array(test_images), np.array(test_labels)