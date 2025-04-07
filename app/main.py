"""
Main application for license plate recognition
"""
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

from preprocessing.plate_locator import PlateLocator
from preprocessing.char_segmentation import CharSegmentation
from models.char_recognizer import CharRecognizer
from models.chinese_recognizer import ChineseRecognizer
from config import CHAR_MODEL_PATH, CHINESE_MODEL_PATH


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="License Plate Recognition")
    
    # Define command-line arguments
    parser.add_argument("-i", "--image", help="Path to input image")
    parser.add_argument("--train", action="store_true", help="Train both models")
    parser.add_argument("--train-chars", action="store_true", help="Train only character model")
    parser.add_argument("--train-chinese", action="store_true", help="Train only Chinese character model")
    parser.add_argument("--visualize", action="store_true", help="Visualize the recognition process")
    parser.add_argument("--batch", action="store_true", help="Use batch training instead of mini-batch")
    
    return parser.parse_args()


def recognize_plate(image_path, visualize=False):
    """
    Recognize license plate from image
    
    Args:
        image_path: Path to input image
        visualize: Whether to visualize the recognition process
        
    Returns:
        str: Recognized license plate number
    """
    # Start timing
    start_time = time.time()
    
    # Read input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    original_image = image.copy()
    
    # Initialize components
    plate_locator = PlateLocator()
    char_segmenter = CharSegmentation()
    char_recognizer = CharRecognizer()
    chinese_recognizer = ChineseRecognizer()
    
    # Load pre-trained models
    try:
        char_recognizer.load_model(CHAR_MODEL_PATH)
        chinese_recognizer.load_model(CHINESE_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the models first using --train option.")
        return None
    
    # Step 1: Locate license plate
    print("Locating license plate...")
    plate_image, success = plate_locator.locate(image)
    
    if not success:
        print("Failed to locate license plate.")
        return None
    
    # Step 2: Segment characters
    print("Segmenting characters...")
    char_images = char_segmenter.segment(plate_image, visualize=visualize)
    
    if not char_images:
        print("Failed to segment characters.")
        return None
    
    # Step 3: Recognize characters
    print("Recognizing characters...")
    plate_chars = []
    
    # Process each character
    for i, char_img in enumerate(char_images):
        # First character is Chinese province character
        if i == 0:
            _, char = chinese_recognizer.predict(char_img)
        else:
            _, char = char_recognizer.predict(char_img)
        
        plate_chars.append(char)
    
    # Combine characters to form plate number
    plate_number = ''.join(plate_chars)
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Display results
    print(f"\nRecognized plate: {plate_number}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Visualize recognition results if requested
    if visualize:
        # Prepare visualization of the recognition process
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with plate highlighted
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Extracted plate
        axes[1].imshow(plate_image, cmap='gray')
        axes[1].set_title("Extracted Plate")
        axes[1].axis('off')
        
        # Segmented characters
        char_imgs_concat = np.hstack([np.pad(img, ((0, 0), (1, 1)), mode='constant', constant_values=255) 
                                     for img in char_images])
        axes[2].imshow(char_imgs_concat, cmap='gray')
        axes[2].set_title("Segmented Characters")
        axes[2].axis('off')
        
        # Add recognition result
        plt.suptitle(f"Recognized Plate: {plate_number}", fontsize=16)
        
        plt.tight_layout()
        plt.show()
    
    return plate_number


def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Train models if requested
    if args.train or args.train_chars or args.train_chinese:
        if args.train or args.train_chars:
            print("\n=== Training Character Recognition Model ===")
            char_recognizer = CharRecognizer()
            if args.batch:
                char_recognizer.train_batch()
            else:
                char_recognizer.train_minibatch()
            char_recognizer.save_model()
            
        if args.train or args.train_chinese:
            print("\n=== Training Chinese Character Recognition Model ===")
            chinese_recognizer = ChineseRecognizer()
            if args.batch:
                chinese_recognizer.train_batch()
            else:
                chinese_recognizer.train_minibatch()
            chinese_recognizer.save_model()
            
        print("\nTraining completed successfully!")
        return
    
    # Recognize license plate from image
    if args.image:
        try:
            recognize_plate(args.image, visualize=args.visualize)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide an input image using the --image argument.")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()