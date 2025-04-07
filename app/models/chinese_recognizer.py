"""
Chinese character recognition model for license plate province characters
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.externals import joblib
from PIL import Image

from models.neural_network import TwoLayerNet
from models.base_recognizer import BaseRecognizer
from config import PROVINCES, CHINESE_CHAR_DIR, TRAIN_PATH, CHINESE_MODEL_PATH, LEARNING_RATE, MAX_ITERATIONS, BATCH_SIZE

class ChineseRecognizer(BaseRecognizer):
    """Chinese character recognition model for license plates"""
    
    def __init__(self):
        """Initialize Chinese character recognizer"""
        super().__init__(
            chars_class=PROVINCES,
            char_dir=CHINESE_CHAR_DIR,
            model_path=CHINESE_MODEL_PATH,
            output_size=31  # Number of Chinese province characters
        )
    
    def load_data(self, path=TRAIN_PATH):
        """
        Load training data for Chinese character recognition
        
        Args:
            path: Path to training data directory
            
        Returns:
            tuple: (training_data, labels)
        """
        chars_train = []
        chars_label = []
        list_label = []
        k = 1
        
        # Walk through directory structure to load images
        for root, dirs, files in os.walk(os.path.join(path, self.char_dir)):
            if not os.path.basename(root).startswith("zh_"):
                continue
                
            pinyin = os.path.basename(root)
            index = self.chars_class.index(pinyin) + 1
            
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = Image.open(filepath).convert('L')  # Convert to grayscale
                digit_img = np.array(digit_img).flatten() / 255  # Convert to 1D vector and normalize
                chars_train.append(digit_img)
                chars_label.append(index)
                
        chars_train = np.array(chars_train)
        
        # Convert to one-hot encoding
        num_classes = 31  # Number of Chinese province characters
        for i in range(len(chars_label)):
            single_label = [0] * num_classes
            if chars_label[i] == 2 * k - 1:
                single_label[k - 1] = 1
                list_label.append(single_label)
            if i != len(chars_label) - 1:
                if chars_label[i + 1] != 2 * k - 1:
                    k = k + 1
                    
        chars_label = np.array(list_label)
        return chars_train, chars_label
    
    def train_minibatch(self, plot_progress=True):
        """Train Chinese character recognition model using mini-batch gradient descent"""
        return super().train_minibatch(
            train_path=TRAIN_PATH,
            learning_rate=LEARNING_RATE,
            max_iterations=MAX_ITERATIONS,
            batch_size=BATCH_SIZE,
            plot_progress=plot_progress
        )
    
    def train_batch(self, plot_progress=True):
        """Train Chinese character recognition model using batch gradient descent"""
        return super().train_batch(
            train_path=TRAIN_PATH,
            learning_rate=LEARNING_RATE,
            max_iterations=MAX_ITERATIONS,
            plot_progress=plot_progress
        )
    
    def save_model(self, path=CHINESE_MODEL_PATH):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if os.path.exists(path):
            os.remove(path)
            
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=CHINESE_MODEL_PATH):
        """Load trained model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict(self, image):
        """
        Predict Chinese character from image
        
        Args:
            image: Image as numpy array (should be flattened and normalized)
            
        Returns:
            tuple: (probability_distribution, predicted_character)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Ensure image is properly formatted
        if image.ndim > 1:
            image = image.flatten() / 255
            
        # Get prediction
        y = self.model.predict(image.reshape(1, -1))
        index = np.argmax(y)
        
        # Get province character (odd indices in PROVINCES array contain actual characters)
        province_index = 2 * (index + 1) - 1
        if province_index < len(self.chars_class):
            return y, self.chars_class[province_index]
        else:
            return y, "Unknown"
    
    def _plot_metrics(self, data, title):
        """Helper method to plot training metrics"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(data)), data)
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel(title.lower())
        plt.grid(True)
        plt.show()


# For direct execution
if __name__ == "__main__":
    recognizer = ChineseRecognizer()
    recognizer.train_minibatch()
    recognizer.save_model()
    recognizer.load_model()
    recognizer.predict()