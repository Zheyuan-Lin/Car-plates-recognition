"""
Character recognition model for license plate characters (letters and digits)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.externals import joblib

from models.neural_network import TwoLayerNet
from models.base_recognizer import BaseRecognizer
from config import CHARS_CLASS, CHAR_DIR, TRAIN_PATH, CHAR_MODEL_PATH, LEARNING_RATE, MAX_ITERATIONS, BATCH_SIZE

class CharRecognizer(BaseRecognizer):
    """Character recognition model for license plate characters"""
    
    def __init__(self):
        """Initialize character recognizer"""
        super().__init__(
            chars_class=CHARS_CLASS,
            char_dir=CHAR_DIR,
            model_path=CHAR_MODEL_PATH
        )
    
    def load_data(self, path=TRAIN_PATH):
        """
        Load training data for character recognition
        
        Args:
            path: Path to training data directory
            
        Returns:
            tuple: (training_data, labels)
        """
        chars_train = []
        chars_label = []
        list_label = []
        k = 0
        
        # Walk through directory structure to load images
        for root, dirs, files in os.walk(os.path.join(path, CHAR_DIR)):
            if len(os.path.basename(root)) > 1:
                continue
                
            index = self.chars_class.index(os.path.basename(root))
            for filename in files:
                filepath = os.path.join(root, filename)
                # Use PIL instead of cv2
                digit_img = Image.open(filepath).convert('L')  # Convert to grayscale
                digit_img = np.array(digit_img).flatten() / 255  # Convert to 1D vector and normalize
                chars_train.append(digit_img)
                chars_label.append(index)
                
        chars_train = np.array(chars_train)
        
        # Convert to one-hot encoding
        for i in range(len(chars_label)):
            single_label = [0] * len(self.chars_class)
            if chars_label[i] == k:
                single_label[k] = 1
                list_label.append(single_label)
            if i != len(chars_label) - 1:
                if chars_label[i + 1] != k:
                    k = k + 1
                    
        chars_label = np.array(list_label)
        return chars_train, chars_label
    
    def train_minibatch(self, plot_progress=True):
        """Train character recognition model using mini-batch gradient descent"""
        return super().train_minibatch(
            train_path=TRAIN_PATH,
            learning_rate=LEARNING_RATE,
            max_iterations=MAX_ITERATIONS,
            batch_size=BATCH_SIZE,
            plot_progress=plot_progress
        )
    
    def train_batch(self, plot_progress=True):
        """Train character recognition model using batch gradient descent"""
        return super().train_batch(
            train_path=TRAIN_PATH,
            learning_rate=LEARNING_RATE,
            max_iterations=MAX_ITERATIONS,
            plot_progress=plot_progress
        )
    
    def save_model(self, path=CHAR_MODEL_PATH):
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
    
    def load_model(self, path=CHAR_MODEL_PATH):
        """Load trained model from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict(self, image):
        """
        Predict character from image
        
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
        
        return y, self.chars_class[index]
    
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
    import time
    
    recognizer = CharRecognizer()
    
    # Train model
    model = recognizer.train_minibatch()
    
    # Save model
    recognizer.save_model()
    
    print("Character recognition model training completed")