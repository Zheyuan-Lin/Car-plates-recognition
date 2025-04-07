"""
Base class for character recognition models
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.externals import joblib

from models.neural_network import TwoLayerNet

class BaseRecognizer:
    """Base class for character recognition models"""
    
    def __init__(self, chars_class, char_dir, model_path, input_size=400, hidden_size=20):
        """Initialize base recognizer"""
        self.chars_class = chars_class
        self.char_dir = char_dir
        self.model_path = model_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = len(chars_class)
        self.model = None
        self.train_loss_list = []
        self.train_acc_list = []
    
    def load_data(self, path):
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
        for root, dirs, files in os.walk(os.path.join(path, self.char_dir)):
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
    
    def initialize_model(self):
        """Initialize a new neural network model for character recognition"""
        self.model = TwoLayerNet(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=self.output_size
        )
        return self.model
    
    def train_minibatch(self, train_path, learning_rate, max_iterations, batch_size, plot_progress=True):
        """
        Train character recognition model using mini-batch gradient descent
        
        Args:
            train_path: Path to training data
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of training iterations
            batch_size: Size of mini-batches
            plot_progress: Whether to plot training metrics
            
        Returns:
            TwoLayerNet: Trained model
        """
        # Clear previous training data
        self.train_loss_list = []
        self.train_acc_list = []
        
        # Load training data
        chars_train, chars_label = self.load_data(train_path)
        print(f"Training set shape: {chars_train.shape}")
        print(f"Label shape: {chars_label.shape}")
        
        # Initialize model if not already initialized
        if self.model is None:
            self.initialize_model()
        
        # Training parameters
        train_size = chars_train.shape[0]
        iter_per_epoch = int(max(train_size / batch_size, 1))
        
        # Training loop
        print("Starting mini-batch training...")
        import time
        start_time = time.time()
        
        for i in range(max_iterations):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = chars_train[batch_mask]
            t_batch = chars_label[batch_mask]
            
            # Calculate gradients and update parameters
            grad = self.model.gradient(x_batch, t_batch)
            for key in ("W1", "b1", "W2", "b2"):
                self.model.params[key] -= learning_rate * grad[key]
            
            # Calculate loss for monitoring
            loss = self.model.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)
            
            # Calculate accuracy periodically
            if i % iter_per_epoch == 0:
                train_acc = self.model.accuracy(chars_train, chars_label)
                test_acc = self.model.accuracy(x_batch, t_batch)
                self.train_acc_list.append(train_acc)
                print(f"Iteration {i}: train acc = {train_acc:.4f}, test acc = {test_acc:.4f}")
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Plot training metrics if requested
        if plot_progress:
            self.plot_metrics()
        
        return self.model
    
    def train_batch(self, train_path, learning_rate, max_iterations, plot_progress=True):
        """
        Train character recognition model using batch gradient descent
        
        Args:
            train_path: Path to training data
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of training iterations
            plot_progress: Whether to plot training metrics
            
        Returns:
            TwoLayerNet: Trained model
        """
        # Clear previous training data
        self.train_loss_list = []
        self.train_acc_list = []
        
        # Load training data
        chars_train, chars_label = self.load_data(train_path)
        print(f"Training set shape: {chars_train.shape}")
        print(f"Label shape: {chars_label.shape}")
        
        # Initialize model if not already initialized
        if self.model is None:
            self.initialize_model()
        
        # Training loop
        print("Starting batch training...")
        import time
        start_time = time.time()
        
        for i in range(max_iterations):
            # Calculate gradients and update parameters (using entire dataset)
            grad = self.model.gradient(chars_train, chars_label)
            for key in ("W1", "b1", "W2", "b2"):
                self.model.params[key] -= learning_rate * grad[key]
            
            # Calculate loss for monitoring
            loss = self.model.loss(chars_train, chars_label)
            self.train_loss_list.append(loss)
            
            # Calculate accuracy
            train_acc = self.model.accuracy(chars_train, chars_label)
            self.train_acc_list.append(train_acc)
            
            if i % 100 == 0:
                print(f"Iteration {i}: train acc = {train_acc:.4f}, loss = {loss:.6f}")
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Plot training metrics if requested
        if plot_progress:
            self.plot_metrics()
        
        return self.model
    
    def save_model(self):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_minibatch() or train_batch() first.")
        
        directory = os.path.dirname(self.model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
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
    
    def plot_metrics(self):
        """Plot training loss and accuracy metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_list)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_list)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show() 