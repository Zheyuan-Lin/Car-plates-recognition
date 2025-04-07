"""
Base Neural Network Components for License Plate Recognition
"""
import numpy as np
from collections import OrderedDict

class Layer:
    """Base class for neural network layers"""
    def __init__(self):
        self.params = {}
    
    def forward(self, x):
        raise NotImplementedError
        
    def backward(self, dout):
        raise NotImplementedError


class Affine(Layer):
    """Fully connected layer"""
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        """Forward pass"""
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
        
    def backward(self, dout):
        """Backward pass"""
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Relu(Layer):
    """ReLU activation function"""
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        """Forward pass"""
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
        
    def backward(self, dout):
        """Backward pass"""
        dout[self.mask] = 0
        dx = dout
        return dx


class SoftmaxWithLoss:
    """Softmax activation with cross-entropy loss"""
    def __init__(self):
        self.loss = None
        self.y = None  # softmax output
        self.t = None  # target labels
        
    def forward(self, x, t):
        """Forward pass"""
        self.t = t
        self.y = self._softmax(x)
        self.loss = self._cross_entropy_error(self.y, self.t)
        return self.loss
        
    def backward(self, dout=1):
        """Backward pass"""
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # if one-hot vector
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
    
    def _softmax(self, x):
        """Softmax activation function with overflow handling"""
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)  # Overflow countermeasure
        return np.exp(x) / np.sum(np.exp(x))
    
    def _cross_entropy_error(self, y, t):
        """Cross-entropy loss function"""
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        # Convert one-hot vector to label indices
        if t.size == y.size:
            t = t.argmax(axis=1)
            
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class TwoLayerNet:
    """Two-layer neural network for classification"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # Create layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        """Forward prediction"""
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        """Calculate loss"""
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
        
    def accuracy(self, x, t):
        """Calculate accuracy"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        """Calculate gradient numerically (slow but accurate)"""
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = self._numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self._numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self._numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self._numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        """Calculate gradient using backpropagation (faster)"""
        # Forward
        self.loss(x, t)
        
        # Backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        # Store gradients
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads
    
    def _numerical_gradient(self, f, x):
        """Helper method to calculate numerical gradient"""
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)
        
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)
            
            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            
            x[idx] = tmp_val  # restore value
            it.iternext()
            
        return grad