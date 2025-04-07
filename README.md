# License Plate Recognition System

A complete license plate recognition system implemented in Python without using deep learning frameworks like PyTorch or TensorFlow. This system uses traditional computer vision techniques and a custom-built neural network for character recognition.

## Features

- License plate detection in images
- Plate extraction and perspective correction
- Character segmentation
- Character recognition (letters, digits, and Chinese characters)
- Simple and modular architecture
- Visualization tools for debugging

## Project Structure

```
license_plate_recognition/
├── models/
│   ├── __init__.py
│   ├── neural_network.py       # Base neural network components
│   ├── char_recognizer.py      # Character recognition model
│   └── chinese_recognizer.py   # Chinese character recognition model
├── preprocessing/
│   ├── __init__.py
│   ├── plate_locator.py        # License plate detection in image
│   └── char_segmentation.py    # Character segmentation from plate
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # Image processing utilities
│   └── data_utils.py           # Data loading and processing
├── training/
│   ├── __init__.py
│   ├── train_char_model.py     # Train character model
│   └── train_chinese_model.py  # Train Chinese character model
├── config.py                   # Configuration and parameters
├── main.py                     # Main application entry point
└── README.md                   # Project documentation
```

## Requirements

- Python 3.6+
- OpenCV 4.0+
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages using the following command:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

## Usage

### Training Models

To train the character recognition models, run:

```bash
python main.py --train
```

For batch training instead of mini-batch:

```bash
python main.py --train --batch
```

### Recognizing License Plates

To recognize a license plate in an image:

```bash
python main.py --image path/to/image.jpg
```

To visualize the recognition process:

```bash
python main.py --image path/to/image.jpg --visualize
```

## Data Organization

The training data should be organized as follows:

```
data/
├── train/
│   ├── chars/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── 2/
│   │   └── ...
│   └── charsChinese/
│       ├── zh_chuan/
│       ├── zh_e/
│       └── ...
├── test/
│   ├── chars/
│   └── charsChinese/
```

Each subdirectory contains character images for training, organized by character class.

## Model Architecture

The character recognition model uses a simple two-layer neural network:

- Input layer (400 nodes): Flattened 20x20 character images
- Hidden layer (20 nodes): With ReLU activation
- Output layer (34 nodes for char, 31 nodes for Chinese): With softmax activation

The model is trained using mini-batch gradient descent with cross-entropy loss.

## License Plate Detection

The license plate detection algorithm works as follows:

1. Preprocess the image (resizing, Gaussian blur, median filter)
2. Filter pixels by color to isolate blue license plate regions
3. Apply morphological operations to enhance plate regions
4. Find contours and select the most likely license plate based on aspect ratio
5. Extract and rectify the license plate using perspective transform

## Character Segmentation

Character segmentation is performed using vertical projection analysis:

1. Remove plate frame and rivets using jump count analysis
2. Calculate vertical projection (count of black pixels by column)
3. Find character regions based on projection peaks
4. Extract individual character images
5. Resize characters to standard 20x20 size for recognition

## Performance Considerations

- The system is optimized for Chinese license plates but can be adapted for other formats
- For improved accuracy, you may need to adjust threshold parameters based on your specific images
- The neural network model is relatively simple; more complex architectures could improve recognition accuracy
- For large-scale applications, consider implementing batch processing for improved efficiency

## Future Improvements

- Add support for video processing
- Implement more advanced neural network architectures
- Add data augmentation for improved training
- Optimize performance for real-time recognition
- Add support for different license plate formats from various countries

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.