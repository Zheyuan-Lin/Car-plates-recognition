"""
Configuration parameters for the license plate recognition system
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAINING_DIR = os.path.join(BASE_DIR, 'training')
PREPROCESSING_DIR = os.path.join(BASE_DIR, 'preprocessing')

LEARNING_RATE = 0.1
MAX_ITERATIONS = 10
BATCH_SIZE = 3

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, TRAINING_DIR, PREPROCESSING_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data paths
TRAIN_PATH = "data/train"
TEST_PATH = "data/test"

# Model parameters
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 36,  # 0-9 and A-Z
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# Training parameters
TRAINING_CONFIG = {
    'validation_split': 0.2,
    'test_split': 0.1,
    'random_seed': 42
}

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    'target_size': (224, 224),
    'normalize': True,
    'augmentation': True
}

# Plate detection parameters
BLUE = 138
GREEN = 63
RED = 23
COLOR_THRESHOLD = 50
ANGLE_THRESHOLD = -45
MIN_AREA = 2000
LICENSE_WIDTH = 440
LICENSE_HEIGHT = 140
MAX_WIDTH = 640

# Character classes
CHARS_CLASS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G",
    "H", "J", "K", "L", "M", "N", "P",
    "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z"
]

# Chinese province characters (with Pinyin mapping)
PROVINCES = [
    "zh_chuan", "川", "zh_e", "鄂", "zh_gan", "赣", "zh_gan1", "甘", 
    "zh_gui", "贵", "zh_gui1", "桂", "zh_hei", "黑", "zh_hu", "沪", 
    "zh_ji", "冀", "zh_jin", "津", "zh_jing", "京", "zh_jl", "吉", 
    "zh_liao", "辽", "zh_lu", "鲁", "zh_meng", "蒙", "zh_min", "闽", 
    "zh_ning", "宁", "zh_qing", "靑", "zh_qiong", "琼", "zh_shan", "陕", 
    "zh_su", "苏", "zh_sx", "晋", "zh_wan", "皖", "zh_xiang", "湘", 
    "zh_xin", "新", "zh_yu", "豫", "zh_yu1", "渝", "zh_yue", "粤", 
    "zh_yun", "云", "zh_zang", "藏", "zh_zhe", "浙"
]

# Directory names
CHAR_DIR = "chars"
CHINESE_CHAR_DIR = "charsChinese"

# Model file paths
CHAR_MODEL_PATH = "models/saved/charNet.pkl"
CHINESE_MODEL_PATH = "models/saved/chineseNet.pkl"