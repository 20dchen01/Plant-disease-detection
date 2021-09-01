import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

import tensorflow as tf

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)

train_data = DataLoader.from_folder('Dataset/train')
test_data = DataLoader.from_folder('Dataset/test/')
validation_data = DataLoader.from_folder('Dataset/valid/')
spec = model_spec.get('resnet_50')
model = image_classifier.create(train_data, model_spec = spec)

model.evaluate(test_data)
model.export(export_dir='.')

# efficientdetVV
# spec = model_spec.get('efficientdet_lite2')
# train_data, validation_data, test_data = object_detector.DataLoader.from_csv('C:/Users/David/Desktop/Dissertation3/Dataset/dataset.csv')
# model = object_detector.create(train_data, model_spec=spec, batch_size=250, train_whole_model=True, validation_data=validation_data)
# model.evaluate(test_data)
# model.export(export_dir='.')
# model.evaluate_tflite('model.tflite', test_data)
