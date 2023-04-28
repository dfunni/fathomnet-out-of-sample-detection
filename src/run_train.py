import warnings
warnings.filterwarnings("ignore")

import yaml
from sklearn.model_selection import train_test_split
import torch
import ultralytics
from ultralytics import YOLO
import os
import pandas as pd

import utils

ultralytics.checks()
torch.cuda.empty_cache()

with open('/data/config.yaml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)


os.system('../clear_dataset.sh')


train_image_data = utils.read_train_image_data()
annotation_data = utils.read_annotation_data()

# category_data = utils.read_category_keys()
# eval_image_data = utils.read_eval_image_data()


data = train_image_data.sample(frac=config['SAMPLE_SIZE'], random_state=config['RANDOM_STATE'])
X_train, X_val = train_test_split(
    data, 
    test_size=config['TEST_SIZE'], 
    random_state=config['RANDOM_STATE'])

print('\nCreating train dataset')
utils.create_yolo_dataset(X_train, annotation_data, data_type='train')
print('\nCreating val dataset')
utils.create_yolo_dataset(X_val, annotation_data, data_type='val')

print('\nRuning model')
model = YOLO('yolov8m.pt')

results = model.train(
   data=str(config['DATASET_CONFIG']),
   epochs=config['N_EPOCHS'],
   imgsz=640,
   batch=config['N_BATCH'],
   save=True, 
   verbose=False,
   name=config['MODEL_NAME'])
   