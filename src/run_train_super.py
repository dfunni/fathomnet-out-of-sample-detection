import warnings
warnings.filterwarnings("ignore")

import yaml
from sklearn.model_selection import train_test_split
import torch
import ultralytics
from ultralytics import YOLO

import utils

ultralytics.checks()
torch.cuda.empty_cache()

with open('/data/config_super.yaml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

# category_data = utils.read_category_keys()
train_image_data = utils.read_train_image_data()
# eval_image_data = utils.read_eval_image_data()
annotation_data = utils.read_annotation_data()


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
model = YOLO('yolov8l.pt')

results = model.train(
   data=str(config['DATASET_CONFIG']),
   epochs=config['N_EPOCHS'],
   imgsz=640,
   batch=config['N_BATCH'],
   save=True, 
   verbose=False,
   name=config['MODEL_NAME'])
   