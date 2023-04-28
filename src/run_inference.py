import warnings
warnings.filterwarnings("ignore")

import yaml
import torch
import ultralytics
from ultralytics import YOLO

ultralytics.checks()
torch.cuda.empty_cache()

with open('/data/config.yaml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

model = YOLO('runs/detect/FathomNet-YOLOv82/weights/best.pt')

predict = model.predict(source=config['EVAL_IMAGES_ROOT'],
                      save=True,
                      save_txt=True,
                      save_conf=True,
                      conf=config['CONFIDENCE'])
