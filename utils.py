import pandas as pd
import yaml
from tqdm import tqdm
import shutil
import numpy as np
from pathlib import Path

with open('config.yaml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

def read_category_keys(file=config['CATEGORY_KEY_FILE'], index_col=config['INDEX']):
    return pd.read_json(file).set_index(config['INDEX'])


def read_train_image_data(file=config['TRAIN_IMAGE_DATA'], index_col=config['INDEX']):
    return pd.read_json(file).set_index(config['INDEX'])


def read_eval_image_data(file=config['EVAL_IMAGE_DATA'], index_col=config['INDEX']):
    return pd.read_json(file).set_index(config['INDEX'])


def read_annotation_data(file=config['ANNOTATION_FILE'], index_col=config['INDEX']):
    return pd.read_json(file).set_index(config['INDEX'])


def create_bboxes(annotation_data):
    '''
    Args:
        annotation_data: dataframe of annotations with keys: image_id, category_id, bbox
    '''
    
    df = annotation_data.copy()
    # df['bbox'] = df['bbox'].apply(ast.literal_eval)

    groupby_image = df.groupby(by='image_id')
    df = groupby_image['bbox'].apply(list).reset_index(name='bboxes').set_index('image_id')
    df['category_ids'] = groupby_image['category_id'].apply(list)  
    return df

def create_yolo_dataset(image_data, annotation_data, data_type='train', image_root=config['IMAGES_ROOT'], dataset_root=config['DATASET_ROOT']):
    '''
    Args:
        image_data: dataframe with keys: image_id(index), file_name(.jpg), width, height
        annotation_data: dataframe of annotations with keys: image_id(index), category_id, bbox
        data_type: string from options ['train', 'eval', 'test']
        image_root: root path to images
        dataset_root: path to output dataset i.e. /data/dataset/
    '''

    bboxes_data = create_bboxes(annotation_data)
    image_ids = image_data.index
    
    for image_id in tqdm(image_ids, total=len(image_ids)):
        bounding_bboxes = bboxes_data['bboxes'].loc[image_id]
        category_ids = bboxes_data['category_ids'].loc[image_id]

        image_row = image_data.loc[image_id]
        image_width = image_row['width']
        image_height = image_row['height']
        
        file_name = Path(image_row['file_name']).with_suffix('.png')
        source_image_path = Path(config['TRAIN_IMAGES_ROOT']) / file_name
        target_image_path = Path(dataset_root) / f'images/{data_type}/{file_name}'
        label_path = (Path(dataset_root) / f'labels/{data_type}/{file_name}').with_suffix('.txt')
        
        #print(file_name)
        
        yolo_data = []
        for bbox, category in zip(bounding_bboxes, category_ids):
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = x + w/2
            y_center = y + h/2
            x_center /= image_width
            y_center /= image_height
            w /= image_width
            h /= image_height
            
            yolo_data.append([category, x_center, y_center, w, h])

        yolo_data = np.array(yolo_data)

        # Create YOLO lable file
        np.savetxt(label_path, yolo_data, fmt=["%d", "%f", "%f", "%f", "%f"])

        # Copy image file
        shutil.copy(source_image_path, target_image_path)
