import pandas as pd
import yaml
from tqdm import tqdm
import shutil
import numpy as np
from pathlib import Path

import glob
import os

##### load config file ####
with open('config.yaml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

##### read dataset files ####
# def read_category_keys(file=config['CATEGORY_KEY_FILE'], index_col=config['INDEX']):
#     return pd.read_json(file).set_index(config['INDEX'])


def read_train_image_data(file=config['TRAIN_IMAGE_DATA'], index_col=config['INDEX']):
    return pd.read_json(file).set_index(config['INDEX'])


# def read_eval_image_data(file=config['EVAL_IMAGE_DATA'], index_col=config['INDEX']):
#     return pd.read_json(file).set_index(config['INDEX'])


def read_annotation_data(file=config['ANNOTATION_FILE'], index_col=config['INDEX']):
    return pd.read_json(file).set_index(config['INDEX'])


#### create datasets for yolov8 ####
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


#### submission file helper functions ####
def cats_to_list(category_str):
    '''Takes a string of detection categories delimited by spaces and returns
    a list of integers
    Args:
        category_str: (string) list of categories delimited by ' ' (e.g. '12 52 34')
    Returns:
        category_lst: list(int) list of integer category labels (e.g. [12, 52, 34])
    '''
    cat_list_str = category_str.split(' ')
    return [int(x) for x in cat_list_str]


def convert_cats(category_lst):
    '''Converts a list of integers into a sting list of categories for submission.
    Args:
        category_lst: list(int) list of integer category labels (e.g. [12, 52, 34])
    Returns:
        category_str: (string) list of categories delimited by ' ' for submission 
        (e.g. '12 52 34' or '[12]')
    '''
    if len(category_lst) > 1:
        category_str = ' '.join([x for x in category_lst])
    else:
        category_str = '[' + str(category_lst[0]) + ']'
    return category_str


def select_top(category_str):
    '''Takes list of detections and returns top hit
    Args: 
        category_str: (string) list of categories delimited by ' ' for submission 
        (e.g. '12 52 34' or '[12]')
    Returns:
        category_str: (string) list len one of top detection (e.g. '[12]')
    '''
    category_str = str(category_str)
    if '[' not in category_str:
        top = [int(x) for x in category_str.split(' ')][0]
        return f'[{top}]'
    else:
        return category_str


def generate_submission_file(df, output):
    '''Generates file for submission.
    Args:
        df: (pd.DataFrame) dataframe of inference output
        output: (string) output filename
    '''
    df[['id', 'categories', 'osd']].to_csv(output, index=False)


def remap(cat_lst, mapper):
    '''Remaps reduced list of detections to original detection categories.
    Args:
        cat_lst: list(int) list of integer category labels (e.g. [12, 52, 34])
        mapper: (dict) input category id to original category id
    Returns:
        out: list(int) list of integer category labels (e.g. [12, 52, 34])
    '''
    out = []
    for i in cat_lst:
        out.append(mapper.get(i))
    return out


def generate_submission_df(input_path='runs/detect/predict/labels/', conf_threshold=0.25):
    '''Generates dataframe of information from infrence output files.
    Args:
        input_path: (string) path to prediction labels files
        conf_thrreshold: (float) minimum confidence threshold for valid detection
    Returns:
        df: (pd.DataFrame) dataframe of inference output
    '''
    out = {}
    filelist = glob.glob(input_path + '*.txt')

    cat_df = pd.read_json('category_key.json')
    shallow = cat_df[cat_df.shallow_species == True]['index'].to_list()
    mapper = cat_df[['id', 'index']].to_dict()['id']

    for i, file in enumerate(filelist):
        with open(file, 'r') as f:
            conf = []
            cats = []
            weak_shallow = 0
            strong_shallow = 0
            osd = 0.7  # set base case that there are detections, this will hold if no shallow detects
            for line in f.readlines():
                category = int(line.split(' ')[0])
                conf_value = float(line.split(' ')[-1])

                if category in shallow: # chech shallow and set osd to 0 if in shallow
                    weak_shallow = 1    # weakly shallow if there is a shallow detection at any confidence
                    if conf_value >= conf_threshold:
                        strong_shallow = 1  # strongly shallow if a high conf shallow detection

                if (category not in cats) and (conf_value >= conf_threshold): # dedup and add to list
                    cats.append(category)
                    conf.append(conf_value)
            
            cats = remap(cats, mapper)

            
            if len(cats) > 1:
                cats = ' '.join([str(x) for x in cats])

            if len(cats) == 0:
                cats = '[52]'
                osd = 1.0
            
            if weak_shallow:
                osd = 0.3
            if strong_shallow:
                osd = 0.0



        out[i] = {'id': os.path.basename(file)[:-4],
                  'categories': cats,
                  'osd': osd,
                  'conf': conf
                  }

    df = pd.DataFrame.from_dict(out, orient='index')
    return df