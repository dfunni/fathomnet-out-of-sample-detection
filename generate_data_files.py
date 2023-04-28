from pycocotools.coco import COCO
import pandas as pd

DATAPATH = "/data/"
EVALPATH = "/data/eval/"
TRAINPATH = "/data/train/"

coco = COCO("object_detection/train.json")
coco_eval = COCO("object_detection/eval.json")


## TODO fix this section to read from json file
# shallow_species = {160: True,
#                    37: False,
#                    119: True,
#                    51: True,
#                    10: True,
#                    146: False, # 1087m
#                    52: False,
#                    88: False,
#                    125: False,
#                    203: False, # 927m
#                    214: True, # not much data
#                    1: False,
#                    259: True,
#                    9: False, # 1000m
#                    105: True,
#                    211: False,
#                    133: True,
#                    142: False,
#                    70: False,
#                    260: True,
#                    274: True,
#                    174: False,
#                    205: False, # not much data
#                    120: False,
#                    219: False, # not much data
#                    81: False,
#                    69: False,
#                    104: True,
#                    218: False,
#                    16: False,
#                    103: True,
#                    224: False,
#                    228: False,
#                    242: False,
#                    61: True, # mostly
#                    116: True,
#                    255: False,
#                    202: False,
#                    108: False, # unknown
#                    11: False
#                    }

cat_df = pd.DataFrame.from_dict(coco.cats, orient='index')


cat_df = cat_df.join(pd.Series(shallow_species, name='shallow_species'), on='id').dropna()#
cat_df = cat_df.reset_index(drop=True)#

cat_df['index'] = range(len(cat_df))


map_dict = cat_df[['id', 'index']].to_dict()['id'] #
map_dict = {value:key for key, value in map_dict.items()}#

ann_df = pd.DataFrame.from_dict(coco.anns, orient='index')


print(ann_df['image_id'].unique().__len__()) #
ann_df = ann_df.join(pd.Series(shallow_species, name='shallow_species'), on='category_id').dropna()#
ann_df['original_category'] = ann_df['category_id']#
ann_df['category_id'] = ann_df['category_id'].map(map_dict)#
remaining_images = pd.Series(ann_df['image_id'].unique(), name='id')#

train_img_df = pd.DataFrame.from_dict(coco.imgs, orient='index')


train_img_df = train_img_df.merge(remaining_images, on=['id'])#

eval_img_df = pd.DataFrame.from_dict(coco_eval.imgs, orient='index')

cat_df.to_json('category_key.json')
ann_df.to_json('annotation.json')
train_img_df.to_json('train_image_data.json')
eval_img_df.to_json('eval_image_data.json')

if __name__ == '__main__':
    pass
