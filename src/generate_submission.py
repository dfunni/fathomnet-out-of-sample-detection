import glob
import os
import pandas as pd
import argparse

def remap(cat_lst, mapper, shallow, cat_df):
    cat_df[cat_df.shallow_species == True]['index'].to_list()
    out = []
    osd = 0.9 # set for base case that there are objects found
    for i in cat_lst:
        osd = 0.5
        if i in shallow:
            osd = 0.1 # update if shallow species present
        out.append(str(mapper.get(i)))
    return out, osd

def main(args):
    out = {}
    filelist = glob.glob('runs/detect/predict2/labels/*.txt')

    cat_df = pd.read_json('category_key.json')
    shallow = [10, 51, 61, 103, 104, 105, 116, 119, 133, 160, 214, 259, 260, 274] ## FIXME
    mapper = cat_df[['id', 'index']].to_dict()['id']

    for i, file in enumerate(filelist):
        with open(file, 'r') as f:
            cats = []
            osd = 0.9
            for line in f.readlines():
                category = int(line.split(' ')[0])
                if category in shallow:
                    osd = 0.1
                if category not in cats:
                    cats.append(category)
    
            cats, osd = remap(cats, mapper, shallow, cat_df)

            
            if len(cats) > 1:
                cats = ' '.join([str(x) for x in cats])
            if len(cats) == 0:
                cats = '[52]'


        out[i] = {'id': os.path.basename(file)[:-4], 'categories': cats, 'osd': osd}

    df = pd.DataFrame.from_dict(out, orient='index')
    df[['id', 'categories', 'osd']].to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inference data into submission file.')
    parser.add_argument('--output', type=str, default='submission_12_int.csv', help='output file')
    args = parser.parse_args()

    main(args)