import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("id", type=str, help="image filename base")
args = parser.parse_args()

path = '/data/eval/'
filename = args.id
img = mpimg.imread(path + args.id + '.png')
plt.imshow(img)
plt.show()