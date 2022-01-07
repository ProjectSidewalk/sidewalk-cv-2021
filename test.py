import csv
import logging
import math
import multiprocessing as mp
from itertools import islice
from time import perf_counter
from PIL import Image, ImageDraw
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

pano_img_path = "pano-downloads/TxjXIx-S1n-iG_vH1HQEJg.jpg"
im = Image.open(pano_img_path)
draw = ImageDraw.Draw(im)

im_width = im.size[0]
im_height = im.size[1]
print(im_width, im_height)

pix = im.load()
print(pix[13312, 6656])
print(pix[13312, 6656] == (0, 0, 0))



