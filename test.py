from threeD_utils import fun001
#from combine import *

import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess

import numpy as np
from skimage.transform import estimate_transform, warp
from cv2 import cv2 as cv2 
from predictor import PosPrediction
import matplotlib.pyplot as plt
from cartoon import Photo2Cartoon

parser = argparse.ArgumentParser()
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
    c2p = Photo2Cartoon()
    cartoon = c2p.inference(img)
    if cartoon is not None:
        cv2.imwrite(args.save_path, cartoon)
        print('Cartoon portrait has been saved successfully!')
    
#####################################################################
    cv2.imshow("cartoon",cartoon)
    cv2.waitKey(0)
    Funnc = fun001()

    img = Funnc.gbrtorgb(cartoon)
    
    bbox = Funnc.fun001_1(img)

    cropped_img, tform, img = Funnc.fun001_2(bbox, img)
    
    pos = Funnc.fun001_3(cropped_img, tform)

    vertices, uv_kpt_ind, face_ind = Funnc.fun001_4(pos, img)

    texture = Funnc.fun001_5(vertices, uv_kpt_ind,pos, img)

    img_3D ,tri_tex = Funnc.fun001_6(texture, vertices, face_ind, img)

    Funnc.fun001_7(vertices, img_3D, img, tri_tex)