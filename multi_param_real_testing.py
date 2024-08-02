import json
import math
import numpy as np
import os
from tqdm import tqdm
import glob
import xlsxwriter

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnext import ConvNeXt
from models.crossvit import VisionTransformer
from models.deit import vit_models
from models.swin_transformer_v2 import SwinTransformerV2
from models.cait import cait_models
import models.vmamba as vmamba
import glob
import cv2

import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Testing model")

parser.add_argument('--network', type=str, default='convnext', 
                    choices=["convnext", "swin", "deit", "crossvit", "cait", "vmamba"], 
                    required=True, help='select model architecture')
parser.add_argument('--learningParam', '--names-list', nargs='+', default='[strike]', choices=["strike", "opening", "top", "bottom", "length"])
parser.add_argument("--imageType", type=str, default='DST_Coherence')
parser.add_argument('--imageSize', type=int, default=512)
parser.add_argument("--specificWeight", type=str, default="")
parser.add_argument("--weights", type=str, default="/output")
parser.add_argument("--inputDir", type=str, default="/real_dyke")

args = parser.parse_args()

network = args.network
image_type = args.imageType
learning_param = args.learningParam
numParam = len(learning_param)
strLearningParam = "_".join(learning_param)
image_size = args.imageSize
weights = args.weights
inputDir = args.inputDir

excel_filename = os.getcwd() + f"{weights}/{network}/{learning_param}/{image_type}" + f'/{network}_{strLearningParam}_real_results.xlsx'
workbook = xlsxwriter.Workbook(excel_filename)

model_choices = {"swin": SwinTransformerV2(img_size=image_size), 'convnext': ConvNeXt(), 
                 'deit': vit_models(img_size=image_size), 'crossvit': VisionTransformer(img_size=(image_size, image_size)),
                 'cait': cait_models(img_size=image_size), 'vmamba': vmamba.vanilla_vmamba_small(image_size)}

for model_weights in glob.glob(os.getcwd() + f'{weights}/{network}/{learning_param}/{image_type}/*.pth'):
    model_no = os.path.split(model_weights)[-1]
    # excel file initilisation
    worksheet = workbook.add_worksheet(model_no)
    row = 0
    worksheet.write(row, 0, 'image name')
    col = 3
    for param in learning_param:
        worksheet.write(row, col, f'output {param} label')
        col += 1
        worksheet.write(row, 4, f'mod output {param} label')
        col += 1
    row += 1

    difference = []
    print("===>Testing using weights: ", model_weights)

    # setup the model using trained weights
    model = model_choices[network]
    checkpoint = torch.load(model_weights)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key[7:]] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    model = nn.DataParallel(model)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for file in glob.glob(os.getcwd() + inputDir + '/*.png'):
            image = cv2.imread(file)

            # replace nan with mean values
            image = np.nan_to_num(image, nan=np.mean(image))

            image = np.divide(image, 255.0)
            filename = os.path.basename(file).split('/')[-1]

            image = np.float32(cv2.resize(image, (image_size, image_size)))
            image = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2)

            output = model(image)

            worksheet.write(row, 0, filename)
            worksheet.insert_image(row, 1, file, {"x_scale": 0.25, "y_scale": 0.25})

            col = 3
            for param in learning_param:
                worksheet.write(row, col, output)
                col += 1
                output = output % 180
                output = (output + 180) % 180

                worksheet.write(row, col, output)
                col += 1


            row += 7
workbook.close()