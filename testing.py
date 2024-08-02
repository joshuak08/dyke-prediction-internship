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
parser.add_argument('--learningParam', type=str, default='strike', choices=["strike", "opening", "top", "bottom", "length"])
parser.add_argument("--imageType", type=str, default='DST_Coherence')
parser.add_argument('--imageSize', type=int, default=512)
parser.add_argument("--lossFunction", type=str, default="L1", choices=["L1", "L2"])
parser.add_argument("--weights", type=str, default='/output')
parser.add_argument("--inputDir", type=str, default="/Dyke")

args = parser.parse_args()

network = args.network
image_type = args.imageType
learning_param = args.learningParam
chosen_loss = args.lossFunction
image_size = args.imageSize
weights = args.weights
inputDir = args.inputDir

excel_filename = os.getcwd() + f"{weights}/{network}/{chosen_loss}/{learning_param}/{image_type}" + f'/test_results.xlsx'
workbook = xlsxwriter.Workbook(excel_filename)

test_images_src = os.getcwd() + f"{inputDir}/{image_type}/file_split.json"
with open(test_images_src, 'r') as f:
    files = json.load(f)['test']

# add different architectures here
model_choices = {"swin": SwinTransformerV2(img_size=image_size), 'convnext': ConvNeXt(), 
                 'deit': vit_models(img_size=image_size), 'crossvit': VisionTransformer(img_size=(image_size, image_size)),
                 'cait': cait_models(img_size=image_size), 'vmamba': vmamba.vanilla_vmamba_small(image_size)}

for model_weights in glob.glob(f'.{weights}/{network}/{chosen_loss}/{learning_param}/{image_type}/*.pth'):
    model_no = os.path.split(model_weights)[-1]
    # excel file initilisation
    worksheet = workbook.add_worksheet(model_no)
    row = 0
    worksheet.write(row, 0, 'image')
    worksheet.write(row, 1, 'original label')
    worksheet.write(row, 2, 'output label')
    worksheet.write(row, 3, 'difference')
    row += 1

    difference = []
    print("===>Testing using weights: ", model_weights)

    # setup the model using trained weights
    model = model_choices[network]
    checkpoint = torch.load(model_weights)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key[7:]] = checkpoint[key] #delete term "module"
            del checkpoint[key]
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    model = nn.DataParallel(model)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for file in files:
            image = cv2.imread(file)

            # replace nan with mean values
            image = np.nan_to_num(image, nan=np.mean(image))

            # image = cv2.resize(image, (512, 512))
            image = np.divide(image, 255.0)
            filename = os.path.basename(file).split('/')[-1]

            params = {"strike": 1, "opening": 5, "top": 7, "bottom": 9, "length": 11}
            label = torch.tensor([float(filename.split("_")[params[learning_param]])]).to(device)
            # label = torch.tensor([float(filename.split("_")[1])]).to(device)
            image = np.float32(cv2.resize(image, (image_size, image_size)))
            image = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2)

            output = model(image)
            # if math.isnan(output):
            #     print(file)
            #     print(weights)
            # print(type(filename.split("_")[1]))
            worksheet.write(row, 0, filename)
            worksheet.write(row, 1, "".join(filename.split("_")[params[learning_param]]))
            worksheet.write(row, 2, output)

            output = output % 180
            output = (output + 180) % 180

            # Compute the angular difference
            angular_diff = abs(output - label)
            angular_diff = min(angular_diff, 180 - angular_diff)
            angular_diff = angular_diff.cpu()
            worksheet.write(row, 3, angular_diff)

            difference.append(angular_diff)

            row += 1
    worksheet.write(2, 6, 'Avg Difference')
    worksheet.write(2, 7, np.mean(difference))
    worksheet.write(3, 6, 'Min Difference')
    worksheet.write(3, 7, np.min(difference))
    worksheet.write(4, 6, 'Max Difference')
    worksheet.write(4, 7, np.max(difference))
workbook.close()
exit()
