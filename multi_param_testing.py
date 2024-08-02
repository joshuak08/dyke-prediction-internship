import json
import numpy as np
import os
import glob
import xlsxwriter

import torch
import torch.nn as nn

from models.convnext import ConvNeXt
from models.crossvit import VisionTransformer
from models.deit import vit_models
from models.swin_transformer_v2 import SwinTransformerV2
from models.cait import cait_models
import models.vmamba as vmamba
import glob
import cv2

import argparse

parser = argparse.ArgumentParser(description="Testing model")

parser.add_argument('--network', type=str, default='convnext', 
                    choices=["convnext", "swin", "deit", "crossvit", "cait", "vmamba"], 
                    required=True, help='select model architecture')
parser.add_argument('--learningParam', '--names-list', nargs='+', default='[strike]', choices=["strike", "opening", "top", "bottom", "length"])
parser.add_argument("--imageType", type=str, default='DST_Coherence')
parser.add_argument('--imageSize', type=int, default=512)
parser.add_argument("--weights", type=str, default='/output')
parser.add_argument("--inputDir", type=str, default="/Dyke")

args = parser.parse_args()

network = args.network
image_type = args.imageType
learning_param = args.learningParam
numParam = len(learning_param)
strLearningParam = "_".join(learning_param)
image_size = args.imageSize
weights = args.weights
inputDir = args.inputDir

excel_filename = os.getcwd() + f"{weights}/{network}/{learning_param}/{image_type}" + f'/test_results.xlsx'
workbook = xlsxwriter.Workbook(excel_filename)

test_images_src = os.getcwd() + f"{inputDir}/{image_type}/file_split.json"
with open(test_images_src, 'r') as f:
    files = json.load(f)['test']

# add different architectures here
model_choices = {"swin": SwinTransformerV2(img_size=image_size, num_classes=numParam), 'convnext': ConvNeXt(), 
                 'deit': vit_models(img_size=image_size, num_classes=numParam), 'crossvit': VisionTransformer(img_size=(image_size, image_size)),
                 'cait': cait_models(img_size=image_size, num_classes=numParam), 'vmamba': vmamba.vanilla_vmamba_small(image_size)}

param_map = {"strike": 1, "opening": 5, "top": 7, "bottom": 9, "length": 11}
for model_weight in glob.glob(f'.{weights}/{network}/{learning_param}/{image_type}/*.pth'):
    model_no = os.path.split(model_weight)[-1]
    # excel file initilisation
    worksheet = workbook.add_worksheet(model_no)
    row = 0
    worksheet.write(row, 0, 'image')
    col = 1
    for param in learning_param:
        worksheet.write(row, col, f'original {param} label')
        col += 1
        worksheet.write(row, col, f'output {param} label')
        col += 1
        worksheet.write(row, col, f'output modded {param} label')
        col += 1
        worksheet.write(row, col, f'{param} difference')
        col += 1
    row += 1

    print("===>Testing using weights: ", model_weight)

    # setup the model using trained weights
    model = model_choices[network]
    checkpoint = torch.load(model_weight)
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
    all_difference = np.array()
    with torch.no_grad():
        for file in files:
            image = cv2.imread(file)

            # replace nan with mean values
            image = np.nan_to_num(image, nan=np.mean(image))

            # image = cv2.resize(image, (512, 512))
            image = np.divide(image, 255.0)
            filename = os.path.basename(file).split('/')[-1]

            image = np.float32(cv2.resize(image, (image_size, image_size)))
            image = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2)

            output = model(image)

            worksheet.write(row, 0, filename)
            # worksheet.write(row, 1, "".join(filename.split("_")[param_map[learning_param]]))
            col = 1
            image_diff = []
            for i, param in enumerate(learning_param):
                label = torch.tensor([float(filename.split("_")[param_map[param]])]).to(device)    
                worksheet.write(row, col, label)
                col += 1
                worksheet.write(row, col, output[i])
                col += 1
                output[i] = output[i] % 180
                output[i] = (output[i] + 180) % 180
                worksheet.write(row, col, output[i])
                col += 1

                difference = abs(output[i] - label)
                difference = min(difference, 180 - difference)
                difference = difference.cpu()

                worksheet.write(row, col, difference)
                col += 1
                image_diff.append(difference)

            all_difference = np.vstack(all_difference, image_diff)

            row += 1
    row = 2
    for i, param in enumerate(learning_param):
        worksheet.write(row, 6, f'Avg {param.capitalize()} Difference')
        worksheet.write(row, 7, np.mean(all_difference[:, i]))
        row += 1
        worksheet.write(row, 6, f'Min {param.capitalize()} Difference')
        worksheet.write(row, 7, np.min(all_difference[:, i]))
        row += 1
        worksheet.write(row, 6, f'Max {param.capitalize()} Difference')
        worksheet.write(row, 7, np.max(all_difference[:, i]))
        row += 2
workbook.close()
exit()
