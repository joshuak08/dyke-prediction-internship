import argparse
import copy
import os
from tqdm import tqdm
from volcano_dataset import Volcano_dataset, RandomCrop, CannyEdgeDetection, Resize
import logging
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset 
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import datetime as dt
from loss_functions import L1AngularLoss, L2AngularLoss
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from models.swin_transformer_v2 import SwinTransformerV2
from models.convnext import ConvNeXt
from models.crossvit import VisionTransformer
from models.deit import vit_models
from models.cait import cait_models
import models.vmamba as vmamba

parser = argparse.ArgumentParser(description="Main Training Code")
parser.add_argument('--network', type=str, default='convnext', 
                    choices=["convnext", "swin", "deit", "crossvit", "cait", "vmamba"], 
                    required=True, help='select model architecture')
parser.add_argument('--learningParam', type=str, default='strike', choices=["strike", "opening", "top", "bottom", "length"])
parser.add_argument('--maxepoch', type=int, default=200, help='number of epochs to train. default: 201 to save epoch 200')
parser.add_argument('--savemodel_epoch', type=int, default=10, help='save model every _ epochs. default: 10 (save model every 10 epochs)')
parser.add_argument('--batchSize', type=int, default=8, help="batch number (default: 8)")
parser.add_argument('--imageSize', type=int, default=512)
parser.add_argument("--lossFunction", type=str, default="L1", choices=["L1", "L2"])
parser.add_argument ("--lr", type=float, default=3e-5)
parser.add_argument("--imageType", type=str, default='DST_Coherence')
parser.add_argument("--outputDir", type=str, default='/output')
parser.add_argument("--inputDir", type=str, default="/Dyke")
parser.add_argument("--edgeDetection", type=bool, default=False)


args = parser.parse_args()

network = args.network
learningParam = args.learningParam
max_epoch = args.maxepoch
save_model_epoch = args.savemodel_epoch
batch_size = args.batchSize
image_size = args.imageSize
chosen_loss = args.lossFunction
lr = args.lr
image_type = args.imageType
outputDir = args.outputDir
inputDir = args.inputDir
edgeDetection = args.edgeDetection

base_dir = os.getcwd() + f"{inputDir}/{image_type}/"

model_choices = {"swin": SwinTransformerV2(img_size=image_size), 'convnext': ConvNeXt(), 
                 'deit': vit_models(img_size=image_size), 'crossvit': VisionTransformer(img_size=(image_size, image_size)),
                 'cait': cait_models(img_size=image_size), 'vmamba': vmamba.vanilla_vmamba_small(image_size)}

model = model_choices[network]

# setup log file
LOG_FILE = os.getcwd() + f"/logs/{network}"
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)
LOG_FILE = LOG_FILE + "/" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d %H_%M_%S') + f' Type_{image_type}' + ".log"
logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
fileHandler = logging.FileHandler("{0}".format(LOG_FILE))
fileHandler.setFormatter(logFormatter)
rootLogger = logging.getLogger()
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
# logs all different arguments
logging.info(str(args))

# transformations resizes images to fit the image size for memory issues
train_transform = [Resize([image_size, image_size])]
val_transform = [Resize([image_size, image_size])]

if edgeDetection:
    train_transform.append(CannyEdgeDetection())
    val_transform.append(CannyEdgeDetection())

train_transform.append(transforms.ToTensor())
val_transform.append(transforms.ToTensor())

# -----------------------------------------------------------------------------------------------------------------------
# setup dataset in tensor form
db_train = Volcano_dataset(data_dir=base_dir, learning_param=learningParam, partition='train', transform=transforms.Compose(train_transform))
db_val = Volcano_dataset(data_dir=base_dir, learning_param=learningParam, partition='val', transform=transforms.Compose(val_transform))

# Split data into train/validation set
sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
indices = range(len(db_train))
for train_index, val_index in sss.split(indices):
    pass

train_ds = Subset(db_train, train_index)
val_ds = Subset(db_val, val_index)

# loads the data with corresponding batch size and partitions
trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

max_iterations = max_epoch * len(trainloader) 
logging.info("{} iterations per epoch. {} max iterations \n".format(len(trainloader), max_iterations))

loss_history = {
    "train": [],
    "val": []
}

best_val_loss = float('inf')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# add different loss functions to this dictionary
loss_choice = {'L1': L1AngularLoss(), 'L2': L2AngularLoss()}

loss_function = loss_choice[chosen_loss]

optimizer = optim.Adam(model.parameters(), lr=lr)

logging.info(f"This training done with {device}")

iterator = tqdm(range(max_epoch+1), ncols=100)
for epoch in iterator:
    print()
    # training phase
    model.train()
    training_loss = 0.0

    for i, batch in enumerate(trainloader):
        image, label = batch["image"], batch["label"]
        image, label = image.to(device=device, dtype=torch.float), label.to(device=device, dtype=torch.float)
        
        optimizer.zero_grad()
        output = model(image)
        
        loss = loss_function(output, label)
        loss.mean().backward()
        optimizer.step()
        training_loss += loss.mean().item()


    epoch_train_loss = training_loss / len(trainloader)
    logging.info(f"Epoch: {epoch + 1}, Train Loss: {epoch_train_loss:.4f}")
    loss_history["train"].append(epoch_train_loss)

    # validation phase
    model.eval()
    validation_loss = 0.0
    for i, batch in enumerate(valloader):
        image, label = batch["image"], batch["label"]
        image, label = image.to(device=device, dtype=torch.float), label.to(device=device, dtype=torch.float)

        val_output = model(image)

        loss = loss_function(val_output, label)

        validation_loss += loss.mean().item()

    epoch_val_loss = validation_loss / len(valloader)
    logging.info(f"Epoch: {epoch + 1}, Validation Loss: {epoch_val_loss:.4f}")
    loss_history["val"].append(epoch_val_loss)

    # save model every x epochs as chosen
    if epoch % save_model_epoch == 0:
        if not os.path.exists(os.getcwd() + f"{outputDir}/{network}/{chosen_loss}/{learningParam}/{image_type}"):
            os.makedirs(os.getcwd() + f"{outputDir}/{network}/{chosen_loss}/{learningParam}/{image_type}", exist_ok=True, mode=0o777)

        torch.save(model.state_dict(), os.path.join(os.getcwd() + f"{outputDir}/{network}/{chosen_loss}/{learningParam}/{image_type}", f"ep_{epoch}.pth"))

    # save model when new best model found based on validation loss
    if epoch_val_loss < best_val_loss:
        if not os.path.exists(os.getcwd() + f"{outputDir}/{network}/{chosen_loss}/{learningParam}/{image_type}"):
            os.makedirs(os.getcwd() + f"{outputDir}/{network}/{chosen_loss}/{learningParam}/{image_type}", exist_ok=True, mode=0o777)
        best_val_loss = epoch_val_loss
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, os.path.join(os.getcwd() + f"{outputDir}/{network}/{chosen_loss}/{learningParam}/{image_type}", "best_model.pth"))
