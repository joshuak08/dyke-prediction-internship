import shutil
import os
import glob
import random
import json
import argparse

parser = argparse.ArgumentParser(description="Testing model")

parser.add_argument("--dir", type=str, default="./Dyke", help="directory containing all training & testing images")
parser.add_argument("--imageType", type=str, default="DST_Coherence")

args = parser.parse_args()

srcDir = args.dir
imageType = args.imageType

totalFiles = [file for file in glob.glob(srcDir + f"/{imageType}/" + "*.png")]

numOfTestingFiles = int(len(totalFiles)/10)

testFiles = set()

while len(testFiles) < numOfTestingFiles:
    testFiles.add(random.choice(totalFiles))

totalFiles = set(totalFiles)

trainFiles =  totalFiles - testFiles

json_data = {'train': list(trainFiles), "test": list(testFiles)}

with open(srcDir + "file_split.json", 'w') as f:
    json.dump(json_data, f, indent=2)

