from pathlib import Path
import random
import json

random.seed(42)

ROOT = Path("data/raw/tusimple/tusimple_preprocessed/training")

IMAGE_DIR = None
MASK_DIR = None

for folder in ROOT.iterdir():
    if folder.is_dir():
        if "mask" in folder.name.lower():
            MASK_DIR = folder
        else:
            IMAGE_DIR = folder

images = sorted(list(IMAGE_DIR.glob("*")))
masks = sorted(list(MASK_DIR.glob("*")))

image_dict = {x.stem: x for x in images}
mask_dict = {x.stem: x for x in masks}

pairs = []

for name in sorted(set(image_dict) & set(mask_dict)):
    pairs.append({
        "image": str(image_dict[name]),
        "mask": str(mask_dict[name])
    })

random.shuffle(pairs)

N = len(pairs)

train = pairs[:int(0.8*N)]
val = pairs[int(0.8*N):int(0.9*N)]
test = pairs[int(0.9*N):]

Path("data/splits").mkdir(parents=True,exist_ok=True)

json.dump(train,open("data/splits/train.json","w"),indent=2)
json.dump(val,open("data/splits/val.json","w"),indent=2)
json.dump(test,open("data/splits/test.json","w"),indent=2)

print("Train:",len(train))
print("Val:",len(val))
print("Test:",len(test))