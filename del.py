import json
import os

train_image_names = os.listdir("stage1/train")
with open('data.json') as f:
    train_labels = json.load(f)
train_labels = train_labels['annotations']
for one_sample in train_labels:
	name = one_sample['name']
	num = one_sample['num']
	if num > 20:
		os.remove(name)
		print('delete one picture')
print('achieve')