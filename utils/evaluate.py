from tqdm import tqdm
import torch
import numpy as np
import os
import copy
import re
from sklearn.metrics import f1_score

label_dict={'browsing': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Email': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'facebook': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'FILE': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'MAIL': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'Skype': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'spotify': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'tor': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'VOIP': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'Youtube': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


def solve_input(image):
    image = image.astype(np.float32)
    image = image / 255
    image = torch.tensor(image)
    image = image.reshape((1,32,32))
    image.requires_grad = True
    return image

def evaluate_train(model, raw_path):
    acc = 0
    raw_label = []
    new_label = []
    count = 0
    patten = '[a-zA-Z]+'
    for file in tqdm(os.listdir(raw_path)):  # 对一张图片开始实验
        count = count+1
        label = re.match(patten, file).group()
        int_label = torch.tensor(label_dict[label]).argmax().item()
        raw_label.append(int_label)
        image_path = raw_path + '/' + file
        # load image
        raw_image = np.load(image_path)
        image = copy.deepcopy(raw_image)
        # preprocess input
        image = solve_input(image)
        predict = model(image).argmax().item()
        new_label.append(predict)
        if(predict==int_label):
            acc = acc+1
    macro = f1_score(raw_label,new_label,average='macro')
    micro = f1_score(raw_label,new_label,average='micro')
    accuracy = acc/count
    print("f1 macro:",macro,"f1 micro:",micro,"accuracy:",accuracy)
    return macro,micro,accuracy

