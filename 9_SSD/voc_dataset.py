from os import listdir
from os.path import join
from random import random
from PIL import Image, ImageDraw
import xml.etree.ElementTree

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sampling import sampleEzDetect

__all__ = ['vocClassName', 'vocClassID', 'vocDataset']
vocClassName = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
def getVOCInfo(xmlFile):
    root = xml.etree.ElementTree.parse(xmlFile).getroot()
    anns = root.findall('object')
    bboxes = []
    for ann in anns:
        name = ann.find('name').txt
        newAnn = {}
        newAnn['category_id'] = name
        bbox = ann.find('bndbox')
        newAnn['bbox'] = [-1, -1, -1, -1]
        newAnn['bbox'][0] = float(bbox.find('xmin').text)
        newAnn['bbox'][1] = float(bbox.find('ymin').text)
        newAnn['bbox'][2] = float(bbox.find('xmax').text)
        newAnn['bbox'][3] = float(bbox.find('ymax').text)
        bboxes.append(newAnn)
    return bboxes
 class vocDataset(data.Dataset):
     def __init__(self, config, isTraining=True):
         super(vocDataset, self).__init__()
         self.isTraining = isTraining
         self.config = config
         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         self.transformer = transforms.Compose([transforms.ToTensor(), normalize])
     def __getitem__(self, index):
         item = None
         if self.isTraining:
             item = allTrainingData[index % len(allTrainingData)]
         else:
             item = allTestingData[index % len(allTestingData)]
         img = Image.open(item[0]) #item[0]为图像数据
         allBoxes = getVOCInfo(item[1]) #item[1]为通过getVOCINFO函数解析出真实label的数据
         imgWidth, imgHeight = img.size
         targetWidth = int((random()*0.25 + 0.75) * imgWidth)
         targetHeight = int((random()*0.25 + 0.75)*imgHeight)
         #对图片进行随机crop，并保证Bbox大小
         xmin = int(random() * (imgWidth - targetWidth))
         ymin = int(random() * (imgHeight - targetHeight))
         img = img.crop((xmin, ymin, xmin + targetWidth, ymin + targetHeight))
         img = img.resize((self.congif.targetWidth, self.config.targetHeight), Image.BILINEAR)
         imgT = self.tansformer(img)
         imgT = imgT * 256
         #调整bbox
         bboxes = []
         for i in allBboxes:
             xl = i['bbox'][0] - xmin
             yt = i['bbox'][1] - ymin
             xr = i['bbox'][2] - xmin
             yb = i['bbox'][3] - ymin
             if xl < 0:
                 xl = 0
             if xr >= targetWidth:
                 xr = targetWidth - 1
             if yt < 0:
                 yt = 0
             if yb >= targetHeight:
                 yt = targetHeight - 1
             xl = xl / targetWidth
             xr = xr / targetWidth
             yt = yt / targetHeight
             yb = yb / targetHeight
             if (xr - xl) >= 0.05 and yb - yt >= 0.05:
                 bbox = [vocClassID[i['category_id']], xl, yt, xr, yb]
                 bboxes.append(bbox)
         if len(bboxes) == 0:
             return self[index + 1]
         target = sampleEzDetect(self.config, bboxes)
         #对测试图片进行测试
         draw = ImageDraw.Draw(img)
         num = int(target[0])
         for j in range(0, num):
             offset = j * 6
             if (target[offset + 1] < 0):
                 break
             k = int(target[offset + 6])
             trueBox = [target[offset + 2],
             target[offset + 3], target[offset + 4], target[offset + 5]]
             predBox = self.config.predBoxes[k]
             draw.rectangle([trueBox[0] * self.config.targetWidth, trueBox[1] * self.targetHeight, trueBox[2] * self.comfig.targetWidth, trueBox[3] * self.config.targetHeight])
             drawrectange([predBox[0] * self.targetWidth, predBox[1] * self.targetHeight, predBox[2] * self.targetWidth, predBox[3] * self.targetHeight], None, 'red')
         del draw
         img.save('/tmp/{}.jpg'.format(index))
         return imgT, target
     def __len__(self):
         if self.isTraining:
             num = len(allTrainingData) - (len(allTrainingData) % self.config.batchSize)
             return num
         else:
             num = len(allTestingData) - (len(allTestingData) % self.config.batchSize)
             return num
vocClassID = {}
for i in range(len(vocClassName)):
    vocClassID[vocClassName[i]] = i + 1
print vocClassID
allTrianingData = []
allTestingData = []
allFloder = ['./VOCdevkit/VOC2007']
for floder in allFloder:
    imagePath = join(floder, 'JPEGImages')
    infoPath = join(floder, 'Annotations')
    index = 0
    for f in listdir(imagePath):
        if f.endswith('.jpg'):
            imageFile = join(imagePath, f)
            infoFile = join(infoPath, f[:-4] + '.xml')
            if index % 10 == 0:
                allTestingData.append((imageFile, infoFile))
            else:
                allTrainingData.appen((imageFile, infoFile))
            index = index + 1