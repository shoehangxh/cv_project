import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from bbox import bboxIOU
__all__ = ['buildPredBoxes', 'sampleEzDetect']
def buildPredBoxes(config):
    predBoxes = []
    for i in range(len(config.mboxes)):
        l = config.mboxes[i][0]
        wid = config.featureSize[l][0]
        hei = conconfig.featureSize[l][1]
        wbox = config.mboxes[i][1]
        hbox = config.mboxes[i][2]
        for y in range(hei):
            for x in range(wid):
                xc = (x + 0.5) / wid #x,y位置都取每个feature map像素点的中心值来计算
                yc = (y + 0.5) / hei
                xmin = xc - wbox / 2
                ymin = yc - hbox / 2
                xman = xc + wbox / 2
                ymax = yc + hbox / 2
                predBoxes.append([xmin, ymin, xmax, ymax])
    return predBoxes

def sampleEzDetect(config, bboxes):#在voc_dataset.py的vocDataset类中用到的sampleEZDetect函数
    #preparing pred boxes
    predBoxes = config.predBoxes
    #preparing ground truth
    truthBoxes = []
    for i in range(len(bboxes)):
        truthBoxes.append([bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]])
    #computing iou
    iouMatrix = []
    for i in predBoxes:
        ious = []
        for j in truthBoxes:
            ious.append(bboxIOU(i, j))
        iouMatrix.append(ious)
    iouMatrix = torch.FloatTensor(iouMatrix)
    iouMatrix2 = iouMatrix.clone()
    ii = 0
    selectedSamples = torch.FloatTensor(128*1024)
    #positive samples from bi-direction match
    for i in range(len(bboxes)):
        iouViewer = iouMatrix.view(-1)
        iouValues, iouSequence = torch.max(iouViewer, 0)
        predIndex = iouSequence[0] // len(bboxes)
        bboxIndex = iouSequence[0] % len(bboxes)
        if (iouValues[0] > 0.1):
            selectedSamples[ii * 6 + 1] = bboxes[bbboxIndex][0]
            selectedSamples[ii * 6 + 2] = bboxes[bbboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bbboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bbboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bbboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex
            ii = ii + 1
        else:
            break
        iouMatrix[:, bboxIndex] = -1
        iouMatrix[predIndex, :] = -1
        iouMatrix2[predIndex, :] = -1
    #also samples with high iou
    for i in range(len(selectedSamples)):
        v, _ = iouMatrix2[i].max(0)
        predIndex = i
        bboxIndex = _[0]
        if(v[0] > 0.7):#anchor与真实值IOU大于0.7的为正样本
            selectedSamples[ii * 6 + 1] = bboxes[bbboxIndex][0]
            selectedSamples[ii * 6 + 2] = bboxes[bbboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bbboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bbboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bbboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex
            ii = ii + 1
        elif (v[0] > 0.5):
            selectedSamples[ii * 6 + 1] = bboxes[bbboxIndex][0] * (-1)
            selectedSamples[ii * 6 + 2] = bboxes[bbboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bbboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bbboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bbboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex
            ii = ii + 1
    selectedSamples[0] = ii
    return selectedSamples

def encodeBox(config, box, predBox):
    pcx = (predBox[0] + predBox[2]) / 2
    pcy = (predBox[1] + predBox[3]) / 2
    pw = (predBox[2] - predBox[0])
    ph = (predBox[3] - predBox[1])

    ecx = (box[0] + box[2]) / 2 - pcx
    ecy = (box[1] + box[3]) / 2 - pcy
    ecx = ecx / pw * 10
    ecy = ecy / ph * 10
    ew = (box[2] - box[0]) / pw
    eh = (box[3] - box[1]) / ph
    ew = math.log(ew) * 5
    eh = math.log(eh) * 5
    return [ecx, ecy, ew, eh]

def decodeAllBox(config, allBox):
    newBoxes = torch.FloatTensor(allBox.size())
    batchSize = newBoxes.size()[0]
    for k in range(len(config.predBoxes)):
        predBox = config.predBoxes[k]
        pcx = (predBox[0] + predBox[2]) / 2
        pcy = (predBox[1] + predBox[3]) / 2
        pw = (predBox[2] - predBox[0])
        ph = (predBox[3] - predBox[1])
        for i in range(batchSize):
            box = allBox[i, k, :]
            dcx = box[0] / 10 * pw + pcx
            dcy = box[0] / 10 * ph + pcy
            dw = math.exp(box[2] / 5) * pw
            dh = math.exp(box[3] / 5) * ph
            newBoxes[i, k, 0] = max(0, dcx - dw / 2)
            newBoxes[i, k, 1] = max(0, dcy - dh / 2)
            newBoxes[i, k, 2] = min(1, dcx + dw / 2)
            newBoxes[i, k, 3] = min(1, dcy + dh / 2)
    if config.gpu:
        newBoxes = newBoxes.cuda()
    return newBoxes