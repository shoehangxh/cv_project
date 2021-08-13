import sys
import math
import torch

__all__ = ['bboxIOU', 'encodeBox', 'decodeAllBox', 'doNMS']
def doNMS(config, classMap, allBoxes, threshold):
    winBoxes = []
    predBoxes = config.predBoxes
    for c in range(1, config.classNumber):
        fscore = claclassMap[:, c]
        v,s = torch.sort(fscore, 0, descending=True)
        pritn('>>>>>>>', c, v[0])
        for i in range(len(v)):
            if(v[i] < threshold):
                continue
            k = s[i]
            boxA = [allBoxes[k, 0], allBoxes[k, 1], allBoxes[k, 2], allBoxes[k, 3]]
            for j in range(i+1, len(v)):
                if (v[j] < threshold):
                    continue
                k = s[j]
                boxB = [allBoxes[k, 0], allBoxes[k, 1], allBoxes[k, 2], allBoxes[k, 3]]
                iouValue = bboxIOU(boxA, boxB)
                if (iouValue > 0.5):
                    v[j] = 0
        for i in range(len(v)):
            if(v[i] < threshold):
                continue
            k = s[i]
            box = [allBoxes[k, 0], allBoxes[k, 1], allBoxes[k, 2], allBoxes[k, 3]]
            winBoxes.append(box)
        return winBoxes