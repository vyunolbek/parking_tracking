from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import cv2 as cv
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import torch

parser = argparse.ArgumentParser(description='')
parser.add_argument('--source', dest='path_source', default='./img/rrr.jpg')
parser.add_argument('--labeled', dest='path_labeled', default='./labeled/rrr.xml')
args = parser.parse_args()

img = cv.imread(args.path_source)
img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

tree = ET.parse(args.path_labeled)

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.conf = 0.1
model.multi_label = True
model.iou = 0.3
model.classes = [2, 7, 5]

root = tree.getroot()
a = []
b = []
for elem in root:
    for subelem in elem:
            for sub in subelem:
                b.append(int(sub.text) // 2)

for i in range(0, len(b), 4):
    a.append(b[i:i+4])


prediction = model(img)
print([[float(prediction.xyxy[0][i][4]), model.names[int(prediction.xyxy[0][i][-1])], int(prediction.xyxy[0][i][-1])] for i in range(len(prediction.xyxy[0]))])

cnt = 0
used = []
for place in a:
    flag = True
    for rect in range(len(prediction.xyxy[0])):
        box = list(map(int, prediction.xyxy[0][rect]))
        dot = [(box[0] + (box[2] - box[0]) // 2),  (box[1] - (box[1] - box[3]) // 2)]
        if (dot[0] >= place[0] and dot[1] >= place[1] and dot[0] <= place[2] and dot[1] <= place[3]):
            cv.circle(img, (dot[0],  dot[1]), radius=5, thickness=4, color=(0, 200, 0))
            flag = False
    if flag:
        if place not in used:
            used.append(place)
            cv.rectangle(img, (place[0], place[1]), (place[2], place[3]), color=(0, 0, 255), thickness=2)
            cnt += 1

cv.putText(img, f'Свободно {cnt} мест из {len(a)}', (img.shape[1] - 500, img.shape[0] - 15), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1, color=(100, 255, 100), thickness=3)
image = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
cv.imshow('frame', img)
cv.waitKey(0)