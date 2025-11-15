import cv2
from ultralytics import YOLO
from collections import defaultdict, Counter
import time
import numpy as np

import torch
from torchvision.models import mobilenet_v2
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myYOLO:
    def __init__(self, pretrained="real_last_final.pt"):
        self.cap = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.detect_model = YOLO("best_ori.pt")
        self.detect_model = YOLO(pretrained)
        # self.results_list = []  

    def detect(self, frame):
        print('\nDetect...')
        outputs = self.detect_model.predict(
            source=frame,
            device=0,
            save=False)
        
        labels = outputs[0].boxes.cls
        scores = outputs[0].boxes.conf
        detection_result = {"AF": 0, "EF": 0, "AT" : 0, "ET": 0}
        class_1_present = False
        class_1_score = 0
        class_3_score = 0

        for i, class_id in enumerate(labels):
            score = scores[i]
            
            if class_id == 0:
                detection_result["AF"] += 1
            elif class_id == 2:
                detection_result["EF"] += 1
            elif class_id == 1:
                detection_result["AT"] += 1
                class_1_present = True
                class_1_score = max(class_1_score, score)
            elif class_id == 3:
                detection_result["ET"] += 1
                class_3_score = max(class_3_score, score)

        if class_1_present:
            if class_1_score > class_3_score:
                detection_result["ET"] = 0

        return detection_result


    def identify(self, cropped_object):
        resized = cv2.resize(cropped_object, (224, 224))  
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        device = next(self.iden_model.parameters()).device 
        tensor = tensor.to(device)

        outputs = self.iden_model(tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    
    
    def print_results(self):
        return self.results_list[1:]  
       