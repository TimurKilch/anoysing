# import glob
# import numpy as np
# import time
# import cv2
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# from utilities.customUtils import *
# from dataTools.dataNormalization import *
# from dataTools.customTransform import *
# import os
# class customDatasetReader(Dataset):
#     def __init__(self, image_list, imagePath, height, width, transformation=True):
#         self.image_list = image_list
#         self.imagePath = imagePath
#         self.transformLR = transforms
#         self.imageH = height
#         self.imageW = width
#         self.normalize = transforms.Normalize(normMean, normStd)
#         self.var = 0.1
#         self.mean = 0.0
#         self.pov = 0.3


#     def __len__(self):
#         return (len(self.image_list))
    
#     def __getitem__(self, i):



#         self.transformHRGT = transforms.Compose([
#                                                 transforms.ToTensor(),
#                                                 self.normalize,
#                                                 ])
        
#         self.gtImageFileName = self.imagePath + extractFileName(self.image_list[i])
#         self.gtImage = Image.open(self.gtImageFileName)

#         # Transforms Images for training 
#         self.gtImageHR = self.transformHRGT(self.gtImage)
        
#         #Noise Modeling
#         sigma = random.uniform(0, self.var ** self.pov)
#         noiseModel = torch.clamp(torch.randn(self.gtImageHR.size()).uniform_(0, 1.) * sigma  + 0., 0., 1.)
    
#         self.inputImage = self.gtImageHR + noiseModel
        
#         return self.inputImage, noiseModel


import glob
import numpy as np
import time
import cv2
import random
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os
import pydicom

class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePath, height, width, transformation=True):
        self.image_list = image_list
        self.imagePath = imagePath
        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        self.normalize = transforms.Normalize(normMean, normStd)
        self.var = 0.1
        self.mean = 0.0
        self.pov = 0.3

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):
        self.transformHRGT = transforms.Compose([
                                                transforms.ToTensor(),
                                                self.normalize,
                                                ])
        
        self.gtImageFileName = self.imagePath + extractFileName(self.image_list[i])
        
        # Определяем, является ли файл DICOM
        if self.gtImageFileName.lower().endswith('.dcm'):
            print('ok')
            # Чтение DICOM файла
            dicom_file = pydicom.dcmread(self.gtImageFileName)
            # Преобразование DICOM в массив numpy (предполагается изображение в градациях серого)
            image_array = dicom_file.pixel_array
            # Конвертация в PIL Image для дальнейших преобразований
            tmp = Image.fromarray(image_array).convert('RGB')
            tmp.save('tmp.png', 'PNG')
            self.gtImage = Image.open('tmp.png')
        else:
            # Для не-DICOM файлов используется стандартная обработка
            self.gtImage = Image.open(self.gtImageFileName)

        # Преобразование изображений для обучения
        self.gtImageHR = self.transformHRGT(self.gtImage)
        
        # Моделирование шума
        sigma = random.uniform(0, self.var ** self.pov)
        noiseModel = torch.clamp(torch.randn(self.gtImageHR.size()).uniform_(0, 1.) * sigma  + 0., 0., 1.)
    
        self.inputImage = self.gtImageHR + noiseModel
        
        return self.inputImage, noiseModel
