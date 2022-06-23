import streamlit as st
import PIL
from PIL import Image, ImageOps
import cv2
import os
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensor
from albumentations import Compose
from torchvision.ops import nms
from torch.utils import model_zoo

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model_zoo.load_url ('https://drive.google.com/uc?export=download&id=1XiaNFXISnfVMmbvRGlTxFKVLV6l5-fZy',map_location=device)

colors =[(0,255,0),(255,0,0),(0,0,255),(255,255,255)]
st.title("Detection of Lung Nodules")
 
with st.container():
  bio_image= cv2.imread ('FORTH.png')
  bio_image = cv2.cvtColor(bio_image, cv2.COLOR_BGR2RGB)
  st.image(bio_image)

path_for_detection = st.file_uploader("Choose an image ! ",type=['png', 'jpg','jpeg'])

placeholder = st.empty()

if path_for_detection is not None:
    image_for_detection= PIL.Image.open(path_for_detection)
    im_gray = ImageOps.grayscale(image_for_detection)
    placeholder.image(im_gray)
        
if st.button("Press for Detection of Lung Nodules"):

    image_arr = np.array(im_gray).astype('float32')
    image_arr = np.expand_dims(image_arr,axis=2)
    image_arr /= 255.
    transform = Compose([ToTensor()])
    imageTensor = transform(image=image_arr)['image']
    imageTensor = imageTensor.to(device) 
    model.eval()

    outputs = model([imageTensor])
    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    keep = nms (boxes=boxes,scores=scores,iou_threshold=0.3)
    scores =scores.detach().cpu().numpy()
    if len(boxes.detach().cpu().numpy())!=0:
        image_arr =np.stack((image_arr,image_arr,image_arr),axis=2)
        image_arr =np.squeeze(image_arr)

        for i in range(len(boxes[keep])):
            cv2.rectangle(image_arr, (int  (boxes[keep][i][0] )  ,   int (boxes[keep][i][1])  ), (int  (boxes[keep][i][2] )  ,   int (boxes[keep][i][3])   ),colors[i],1)
            cv2.putText(img = image_arr,text = str('{:.3}'.format(scores[i])),org = (100+i*150,100),fontFace = cv2.cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color =colors[i],thickness =2)
            placeholder.image(image_arr)

    elif len(boxes.detach().cpu().numpy())==0 :
        st.write('"NO NODULE DETECTED !!!"')

    
