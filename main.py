import streamlit as st
import PIL
from PIL import Image, ImageOps
import cv2
import os
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2 
from albumentations import Compose
from torchvision.ops import nms
import gdown 
# import torchvision.datasets.utils as utils

st.experimental_memo.clear()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# url = https://drive.google.com/file/d/1XiaNFXISnfVMmbvRGlTxFKVLV6l5-fZy/view?usp=sharing
# gdown.download(url, '/app/lung_nodule_detection/weight_path', quiet=False)


# url = "https://drive.google.com/uc?export=download&id=1XiaNFXISnfVMmbvRGlTxFKVLV6l5-fZy"

# @st.cache
# def download_weights(url):

# #     utils.download_url(url, 'weight_path')
#     gdown.download(url,'weight_path', quiet=False)

@st.cache
def download_weights(url):
    try:
        gdown.download(url, 'weight_path', quiet=False)
        st.write("File successfully downloaded to 'weight_path'")
    except Exception as e:
        print(f"Failed to download file: {e}")

   
# @st.cache
# def load_model():
#  print(" MODEL LOADED !!!")
#  return torch.load('weight_path',map_location=device)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
url = "https://drive.google.com/uc?export=download&id=1XiaNFXISnfVMmbvRGlTxFKVLV6l5-fZy"

download_weights(url)
# model = load_model()
model = torch.load('weight_path',map_location=device)

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
    transform = Compose([ToTensorV2 ()])
    imageTensor = transform(image=image_arr)['image']
    imageTensor = imageTensor.to(device) 
    model.eval()
    outputs = model([imageTensor])
    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    keep = nms (boxes=boxes,scores=scores,iou_threshold=0.3)
    scores =scores.detach().cpu().numpy()
    print(outputs)
    

    
    if len(boxes.detach().cpu().numpy())!=0:
        image_arr =np.stack((image_arr,image_arr,image_arr),axis=2)
        image_arr =np.squeeze(image_arr)

        for i in range(len(boxes[keep])):
            cv2.rectangle(image_arr, (int  (boxes[keep][i][0] )  ,   int (boxes[keep][i][1])  ), (int  (boxes[keep][i][2] )  ,   int (boxes[keep][i][3])   ),colors[i],1)
            h, w, _ = image_arr.shape
            font_scale = min(1,max(3,int(w/500)))
            font_thickness = min(2, max(10,int(w/50)))
            p1, p2 = (int  (boxes[keep][i][0] ), int  (boxes[keep][i][1] )), (int  (boxes[keep][i][2] ), int  (boxes[keep][i][3] ))
            tw, th = cv2.getTextSize(
                str('{:.3}'.format(scores[i])), 
                0, fontScale=font_scale, thickness=font_thickness
            )[0]
            p2 = p1[0] + tw, p1[1] + -th - 10
            cv2.rectangle(image_arr, p1,p2 ,colors[i],-1)
            cv2.putText(img = image_arr,text = str('{:.3}'.format(scores[i])),org = (int  (boxes[keep][i][0]),int  (boxes[keep][i][1])-5),fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=font_scale,color =(255,255,255),thickness =font_thickness)
        placeholder.image(image_arr, clamp=True)

    elif len(boxes.detach().cpu().numpy())==0 :
        st.write('"NO NODULE DETECTED !!!"')

    
