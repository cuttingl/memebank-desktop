from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import streamlit as st
import numpy as np
import requests


def image_read(url_image):
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(url_image)

    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    
    return logits, bboxes

if __name__ == '__main__':
    url_image = "./images/image.jpg"
    bboxes,logits = image_read(url_image)
    # print(logits, bboxes)
    st.image(url_image, use_column_width=True)
    st.write(bboxes)
    