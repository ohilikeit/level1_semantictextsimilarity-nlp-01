import os
import gdown
import torch
import streamlit as st
import yaml
from transformers import AutoTokenizer
import sys
from models import *
with open("streamlit_folder/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 구글 드라이브를 이용한 모델 다운로드
def download_model_file(url):
    output = "model.pt"
    gdown.download(url, output, quiet=False)

@st.cache_resource
def load_model():
    '''
    Return:
        model: 구글 드라이브에서 가져온 모델 return 
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("model.pt"):
        download_model_file(config['model_path'])
    
    model_path = 'model.pt'
    model = torch.load(model_path, map_location=device)

    return model

def get_prediction(model, sentence1: str, sentence2: str) -> float:
    '''
    Args:
        model: 학습된 모델
        sentence1: 유사도 측정 문장 1
        sentence2: 유사도 측정 문장 2

    Return:
        유사도 측정값
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    inputs = tokenizer(sentence1, sentence2, return_tensors="pt",
                       max_length=160, padding='max_length', truncation=True)['input_ids'].to(device)
    outputs = model(inputs)
    scalar_value = outputs.detach().cpu().item()

    return min(5., max(0., round(scalar_value,2)))
