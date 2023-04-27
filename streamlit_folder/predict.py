import os
import gdown
import torch
import streamlit as st
from streamlit_folder.model import MySTSModel
import yaml
from transformers import AutoTokenizer
import sys
with open("streamlit_folder/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
sys.path.insert(0, config['model_path'])
sys.path.insert(0, config['home_path'])

# Function to download the model file
def download_model_file(url):
    output = "model.pt"
    gdown.download(url, output, quiet=False)

@st.cache_resource
def load_model() -> MySTSModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("model.pt"):
        download_model_file('https://drive.google.com/uc?id=1ol_Pd3LDmTcG04h7VGHYkTD4TpgUlVk1')
    
    model_path = 'model.pt'
    model = torch.load(model_path, map_location=device)

    return model

def get_prediction(model, sentence1, sentence2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    inputs = tokenizer(sentence1, sentence2, return_tensors="pt",
                       max_length=160, padding='max_length', truncation=True)['input_ids'].to(device)
    outputs = model(inputs)
    scalar_value = outputs.detach().cpu().item()

    return min(5., max(0., round(scalar_value,2)))
