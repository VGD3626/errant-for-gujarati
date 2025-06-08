import json 
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path
from .ModelClasses import MorphAnalysis, CustomTokenClassificationModel, PosMorphClassificationModel
import sys

sys.modules["__main__"].PosMorphClassificationModel = PosMorphClassificationModel
sys.modules["__main__"].CustomTokenClassificationModel = CustomTokenClassificationModel


def load_feature_values(path, encoding="utf-8"):
    with open(path) as mappings:
        mapping_dict = json.load(mappings)        
    return mapping_dict

NA='NA'
MAX_LENGTH=120

base_dir = Path(__file__).resolve().parent

feature_values=load_feature_values(base_dir/"features.json")
feature_seq=list(feature_values.keys())
EXTRA_TOKEN=[-100]*len(feature_seq)


total_number_of_features=0
feature_value2id={}
feature_id2value={}
feature_start_range={}

start_range=0
for key,values in feature_values.items():
  feature_value2id[key]={}
  feature_start_range[key]=start_range
  for i,value in enumerate(values):
    feature_value2id[key][value]=i+start_range
  feature_id2value[key]={(y-start_range):x for x,y in feature_value2id[key].items()}
  start_range+=len(values)
  total_number_of_features+=len(values)
number_of_labels=total_number_of_features

base_model = CustomTokenClassificationModel(
    bert_model=AutoModelForTokenClassification.from_pretrained("l3cube-pune/gujarati-bert"),
    feature_seq=feature_seq
)
inference_model = PosMorphClassificationModel(base_model, feature_seq)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_checkpoint = "/content/drive/MyDrive/NLP Gujarati POS & Morph Analysis/POS_MORPH_MODEL/trained_model_binary_file/GUJ_POS_MORPH_ANAYLISIS_WRAPPER-v6.0-model.pth"
inference_model=torch.load(inference_checkpoint, map_location=device, weights_only=False)
inference_model.eval()
inference_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/gujarati-bert")
model = MorphAnalysis(tokenizer, inference_model, feature_seq, feature_id2value, MAX_LENGTH,NA)

