import json 
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from pathlib import Path

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

class CustomTokenClassificationModel(nn.Module):
    def __init__(self, bert_model, feature_seq):
        super(CustomTokenClassificationModel, self).__init__()
        self.bert_model = bert_model

        self.module_list = nn.ModuleList()

        for key in feature_seq:
          num_classes = len(feature_values[key])
          module = nn.Linear(number_of_labels, num_classes)
          self.module_list.append(module)

    def forward(self, input_ids, attention_mask):
        # 1) Run BERT to get last_hidden_state (shape: [B, L, 768])
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.logits
        
        # 2) Run each head on the 768‐dim embeddings
        logits_list = []
        for head in self.module_list:
            logits = head(hidden_states)
            # logits has shape [batch_size, seq_len, num_classes_for_that_feature]
            logits_list.append(logits)

        # 3) Return the list of all feature‐head logits
        return logits_list

class PosMorphClassificationModel(nn.Module):
    def __init__(self, custom_model, feature_seq):
        super(PosMorphClassificationModel, self).__init__()
        self.custom_model = custom_model

    def forward(self, input_ids, attention_mask):
        return self.custom_model(
              input_ids,
              attention_mask=attention_mask
        )

class MorphAnalysis:
    def __init__(self, tokenizer, inference_model, feature_seq, feature_id2value, max_length,NA):
        self.tokenizer = tokenizer
        self.inference_model = inference_model
        self.feature_seq = feature_seq
        self.feature_id2value = feature_id2value
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NA = NA

    def prepare_mask(self, word_ids):
        mask = []
        last = None
        for i in word_ids:
            if i is None or i == last:
                mask.append(0)
            else:
                mask.append(1)
            last = i
        return mask

    def tokenize_sentence(self, sentence, splitted=False):
        if not splitted:
            tokens = sentence.split(' ')
        else:
            tokens = sentence

        tokenized_inputs = self.tokenizer(
            tokens,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
        )
        mask = self.prepare_mask(tokenized_inputs.word_ids(0))
        sample = {
            "tokens": tokens,
            "mask": mask,
            "input_ids": tokenized_inputs['input_ids'],
            "attention_mask": tokenized_inputs['attention_mask']
        }
        return sample

    def prepare_output(self, sample):
        tokens = sample['tokens']
        output = []

        for i, token in enumerate(tokens):
          features = {}
          for feat in self.feature_seq:
            feat_val = sample[feat][i]
            if feat_val != self.NA:
              features[feat] = feat_val
          output.append((token, features))
        return output

    def infer(self, sentence):
        batch = self.tokenize_sentence(sentence)

        input_ids = torch.tensor([batch["input_ids"]]).to(self.device)
        attention_mask = torch.tensor([batch["attention_mask"]]).to(self.device)
        mask = torch.tensor([batch['mask']]).to(self.device)

        logits_list = self.inference_model(input_ids, attention_mask=attention_mask)

        curr_sample = {
            "tokens": batch["tokens"],
        }

        for i, logits in enumerate(logits_list):
            key = self.feature_seq[i]
            curr_mask = (mask != 0)
            valid_logits = logits[curr_mask]
            probabilities = F.softmax(valid_logits, dim=-1)
            valid_predicted_labels = torch.argmax(probabilities, dim=-1)
            curr_id2value_map = self.feature_id2value[key]
            curr_sample[key] = [curr_id2value_map[x] for x in valid_predicted_labels.tolist()]


        output = self.prepare_output(curr_sample)
        return output
