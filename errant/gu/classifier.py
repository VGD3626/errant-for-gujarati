from pathlib import Path
import json
from errant.gu.gu_nlp_pipeline import nlp_gu

# Load Hunspell word list for Gujarati
def load_word_list(path):
    with open(path, encoding="utf-8") as word_list:
        return set([word.strip() for word in word_list])
    
# Load mappings for POS tags
def load_pos_map(path, encoding="utf-8"):
    with open(path) as mappings:
        mapping_dict = json.load(mappings)        
    return mapping_dict

# Classifier resources
base_dir = Path(__file__).resolve().parent

# Spacy pipeline
nlp = nlp_gu

# Hunspell-gu dictionary 
vocab = load_word_list(base_dir/"resources"/"hunspell-gu-dict.txt")

#POS mappings
pos_map = load_pos_map(base_dir/"resources"/"pos_mapping.json")

# Common POS tags
coarse_pos = {"ADJ", "ADV", "NOUN", "VERB", "ADP", "PRON", "AUX"}

# Rare POS tags that make uninformative error categories
rare_pos = {"INTJ", "NUM", "SYM", "X"}

# Input: An Edit object
# Output: The same Edit object with an updated error type
def classify(edit):
    # Nothing to nothing is a detected but not corrected edit
    if not edit.o_toks and not edit.c_toks:
        edit.type = "UNK"
    
    elif not edit.o_toks and edit.c_toks:
        op = "M:"
        cat = get_one_sided_type(edit.c_toks)
        edit.type = op+cat

     # Unnecessary
    elif edit.o_toks and not edit.c_toks:
        op = "U:"
        cat = get_one_sided_type(edit.o_toks)
        edit.type = op+cat

    else:
        # Same to same is a detected but not corrected edit
        if edit.o_str == edit.c_str:
            edit.type = "UNK"

        # Replacement
        else:
            op = "R:"
            cat = get_two_sided_type(edit.o_toks, edit.c_toks)
            edit.type = op+cat

    return edit
    
def get_edit_info(toks):
    feat = []
    for tok in toks:
        feat.append(tok._.feat)
    return feat


def get_one_sided_type(toks):
    feat_list = get_edit_info(toks)
    pos_list = [pos_map[f.get("pos")] for f in feat_list]

    if len(pos_list) == 1 and pos_list[0] not in rare_pos:
        return pos_list[0]
    
    # infinitives and phrasal verbs
    if set(pos_list) == {"PART", "VERB"}:
        return "VERB"
    # Tricky cases
    else:
        return "OTHER"

def get_two_sided_type(e,g):
    return "VRU"