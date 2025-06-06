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
    pos = []
    for tok in toks:
        pos.append(pos_map[tok.tag_])
    return pos


def get_one_sided_type(e):
    
    return "VRU"

def get_two_sided_type(e,g):
    return "ND"