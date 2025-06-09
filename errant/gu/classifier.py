from pathlib import Path
import json
from errant.gu.gu_nlp_pipeline import nlp_gu
import Levenshtein
from errant.gu.GujaratiStemmer import Stemmer

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

# matras in Gujarati
matras = ['ા', 'િ', 'ી', 'ુ', 'ૂ', 'ે', 'ૈ', 'ો', 'ૌ', 'ૃ', 'ૄ', 'ૅ', 'ૉ']

#stemmer for Gujarati
stemmer = Stemmer()

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
    # print(feat_list)
    # if not feat_list:
    pos_list = []
    pos_list = [pos_map[f.get("pos", "NA")] for f in feat_list]

    if len(pos_list) == 1 and pos_list[0] not in rare_pos:
        return pos_list[0]
    
    # infinitives and phrasal verbs
    if set(pos_list) == {"PART", "VERB"}:
        return "VERB"
    # Tricky cases
    else:
        return "OTHER"

def get_two_sided_type(o_toks,c_toks):
        
    cat = ""
    # Orthography; i.e. whitespace errors
    if is_only_orth_change(o_toks, c_toks): 
        return "ORTH"
    
    # Word Order; only matches exact reordering.
    if is_exact_reordering(o_toks, c_toks):
        return "WO"


    # 1:1 replcement- very common
    if len(o_toks) == len(c_toks) == 1:
        o_tok = o_toks[0]
        c_tok = c_toks[0]

         # Spelling (A case of 1:1 replacement)
        if is_spelling_special_case(o_tok.text, c_tok.text):
            return "SPELL"

        if o_tok.text not in vocab:
            char_ratio = Levenshtein.ratio(o_tok.text, c_tok.text)
            char_dist = Levenshtein.distance(o_tok.text, c_tok.text)

            # Ratio > 0.5 means both correction and input share at least half the same chars.
            # WARNING: THIS IS AN APPROXIMATION.
            if char_ratio > 0.5 or char_dist == 1:
              cat = "SPELL"
              if mismatched_are_matras_only(o_tok.text, c_tok.text):
                cat += ":MATRA"
                return cat
              if mismatched_is_anusvara_only(o_tok.text, c_tok.text):
                cat += ":ANUSVARA"
                return cat
            # If ratio is <= 0.5, the error is more complex e.g. tolk -> say
            else:
                return "OTHER"
            
       

         # Spelling (A case of 1:1 replacement)
        if is_spelling_special_case(o_tok.text, c_tok.text):
            return "SPELL"

        if o_tok.text not in vocab:
            char_ratio = Levenshtein.ratio(o_tok.text, c_tok.text)
            char_dist = Levenshtein.distance(o_tok.text, c_tok.text)

            # Ratio > 0.5 means both correction and input share at least half the same chars.
            # WARNING: THIS IS AN APPROXIMATION.
            if char_ratio > 0.5 or char_dist == 1:
              cat = "SPELL"
              if mismatched_are_matras_only(o_tok.text, c_tok.text):
                cat += ":MATRA"
                return cat
              if mismatched_is_anusvara_only(o_tok.text, c_tok.text):
                cat += ":ANUSVARA"
                return cat
            # If ratio is <= 0.5, the error is more complex e.g. tolk -> say
            else:
                return "OTHER"
            
       
        # MORPHOLOGY
        # Only ADJ, ADV, NOUN and VERB can have inflectional changes.
        lemma_ratio = Levenshtein.ratio(o_tok.lemma_, c_tok.lemma_)
        o_pos = pos_map[o_tok._.feat.get("pos", "NA")]
        c_pos = pos_map[c_tok._.feat.get("pos", "NA")]
        o_feat = o_tok._.feat
        c_feat = c_tok._.feat

        print(o_toks[0].lemma_, o_feat, c_toks[0].lemma_,c_feat)      
        
        if (lemma_ratio >= .85) and \
            pos_map[o_toks[0]._.feat.get("pos", "NA")] in coarse_pos and \
            pos_map[c_toks[0]._.feat.get("pos", "NA")] in coarse_pos:

        # if o_toks[0].lemma == c_toks[0].lemma and \
        #     pos_map[o_toks[0]._.feat.get("pos", "NA")] in coarse_pos and \
        #     pos_map[c_toks[0]._.feat.get("pos", "NA")] in coarse_pos:

            if o_pos == c_pos:

                if o_pos in ("NOUN") and o_tok.lemma == c_tok.lemma:
                    return o_pos + ":INFL"
            
                if o_pos in ("PRON") and o_tok.lemma == c_tok.lemma:
                    if o_feat.get('number') == c_feat.get('number'):
                        return o_pos + ":INFL"

                if o_pos in ("ADJ", "ADP"):
                    return o_pos + ":INFL"
                
                # Verbs - various types
                if o_pos in ("VERB", "AUX"):
                    if o_pos == c_pos:
                        if o_feat.get('tense') == c_feat.get('tense'):
                            return "VERB:INFL"
                        else:
                            return "VERB:FORM"   
                                
        # Derivational morphology
        if stemmer.stem(o_tok.text) == stemmer.stem(c_tok.text) and \
            o_pos in coarse_pos and \
            c_pos in coarse_pos:
            return "MORPH"  

        # Auxiliaries with different lemmas
        if o_pos == "AUX" and o_pos == "AUX":
            return "VERB:TENSE"

        if o_pos == c_pos and o_pos in ("NOUN", "VERB", "ADP", "ADV", "PRON", "ADJ", "CONJ", "NUM"):
            return o_pos

        return "OTHER" 

    o_pos = [pos_map[o_tok._.feat.get("pos", "NA")] for o_tok in o_toks]
    c_pos = [pos_map[c_tok._.feat.get("pos", "NA")] for c_tok in c_toks]

    # Multi-token replacements (uncommon)
    if set(o_pos + c_pos).issubset({"AUX"}):
        return "VERB:TENSE"

    return "OTHER"
    return "OTHER"

def is_only_orth_change(o_toks: list, c_toks: list) -> bool:
    o_join = "".join(o_tok.text for o_tok in o_toks)
    c_join = "".join(c_tok.text for c_tok in c_toks)
    if o_join == c_join:
        return True
    return False

def is_exact_reordering(o_toks: list, c_toks: list) -> bool:
    # Sorting lets us keep duplicates.
    o_set = sorted(o.text for o in o_toks)
    c_set = sorted(c.text for c in c_toks)
    return o_set == c_set

def is_spelling_special_case(o_tok: str, c_tok: str) -> bool:
    for orig_pair in (('આ', 'યા'), ('ઇ', 'ઈ'), ('ઉ', 'ઊ'), ('ચ્', 'છ્'), ('ઘ્', 'ધ્'), ('શ્', 'ષ્')):
        for pair in (orig_pair, orig_pair[::-1]):
            if o_tok.endswith(pair[0]) and c_tok.endswith(pair[1]):
                return o_tok[:-len(pair[0])] == c_tok[:-len(pair[1])]
    return False

def mismatched_are_matras_only(o_tok: str, c_tok: str) -> bool:
    o_tok, c_tok = pad_with_spaces(o_tok, c_tok)
    for x, y in zip(o_tok, c_tok):
            if x != y:
                if x not in matras or y not in matras and not (x==" " or y==" "):
                    return False
    return True    

def mismatched_is_anusvara_only(o_tok: str, c_tok: str) -> bool:
    o_tok, c_tok = pad_with_spaces(o_tok, c_tok)
    for x, y in zip(o_tok, c_tok):
            if x != y:
                if x!=' ં' or y!=' ં'  and not (x==" " or y==" "):
                    return False
    return True 

def pad_with_spaces(s1, s2):
    max_len = max(len(s1), len(s2))
    return s1.ljust(max_len), s2.ljust(max_len)
    