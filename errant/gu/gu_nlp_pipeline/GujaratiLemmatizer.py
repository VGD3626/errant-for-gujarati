from pathlib import Path
import json

suffixes = ['નાં','ના','ની','નો','નું','ને','થી','માં','એ','ઓ','તા','તી','વા','મા','વું','વુ','માંથી','શો','ીશ','ીશું','શે',
			'તો','તું','તાં','્યો','યો','યાં','્યું','યું','ોઈશ', 'ોઈશું', '્યા','યા','્યાં','સ્વી',     
            'ેલ', 'ેલો', 'ેલા', 'ેલું', 'ેલી', 'ણે', 'ણા', 'ણું', 'ણો', 'ણી']

pronoun_mapping = {"હું": "હું", "મને": "હું", "મુજને": "હું", "મારો": "હું", "મારી": "હું", "મારું": "હું", 
                   "મારા": "હું", "મારે": "હું", "મારી જાત": "હું", "અમે": "અમે", "અમને": "અમે", 
                   "અમારો": "અમે", "અમારી": "અમે", "અમારું": "અમે", "અમારા": "અમે", 
                   "અમારે": "અમે", "અમારી જાત": "અમે", "આપણે": "આપણે", "આપણને": "આપણે", 
                   "આપણો": "આપણે", "આપણી": "આપણે", "આપણું": "આપણે", "આપણા": "આપણે", 
                   "આપણી જાત": "આપણે", "તું": "તું", "તને": "તું", "તારો": "તું", "તારી": "તું", 
                   "તારું": "તું", "તારા": "તું", "તારે": "તું", "તારી જાત": "તું", "તમે": "તમે", 
                   "તમને": "તમે", "તમારો": "તમે", "તમારી": "તમે", "તમારું": "તમે", "તમારા": "તમે", 
                   "તમારે": "તમે", "તમારી જાત": "તમે", "એ": "એ", "તે": "એ", "તેને": "એ", 
                   "તેનો": "એ", "તેની": "એ", "તેનું": "એ", "તેનાં":"એ", "તેના": "એ", "તેનાથી": "એ", 
                   "તેનામાં": "એ", "તેની જાત": "એ", "તેઓ": "તેઓ", "તેમને": "તેઓ", "તેમનો": "તેઓ", 
                   "તેમની": "તેઓ", "તેમનું": "તેઓ", "તેમના": "તેઓ", "તેમનાથી": "તેઓ", 
                   "તેમનામાં": "તેઓ", "તેમની જાત": "તેઓ", 
                   "એને": "એ", 
                   "એનો": "એ", "એની": "એ", "એનું": "એ", "એના": "એ", "એનાથી": "એ", 
                   "એનામાં": "એ", "એની જાત": "એ", "એઓ": "એઓ", "એમને": "એઓ", "એમનો": "એઓ", 
                   "એમની": "એઓ", "એમનું": "એઓ", "એમના": "એઓ", "એમનાથી": "એઓ", 
                   "એમનામાં": "એઓ", "એમની જાત": "એઓ", "આ": "આ"
}

aux_suffixes = {
    'તો','તું', 'તુ',  'ં', 'તાં','તા','તી','શો', 'ોઈશ', 'ોઈશું', 'ોઈશુ', 'શે','ે', 'ું', 'ં', 'ીએ', 'ો'
}

noun_and_adj_suffixes = {
    'નાં','ના','ની','નો', 'ં', 'નું','ને','થી', 'મા', 'માં','એ','ઓ','માંથી','સ્વી', 'ી', 'ં', 'ે', 'ા', 'ો', 'ું',
}

verb_suffixes = {
    'તા','તી','તો','તું','તાં', 'ઇ', 'ઈ', 'શો', 'શે', 'શ', 'ેલાં', 'ેલ', 'ેલો', 'ેલા', 'ેલું', 'ેલી', 'ો'
    '્યા','યા','્યાં','્યો','યા','યો','યાં','્યું','્યુ','યું', 'યુ', 'વું','વુ'
}

def load_lemma_dict(path, encoding="utf-8"):
    with open(path, encoding="utf-8") as mappings:
        mapping_dict = json.load(mappings)        
    return mapping_dict

base_dir = Path(__file__).resolve().parent

lemma_dict = load_lemma_dict(base_dir/"lemma_dict.json")


def word_lemmatization(token):
    word,pos = token.text, token._.feat.get("pos", "NA")

    #Pronoun
    if pos.startswith("PR_") and word in pronoun_mapping:
        return pronoun_mapping[word.strip()]

    #Noun and Adj (Both can have similar suffixes)
    if pos.startswith("N_") or pos.startswith("JJ"):
        for suffix in sorted(noun_and_adj_suffixes, key=len, reverse=True): #largest suffix first
            if word.endswith(suffix):
                return word[:-len(suffix)]

    #Verb      
    if pos.startswith("V_VM"):
        for suffix in sorted(verb_suffixes, key=len, reverse=True): #largest suffix first
            if word.endswith(suffix):
                return word[:-len(suffix)]
    
    #aux
    if pos.startswith("V_V"):
        for suffix in sorted(aux_suffixes, key=len, reverse=True): #largest suffix first
            if word.endswith(suffix):
                return word[:-len(suffix)]
            
    return word



def gujarati_lemmatizer(token):
    
    # Follow rule-based approach it word is not found in dictionary
    # A complete dictionary is prepared for pronoun, so no need to rely on dict (from Unimorph)
    if token._.feat.get("pos", "NA").startswith("PR_") or token.text not in lemma_dict.keys():
        return word_lemmatization(token)
    else:
        return lemma_dict[token.text]
