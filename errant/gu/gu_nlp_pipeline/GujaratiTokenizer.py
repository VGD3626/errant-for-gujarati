import re
import spacy
from spacy.tokens import Doc, Token

stopwords = []

def gujarati_tokenizer(nlp):

    #changes are needed
    def GujaratiTokenizer(data, keep_stopwords = True):
        
        data = re.sub(r'([.,\'\\"!?%#@*<>|\+\-\(\)])', r' \1 ', data)
        data = re.sub(r"   ", '', data)
        data = re.sub(r'…', " ", data)
        data = re.split(r'[ -]',data)
        words = []
        
        if not keep_stopwords:
            for word in data:
                if word not in stopwords:
                    words.append(word)
            spaces=(len(words)-1)*[True]+[False]
            return Doc(nlp.vocab, words=words, spaces=spaces)

        for i in data:
            if i:
                words.append(i)
        spaces=(len(words)-1)*[True]+[False]
        return Doc(nlp.vocab, words=words, spaces=spaces)    

    return GujaratiTokenizer