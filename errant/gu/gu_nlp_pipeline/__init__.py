from errant.gu.gu_nlp_pipeline.GujaratiTokenizer import gujarati_tokenizer
from errant.gu.gu_nlp_pipeline.GujaratiStemmer import Stemmer
import spacy
from spacy.language import Language


@Language.component("GujaratiStemmer")
def GujStemmer(doc):
    gstmr = Stemmer()
    for token in doc:
        lemma = gstmr.stem(token.text)
        token.lemma_ = lemma
    return doc

nlp_gu = Language()
nlp_gu.tokenizer = gujarati_tokenizer
nlp_gu.add_pipe("GujaratiStemmer", "stemmer")

