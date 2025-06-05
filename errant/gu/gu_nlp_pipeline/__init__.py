from errant.gu.gu_nlp_pipeline.GujaratiTokenizer import gujarati_tokenizer
from errant.gu.gu_nlp_pipeline.GujaratiStemmer import Stemmer
from errant.gu.gu_nlp_pipeline.GujaratiTagger import Tagger

import spacy
from spacy.language import Language


@Language.component("GujaratiStemmer")
def GujStemmer(doc):
    gstmr = Stemmer()
    lemmas = gstmr.stem_word(str(doc))
    for (token, lemma) in zip(doc, lemmas):
        token.lemma_ = lemma
    return doc

@Language.component("GujaratiTagger")
def GujTagger(doc):
    gtagger = Tagger()
    tags = gtagger.tag(str(doc))
    for token, tag in zip(doc, tags):
        token.tag_ = tag#tag[1]
    return doc

nlp_gu = Language()
nlp_gu.tokenizer = gujarati_tokenizer(nlp_gu)
nlp_gu.add_pipe("GujaratiStemmer", "stemmer")
nlp_gu.add_pipe("GujaratiTagger", "tagger")

print(nlp_gu("1 . બાળકો ક્રિકેટનો ખેલ રમી રહ્યા છે"))
