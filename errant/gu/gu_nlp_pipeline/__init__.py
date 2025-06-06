from errant.gu.gu_nlp_pipeline.GujaratiTokenizer import gujarati_tokenizer
from errant.gu.gu_nlp_pipeline.GujaratiStemmer import Stemmer
from errant.gu.gu_nlp_pipeline.GujaratiMorphAnalyzer.GujaratiAnalyzer import gujarati_analyzer
import spacy
from spacy.language import Language
from spacy.tokens import Token

Token.set_extension("gender", default="NA", force=True)
Token.set_extension("number", default="NA", force=True)
Token.set_extension("person", default="NA", force=True)
Token.set_extension("tense", default="NA", force=True)
Token.set_extension("case", default="NA", force=True)
Token.set_extension("aspect", default="NA", force=True)

@Language.component("GujaratiStemmer")
def GujStemmer(doc):
    gstmr = Stemmer()
    lemmas = gstmr.stem_word(str(doc))
    for (token, lemma) in zip(doc, lemmas):
        token.lemma_ = lemma
    return doc

@Language.component("GujaratiMorphAnalyzer")
def GujAnalyzer(doc):
    features = gujarati_analyzer(str(doc))
    for token, feats in zip(doc, features):
        token.tag_ = feats.get("pos", "NA")
        token._.gender = feats.get("gender", "NA")
        token._.number = feats.get("number", "NA")
        token._.person = feats.get("person", "NA")
        token._.tense = feats.get("tense", "NA")
        token._.case = feats.get("case", "NA")
        token._.aspect = feats.get("aspect", "NA")
    print(features)
    return doc


nlp_gu = Language()
nlp_gu.tokenizer = gujarati_tokenizer(nlp_gu)
nlp_gu.add_pipe("GujaratiStemmer", "stemmer")
nlp_gu.add_pipe("GujaratiMorphAnalyzer", "analyzer")

print(nlp_gu("1 . બાળકો ક્રિકેટનો ખેલ રમી રહ્યા છે"))
