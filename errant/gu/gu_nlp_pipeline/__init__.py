from errant.gu.gu_nlp_pipeline.GujaratiTokenizer import gujarati_tokenizer
from errant.gu.gu_nlp_pipeline.GujaratiMorphAnalyzer.GujaratiAnalyzer import gujarati_analyzer
from errant.gu.gu_nlp_pipeline.GujaratiLemmatizer import gujarati_lemmatizer
from spacy.language import Language
from spacy.tokens import Token


Token.set_extension("feat", default={}, force=True)
# Token.set_extension("number", default="NA", force=True)
# Token.set_extension("person", default="NA", force=True)
# Token.set_extension("tense", default="NA", force=True)
# Token.set_extension("case", default="NA", force=True)
# Token.set_extension("aspect", default="NA", force=True)

@Language.component("GujaratiLemmatizer")
def GujLemmatizer(doc):
    for token in doc:
        token.lemma_ = gujarati_lemmatizer(token)
    return doc

@Language.component("GujaratiMorphAnalyzer")
def GujAnalyzer(doc):
    features = gujarati_analyzer(str(doc))
    for token, feats in zip(doc, features):
        token._.feat = feats
    print(features)
    return doc


nlp_gu = Language()
nlp_gu.tokenizer = gujarati_tokenizer(nlp_gu)
nlp_gu.add_pipe("GujaratiLemmatizer", "lemmatizer")
nlp_gu.add_pipe("GujaratiMorphAnalyzer", "analyzer")

print(nlp_gu("1 . બાળકો ક્રિકેટનો ખેલ રમી રહ્યા છે"))
