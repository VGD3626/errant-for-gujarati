from importlib import import_module
import spacy
from errant.annotator import Annotator
from errant.gu.gu_nlp_pipeline import nlp_gu

# ERRANT version
__version__ = '3.0.0'

# Load an ERRANT Annotator object for a given language
def load(lang, nlp=None):
    # Make sure the language is supported
    supported = {"en", "gu"}
    if lang not in supported:
        raise Exception(f"{lang} is an unsupported or unknown language")

    # Load spacy (small model if no model supplied)
    if lang == "en":    
        nlp = nlp or spacy.load(f"{lang}_core_web_sm", disable=["ner"])

    #GUJ
    if lang == "gu":
        nlp = nlp_gu

    # Load language edit merger
    merger = import_module(f"errant.{lang}.merger")

    # Load language edit classifier
    classifier = import_module(f"errant.{lang}.classifier")
    # The English classifier needs spacy
    if lang == "en": classifier.nlp = nlp

    if lang == "gu": classifier.nlp = nlp

    # Return a configured ERRANT annotator
    return Annotator(lang, nlp, merger, classifier)