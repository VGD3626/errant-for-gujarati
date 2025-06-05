
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Transformer-based POS-tagger for Gujarati
class Tagger:
    def __init__(self, model_name="om-ashish-soni/guj-pos-tagging-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        # self.label_map = self.model.config.id2label

    def tag(self, sentence):
        return "pos"
        # Tokenize input
        tokens = self.tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Map token indices to words
        word_ids = (
            tokens['input_ids'][0].cpu().tolist()
        )
        token_word_mapping = (
            tokens.encodings[0].word_ids if hasattr(tokens, 'encodings') else self.tokenizer(sentence).word_ids()
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**tokens)
        predictions = torch.argmax(outputs.logits, dim=2)

        # Convert token IDs to tokens
        input_tokens = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        predicted_labels = predictions[0].cpu().tolist()

        # Align predictions with words
        word_id_to_word = {}
        for idx, word_id in enumerate(token_word_mapping):
            if word_id is None:
                continue
            token = input_tokens[idx]
            label = self.label_map[predicted_labels[idx]]
            if word_id not in word_id_to_word:
                word_id_to_word[word_id] = (token.replace("##", ""), label)
            else:
                prev_word, prev_label = word_id_to_word[word_id]
                word_id_to_word[word_id] = (prev_word + token.replace("##", ""), prev_label)

        return list(word_id_to_word.values())

if __name__ == "__main__":
    tagger = Tagger()
    sentence = "એક છોકરો નવા આવેલા ત્રીસ અંગ્રેજ અફસરની આગળ ચાલ્યો ."
    tagged = tagger.tag(sentence)
    for word, tag in tagged:
        print(f"{word:20} -> {tag}")
