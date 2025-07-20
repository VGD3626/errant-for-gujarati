# ERRANT for Gujarati

This repository provides an adaptation of the [ERRANT](https://github.com/chrisjbryant/errant) (ERRor ANnotation Toolkit) framework for **Gujarati grammatical error correction (GEC)**. It supports automatic extraction and classification of edits between original and corrected Gujarati sentences.

This work is part of **Smruti: Grammatical Error Correction for Gujarati using LLMs with Non-Parametric Memory**.

---

## Example

Example:

Original (Incorrect) Sentence:
તે બાળા ખૂબ તેજસ્વી હતો .

Corrected Sentence:
તે બાળા ખૂબ તેજસ્વી હતી .

Output:
S તે બાળા ખૂબ તેજસ્વી હતો .

A 4 5|||R:VERB:INFL|||હતી|||REQUIRED|||-NONE-|||0

---

## Usage

For usage examples, setup instructions, and evaluation tools, refer to the original ERRANT repository:

🔗 [Original ERRANT GitHub Repository](https://github.com/chrisjbryant/errant)

The Gujarati version follows the same structure and usage pattern, adapted for Gujarati language support.

---

## References

This toolkit builds on:

- Bryant, C., Felice, M., & Briscoe, T. (2017). *Automatic annotation and evaluation of error types for grammatical error correction*. ACL 2017.  
- Felice, M., Bryant, C., & Briscoe, T. (2016). *Automatic extraction of learner errors in ESL sentences using linguistically enhanced alignments*. COLING 2016.

---

## License

This project inherits the original ERRANT license. See the `LICENSE` file for more details.

---
