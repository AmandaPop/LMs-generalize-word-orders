"""
To try out new sentences:
1) Change them in test_sentence.txt
2) Run python scripts/stanza_serialize.py w test_sentences.txt sentences.pkl -t
3) Then can run this file 
"""

import pickle
#import stanza
#from corpus_perturbation.scripts.perturbator import SimpleNPFilter

with open("sentences.pkl", "rb") as f:
    doc = pickle.load(f)

for sentence in doc.sentences:
    print(f"Sentence: {sentence.text}")
    for word in sentence.words:
        print(word)

# perturbator = SimpleNPFilter(
#     f_in="tmp_sentence.pkl",
#     f_accept_out_train="tmp_single_accept_train.txt",
#     f_accept_out_val="tmp_single_accept_val.txt",
#     f_accept_out_test="tmp_single_accept_test.txt",
#     f_reject_out="tmp_single_rejected.txt",
# )

# perturbator.filter_write()
# for stanza_sent in doc.sentences:
#     is_complex = perturbator._has_complex_np(stanza_sent)
#     if is_complex:
#         print(f"Complex NP Sentence: {sent}")
#     else:
#         print(f"Simple NP Sentence: {sent}")

# print("perturbed variants:", perturbator.perturbed_sentence_txt)
# print("minimal pairs:", perturbator.minimal_pairs)