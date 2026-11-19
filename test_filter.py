import sys
import pickle
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parent / "corpus_filtering"
sys.path.insert(0, str(PACKAGE_PARENT))

import stanza
from scripts.perturbator import SimpleNPFilter


TEST_SENTENCE = "Over those two years it prospered."

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
doc = nlp(TEST_SENTENCE)

with open("tmp_sentence.pkl", "wb") as file:
    pickle.dump(doc, file)

#then just replace this with your new filter
with SimpleNPFilter(
    f_in="tmp_sentence.pkl",
    f_accept_out_train="train.txt",
    f_accept_out_val="txt",
    f_accept_out_test="test.txt",
    f_reject_out="rejected.txt",
) as perturbator:
    
    perturbator.filter_write()