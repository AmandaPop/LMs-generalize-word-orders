import itertools
import random
from collections import defaultdict
from abc import abstractmethod
from stanza.models.common.doc import Sentence as StanzaSentence
from corpus_filtering.filters.core_filters import register_filter
from corpus_filtering.filters.stanza_filters import PickleStanzaDocCorpusFilterWriter

class Perturbance:
    @abstractmethod
    def perturb(self, sent: StanzaSentence) -> list[str]:
        """Return a list of perturbed sentences as strings."""
        pass

class Filter(Perturbance, PickleStanzaDocCorpusFilterWriter):
    """Base filter: writes accepted sentences or perturbs rejected ones."""

    def _write(self, sent: str, reject: bool):
        self.total_sentences += 1
        rand_val = random.random()
        #assign split first
        if rand_val < 0.8:
            split = "train"
            split_file = self._f_accept_out_train
        elif rand_val < 0.9:
            split = "val"
            split_file = self._f_accept_out_val
        else:
            split = "test"
            split_file = self._f_accept_out_test

        sent_str = self._sent_to_str(sent)

        if not reject:
            # accepted sentence
            split_file.write(sent_str + "\n")
            self.simple_sentences += 1
        else:
            # perturb and write all perturbed variants
            sents = self.perturb(sent, split=split)
            for s in sents:
                split_file.write(s + "\n")
            self.perturbed_sentences += len(sents)

@register_filter("simple-np")
class SimpleNPFilter(Filter):
    demonstratives = {"this", "that", "these", "those"}

    def __init__(self, f_in, f_accept_out_train, f_accept_out_val, f_accept_out_test, f_reject_out=None, *args, **kwargs):
        super().__init__(
            f_in=f_in,
            f_accept_out_path=f_accept_out_train,
            f_reject_out_path=f_reject_out,
            *args,
            **kwargs,
        )
        #store split files
        self._f_accept_out_train = open(f_accept_out_train, "w", encoding="utf-8")
        self._f_accept_out_val = open(f_accept_out_val, "w", encoding="utf-8")
        self._f_accept_out_test = open(f_accept_out_test, "w", encoding="utf-8")
        #stats
        self.total_sentences = 0
        self.simple_sentences = 0
        self.complex_sentences = 0
        self.perturbed_sentences = 0
        self.cito_sentences = 0
        self.perturbed_sentence_txt = []
        self.minimal_pairs = []
        self.minimal_3_pairs = []

    def close(self):
        for output_file in (
            self._f_accept_out_train,
            self._f_accept_out_val,
            self._f_accept_out_test,
        ):
            if not output_file.closed:
                output_file.close()
        super().close()

    def _has_CITO(self, sent: StanzaSentence) -> bool:
        for i in range(len(sent.words) - 1):
            w, next_w = sent.words[i], sent.words[i + 1]
            if w.text.lower() == "that" and w.pos == "SCONJ" and next_w.text.lower() == "one":
                self.cito_sentences += 1
                return True
        return False

    def _has_complex_np(self, sent: StanzaSentence) -> bool:
        if self._has_CITO(sent):
            self.complex_sentences += 1
            return True

        mod_types = defaultdict(set)
        for head, deprel, word in sent.dependencies:
            if deprel == "det" and word.text.lower() in self.demonstratives:
                mod_types[head].add("det")
            elif deprel == "nummod":
                mod_types[head].add("num")
            elif deprel == "amod":
                mod_types[head].add("adj")

        for types in mod_types.values():
            if len(types) > 1:
                self.complex_sentences += 1
                return True

        return False

    def _exclude_sent(self, sent: StanzaSentence) -> bool:
        return self._has_complex_np(sent)

    def extract_modifiers(self, sent: StanzaSentence):
        mod_nouns = defaultdict(list)
        adj_groups = defaultdict(list)
        adv_map = defaultdict(list)

        #map adverbs modifying adjectives
        for head, deprel, word in sent.dependencies:
            if deprel == "advmod" and word.upos == "ADV":
                adv_map[head].append(word.text.lower())

        for head, deprel, word in sent.dependencies:
            if deprel == "det" and word.text.lower() in self.demonstratives:
                mod_nouns[(head.text, head.id)].append((word.text.lower(), "Dem"))
            elif deprel == "nummod":
                mod_nouns[(head.text, head.id)].append((word.text.lower(), "Num"))
            elif deprel == "amod":
                advs = adv_map.get(word, [])
                adj_block = " ".join(advs + [word.text.lower()])
                adj_groups[head.id].append(adj_block)

        #combine multiple adjectives into one block
        for noun_id, adjs in adj_groups.items():
            adj_block = " ".join(adjs)
            noun_word = next(w for w in sent.words if w.id == noun_id)
            mod_nouns[(noun_word.text, noun_id)].append((adj_block, "Adj"))

        return mod_nouns

    def build_sentence(self, sent, chosen_mods, mod_nouns):
        new_sentence = []
        for word in sent.words:
            if word.deprel in {"nummod", "amod"} or (word.deprel == "det" and word.text.lower() in self.demonstratives):
                continue

            # skip adverbs modifying adjectives (they are prepended in adj block)
            if word.deprel == "advmod":
                head_idx = word.head
                if head_idx > 0:
                    head_word = sent.words[head_idx - 1]
                    if head_word.deprel == "amod":
                        continue

            if word.upos == "NOUN":
                key = (word.text, word.id)
                if key in chosen_mods:
                    phrase = " ".join(m[0] for m in chosen_mods[key])
                    new_sentence.append(f"{phrase} {word.text}")
                else:
                    new_sentence.append(word.text)
            else:
                new_sentence.append(word.text)
        return " ".join(new_sentence)

    def perturb(self, sent: StanzaSentence, split: str = None):
        mod_nouns = self.extract_modifiers(sent)

        # create minimal pair
        pair = self.build_2_minimal_pair(sent, mod_nouns, split=split)
        if pair:
            self.minimal_pairs.append(pair)
        
        #create 3 modifier minimal pairs

        # for each noun with 3 or more modifiers
        for noun, mods in mod_nouns.items():
            # get the type sequence in order of appearance in the sentence
            type_seq = [m[1] for m in mods]
            if type_seq == ["Dem", "Num", "Adj"]:
                # only now build 3-minimal-pairs for this noun
                pair_3 = self.build_3_minimal_pairs(sent, mod_nouns, split)
                if pair_3:
                    self.minimal_3_pairs.append(pair_3)

        # Build all single-mod variants
        noun_options = []
        nouns_with_multiple_modifiers = []
        for noun, mods in mod_nouns.items():
            options = [[m] for m in mods]
            if len(mods) > 1:
                noun_options.append(options)
                nouns_with_multiple_modifiers.append(noun)
        combos = itertools.product(*noun_options)

        sentences = []
        for combo in combos:
            chosen = {
                noun: mod
                for noun, mod in zip(nouns_with_multiple_modifiers, combo)
            }
            sentences.append(self.build_sentence(sent, chosen, mod_nouns))

        self.perturbed_sentence_txt.append(sentences)
        return sentences

    def build_2_minimal_pair(self, sent, mod_nouns, split: str = None):
        candidates = [k for k, v in mod_nouns.items() if len(v) >= 2]
        if not candidates:
            return {}

        noun = random.choice(candidates)
        mods = random.sample(mod_nouns[noun], 2)  #random two modifiers
        m1, m2 = mods

        key1 = f"{m1[1]}-{m2[1]}"
        key2 = f"{m2[1]}-{m1[1]}"

        chosen1 = {noun: [m1, m2]}
        chosen2 = {noun: [m2, m1]}

        sent1 = self.build_sentence(sent, chosen1, mod_nouns)
        sent2 = self.build_sentence(sent, chosen2, mod_nouns)

        #noun phrases being altered
        np1 = f"{m1[0]} {m2[0]} {noun[0]}"
        np2 = f"{m2[0]} {m1[0]} {noun[0]}"

        pair_id = len(self.minimal_pairs) + 1

        return {
            key1: sent1,
            key2: sent2,
            "np_good": np1,
            "np_bad": np2,
            "split": split,
            "original" : sent.text,
            "pairID": pair_id
        }

    def build_3_minimal_pairs(self, sent, mod_nouns, split: str = None):
        noun = next(k for k, v in mod_nouns.items() if len(v) > 2)
        mods = mod_nouns[noun]

        pair_id = len(self.minimal_3_pairs) + 1
        results = {}

        for perm in itertools.permutations(mods):
            m1, m2, m3 = perm

            key = f"{m1[1]}-{m2[1]}-{m3[1]}"

            chosen = {noun: [m1, m2, m3]}
            sent_variant = self.build_sentence(sent, chosen, mod_nouns)

            np_variant = f"{m1[0]} {m2[0]} {m3[0]} {noun[0]}"

            results[key] = {
                "sentence": sent_variant,
                "np": np_variant
            }

        results["split"] = split
        results["original"] = sent.text
        results["pairID"] = pair_id

        return results