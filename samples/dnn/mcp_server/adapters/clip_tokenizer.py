"""CLIP's byte-level BPE, for the text input of OWLv2 and SAM 3.

CLIP marks the last piece of each word with "</w>". cv.dnn.Tokenizer does not handle that marker
yet, so it encodes these prompts into the wrong token ids. Once it does, the adapters can call
cv.dnn.Tokenizer.load() instead and this file can go.

Reads the tokenizer.json of any CLIP model, e.g. openai/clip-vit-base-patch32; they share one vocabulary.
"""

import json
import re
import unicodedata

import numpy as np

# The Split pattern of CLIP's tokenizer.json, with \p{L} and \p{N} written for Python's re module.
WORD = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+")


def bytes_to_unicode():
    """GPT-2's map from the 256 byte values to printable characters, which the vocabulary uses."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ClipTokenizer:
    def __init__(self, tokenizer_json):
        with open(tokenizer_json, encoding="utf-8") as f:
            model = json.load(f)["model"]
        self.vocab = model["vocab"]
        self.ranks = {tuple(m.split(" ") if isinstance(m, str) else m): i for i, m in enumerate(model["merges"])}
        self.bos = self.vocab["<|startoftext|>"]
        self.eos = self.vocab["<|endoftext|>"]
        self.byte_map = bytes_to_unicode()
        self.cache = {}

    def bpe(self, word):
        """Merge the characters of one word, lowest merge rank first, as BPE does."""
        if word in self.cache:
            return self.cache[word]
        parts = list(word[:-1]) + [word[-1] + "</w>"]
        while len(parts) > 1:
            pairs = [(self.ranks.get(p, float("inf")), i) for i, p in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == float("inf"):
                break
            first, second = parts[i], parts[i + 1]
            merged = []
            j = 0
            while j < len(parts):
                if j < len(parts) - 1 and parts[j] == first and parts[j + 1] == second:
                    merged.append(first + second)
                    j += 2
                else:
                    merged.append(parts[j])
                    j += 1
            parts = merged
        self.cache[word] = parts
        return parts

    def encode(self, text):
        """Token ids of one prompt, with the start and end tokens."""
        text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()
        ids = []
        for word in WORD.findall(text):
            word = "".join(self.byte_map[b] for b in word.encode("utf-8"))
            ids += [self.vocab[p] for p in self.bpe(word)]
        return [self.bos] + ids + [self.eos]

    def encode_batch(self, texts, length, pad_id=0):
        """input_ids and attention_mask of shape (len(texts), length), as int64.
        A prompt that is too long is cut, keeping its end token."""
        ids = np.full((len(texts), length), pad_id, dtype=np.int64)
        mask = np.zeros((len(texts), length), dtype=np.int64)
        for i, text in enumerate(texts):
            tokens = self.encode(text)
            if len(tokens) > length:
                tokens = tokens[:length - 1] + [self.eos]
            ids[i, :len(tokens)] = tokens
            mask[i, :len(tokens)] = 1
        return ids, mask
