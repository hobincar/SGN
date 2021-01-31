import re
import string

from nltk.tokenize import wordpunct_tokenize
import numpy as np
import torch


class UniformSample:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, frames):
        n_frames = len(frames)
        if n_frames < self.n_sample:
            return frames

        sample_indices = np.linspace(0, n_frames-1, self.n_sample, dtype=int)
        samples = [ frames[i] for i in sample_indices ]
        return samples


class RandomSample:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, frames):
        n_frames = len(frames)
        if n_frames < self.n_sample:
            return frames

        block_len = int(n_frames / self.n_sample)
        start_of_final_clip = n_frames - block_len - 1
        uniformly_sampled_indices = np.linspace(0, start_of_final_clip, self.n_sample, dtype=int)
        random_noise = np.random.choice(block_len, self.n_sample, replace=True)
        randomly_sampled_indices = uniformly_sampled_indices + random_noise
        samples = [ frames[i] for i in randomly_sampled_indices ]
        return samples


class TrimIfLongerThan:
    def __init__(self, n):
        self.n = n

    def __call__(self, frames):
        if len(frames) > self.n:
            frames = frames[:self.n]
        return frames


class ZeroPadIfLessThan:
    def __init__(self, n):
        self.n = n

    def __call__(self, frames):
        while len(frames) < self.n:
            frames = np.vstack([ frames, np.zeros_like(frames[0]) ])
        return frames


class ToTensor:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, array):
        np_array = np.asarray(array)
        t = torch.from_numpy(np_array)
        if self.dtype:
            t = t.type(self.dtype)
        return t


class NLTKWordpunctTokenizer:

    def __call__(self, sentence):
        return wordpunct_tokenize(sentence)


class TrimExceptAscii:
    def __init__(self, corpus):
        self.corpus = corpus

    def __call__(self, sentence):
        if self.corpus == "MSVD":
            s = sentence.decode('ascii', 'ignore').encode('ascii')
        elif self.corpus in [ "MSR-VTT", "VATEX", "YouCook2" ]:
            s = sentence.encode('ascii', 'ignore')
        return s


class RemovePunctuation:
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, sentence):
        return self.regex.sub('', sentence)


class Lowercase:

    def __call__(self, sentence):
        return sentence.lower()


class SplitWithWhiteSpace:

    def __call__(self, sentence):
        return sentence.split()


class Truncate:
    def __init__(self, n_word):
        self.n_word = n_word

    def __call__(self, words):
        return words[:self.n_word]


class PadFirst:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return [ self.token ] + words


class PadFirst:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return [ self.token ] + words


class PadLast:
    def __init__(self, token):
        self.token = token

    def __call__(self, words):
        return words + [ self.token ]


class PadToLength:
    def __init__(self, token, length):
        self.token = token
        self.length = length

    def __call__(self, words):
        n_pads = self.length - len(words)
        return words + [ self.token ] * n_pads


class ToIndex:
    def __init__(self, word2idx):
        self.word2idx = word2idx

    def __call__(self, words): # Ignore unknown (or trimmed) words.
        return [ self.word2idx[word] for word in words ]

