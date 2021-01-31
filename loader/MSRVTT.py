import json

from loader.data_loader import CustomVocab, CustomDataset, Corpus


class MSRVTTVocab(CustomVocab):
    """ MSR-VTT Vocaburary """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        captions = []
        for vid, depth1 in data.items():
            for sid, caption in depth1.items():
                captions.append(caption)
        return captions


class MSRVTTDataset(CustomDataset):
    """ MSR-VTT Dataset """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        for vid, depth1 in data.items():
            for caption in depth1.values():
                self.captions[vid].append(caption)


class MSRVTT(Corpus):
    """ MSR-VTT Corpus """

    def __init__(self, C):
        super(MSRVTT, self).__init__(C, MSRVTTVocab, MSRVTTDataset)

