from collections import Counter, defaultdict
import json

import nltk
import pandas as pd
from torchvision import transforms
from tqdm import tqdm

from loader.transform import TrimExceptAscii, Lowercase, RemovePunctuation, SplitWithWhiteSpace


def load_MSVD_captions(caption_fpath):
    df = pd.read_csv(caption_fpath)
    df = df[df['Language'] == 'English']
    df = df[pd.notnull(df['Description'])]
    video_names = df['VideoID'].values
    start_times = df['Start'].values
    end_times = df['End'].values
    vids = [ '{}_{}_{}'.format(v, s, e) for v, s, e in zip(video_names, start_times, end_times) ]
    captions = df['Description'].values

    transform_sentence = transforms.Compose([
        TrimExceptAscii('MSVD'),
        Lowercase(),
        RemovePunctuation(),
        SplitWithWhiteSpace() ])
    captions = [ transform_sentence(caption) for caption in captions ]

    vid2captions = defaultdict(lambda: [])
    assert len(vids) == len(captions)
    for vid, caption in zip(vids, captions):
        vid2captions[vid].append(caption)
    return vid2captions


def load_MSRVTT_captions(caption_fpath):
    with open(caption_fpath, 'r') as fin:
        data = json.load(fin)

    vid2captions = defaultdict(lambda: [])
    for vid, depth1 in data.items():
        for caption in depth1.values():
            vid2captions[vid].append(caption)

    transform_sentence = transforms.Compose([
        TrimExceptAscii('MSR-VTT'),
        Lowercase(),
        RemovePunctuation(),
        SplitWithWhiteSpace() ])
    for vid, captions in vid2captions.items():
        vid2captions[vid] = [ transform_sentence(caption) for caption in vid2captions[vid] ]

    return vid2captions


def extract_negative_samples(corpus, vid2captions):
    def captions2words(captions, stopwords=[]):
        words = []
        for caption in captions:
            words += [ w for w in caption if w not in stopwords ]
        return set(words)

    """ Word Stats """
    total_words = []
    for captions in vid2captions.values():
        total_words += captions2words(captions)
    total_words_counter = Counter(total_words)
    with open("data/{}/metadata/words_stats.csv".format(corpus), 'w') as fout:
        for word, cnt in total_words_counter.items():
            fout.write("{}, {}\n".format(word, cnt))

    """ Stopwords Stats """
    total_words = set(total_words)
    total_stopwords = nltk.corpus.stopwords.words('english')
    stopwords = [ word for word in total_words if word in total_stopwords ]
    with open("data/{}/metadata/stopwords_stats.csv".format(corpus), 'w') as fout:
        batch_size = 10
        for i in range(0, len(stopwords), batch_size):
            fout.write(','.join(stopwords[i:i + batch_size]) + '\n')

    vid2words = { vid: captions2words(captions, stopwords) for vid, captions in vid2captions.items() }
    with open("data/{}/metadata/vid2vocabs.json".format(corpus), 'w') as fout:
        json.dump({ vid: list(vocabs) for vid, vocabs in vid2words.items() }, fout)

    vid2neg_vids = defaultdict(lambda: [])
    for vid1 in tqdm(vid2words.keys()):
        for vid2 in vid2words.keys():
            if not vid2words[vid1] & vid2words[vid2]:
                vid2neg_vids[vid1].append(vid2)
        K = 0
        while len(vid2neg_vids[vid1]) == 0:
            K += 1
            print("{} does not have any negative videos. Allow {} words to be overlapped.".format(vid1, K))
            for vid2 in vid2words.keys():
                if len(vid2words[vid1] & vid2words[vid2]) <= K:
                    vid2neg_vids[vid1].append(vid2)

    vid2neg_captions = defaultdict(lambda: {})
    for pos_vid, pos_captions in tqdm(vid2captions.items()):
        vid2neg_captions[pos_vid]['captions'] = []
        for pos_caption in pos_captions:
            vid2neg_captions[pos_vid]['captions'].append(' '.join(pos_caption))

        vid2neg_captions[pos_vid]['negative_videos'] = {}
        neg_vids = vid2neg_vids[pos_vid]
        for neg_vid in neg_vids:
            neg_captions = vid2captions[neg_vid]
            vid2neg_captions[pos_vid]['negative_videos'][neg_vid] = []
            for neg_caption in neg_captions:
                vid2neg_captions[pos_vid]['negative_videos'][neg_vid].append(' '.join(neg_caption))
    with open("data/{}/metadata/vid2neg_captions.json".format(corpus), 'w') as fout:
        json.dump(vid2neg_captions, fout)


    """ Negative Video Stats """
    num_neg_vids = [ len(vids) for vids in vid2neg_vids.values() ]
    counter = Counter(num_neg_vids)
    print("The percentage of videos that have negative videos: {:.1f}%".format(
        100. * len(vid2neg_vids) / len(vid2captions) ))
    print("Average number of negative videos: {:.1f} videos".format(
        float(sum(num_neg_vids)) / len(num_neg_vids) ))
    with open("data/{}/metadata/neg_vids_stats.csv".format(corpus), 'w') as fout:
        for num, cnt in counter.items():
            fout.write("{}, {}\n".format(num, cnt))
    return vid2neg_vids


def main(dataset, split):
    if dataset == 'MSVD':
        caption_fpath = "./data/{}/metadata/{}.csv".format(dataset, split)
        vid2captions = load_MSVD_captions(caption_fpath)
    elif dataset == 'MSR-VTT':
        caption_fpath = "./data/{}/metadata/{}.json".format(dataset, split)
        vid2captions = load_MSRVTT_captions(caption_fpath)
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(dataset))

    negative_videos = extract_negative_samples(dataset, vid2captions)

    with open("data/{}/metadata/neg_vids_{}.json".format(dataset, split), 'w') as fout:
        json.dump(negative_videos, fout)

if __name__ == '__main__':
    for dataset in [ 'MSVD', 'MSR-VTT' ]:
        for split in [ 'train', 'val', 'test' ]:
            main(dataset, split)

