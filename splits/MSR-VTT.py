from collections import defaultdict
import json

import h5py

from config import MSRVTTSplitConfig as C


def load_metadata():
    with open(C.train_val_caption_fpath, 'r') as fin:
        train_val_data = json.load(fin)
    with open(C.test_caption_fpath, 'r') as fin:
        test_data = json.load(fin)
    captions = train_val_data['sentences'] + test_data['sentences']

    vid2caps = defaultdict(lambda: {})
    for caption in captions:
        vid = caption['video_id']
        cid = caption['sen_id']
        caption = caption['caption']
        vid2caps[vid][cid] = caption
    return vid2caps


def load_videos():
    f = h5py.File(C.video_fpath, 'r')
    return f


def load_splits():
    with open('data/MSR-VTT/metadata/train.list', 'r') as fin:
        train_vids = json.load(fin)
    with open('data/MSR-VTT/metadata/valid.list', 'r') as fin:
        val_vids = json.load(fin)
    with open('data/MSR-VTT/metadata/test.list', 'r') as fin:
        test_vids = json.load(fin)
    return train_vids, val_vids, test_vids


def save_video(fpath, vids, videos):
    fout = h5py.File(fpath, 'w')
    for vid in vids:
        fout[vid] = videos[vid].value
    fout.close()
    print("Saved {}".format(fpath))


def save_metadata(fpath, vids, metadata):
    data = { vid: metadata[vid] for vid in vids }
    with open(fpath, 'w') as fout:
        json.dump(data, fout)
    print("Saved {}".format(fpath))


def split():
    videos = load_videos()
    metadata = load_metadata()

    train_vids, val_vids, test_vids = load_splits()

    save_video(C.train_video_fpath, train_vids, videos)
    save_video(C.val_video_fpath, val_vids, videos)
    save_video(C.test_video_fpath, test_vids, videos)

    save_metadata(C.train_metadata_fpath, train_vids, metadata)
    save_metadata(C.val_metadata_fpath, val_vids, metadata)
    save_metadata(C.test_metadata_fpath, test_vids, metadata)


if __name__ == "__main__":
    split()

