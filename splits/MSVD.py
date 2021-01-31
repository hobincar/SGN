import json

import h5py
import pandas as pd

from config import MSVDSplitConfig as C


def load_metadata():
    df = pd.read_csv(C.caption_fpath)
    df = df[df['Language'] == 'English']
    df = df[pd.notnull(df['Description'])]
    df = df.reset_index(drop=True)
    return df


def load_videos():
    f = h5py.File(C.video_fpath, 'r')
    return f


def load_splits():
    with open('data/MSVD/metadata/train.list', 'r') as fin:
        train_vids = json.load(fin)
    with open('data/MSVD/metadata/valid.list', 'r') as fin:
        val_vids = json.load(fin)
    with open('data/MSVD/metadata/test.list', 'r') as fin:
        test_vids = json.load(fin)
    return train_vids, val_vids, test_vids


def save_video(fpath, vids, videos):
    fout = h5py.File(fpath, 'w')
    for vid in vids:
        fout[vid] = videos[vid].value
    fout.close()
    print("Saved {}".format(fpath))


def save_metadata(fpath, vids, metadata_df):
    vid_indices = [ i for i, r in metadata_df.iterrows() if "{}_{}_{}".format(r[0], r[1], r[2]) in vids ]
    df = metadata_df.iloc[vid_indices]
    df.to_csv(fpath)
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

