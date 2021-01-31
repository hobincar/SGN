import os


class MSVDSplitConfig:
    model = "3DResNext101"

    video_fpath = "data/MSVD/features/{}.hdf5".format(model)
    caption_fpath = "data/MSVD/metadata/MSR Video Description Corpus.csv"

    train_video_fpath = "data/MSVD/features/{}_train.hdf5".format(model)
    val_video_fpath = "data/MSVD/features/{}_val.hdf5".format(model)
    test_video_fpath = "data/MSVD/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "data/MSVD/metadata/train.csv"
    val_metadata_fpath = "data/MSVD/metadata/val.csv"
    test_metadata_fpath = "data/MSVD/metadata/test.csv"


class MSRVTTSplitConfig:
    model = "3DResNext101"

    video_fpath = "data/MSR-VTT/features/{}.hdf5".format(model)
    train_val_caption_fpath = "data/MSR-VTT/metadata/train_val_videodatainfo.json"
    test_caption_fpath = "data/MSR-VTT/metadata/test_videodatainfo.json"

    train_video_fpath = "data/MSR-VTT/features/{}_train.hdf5".format(model)
    val_video_fpath = "data/MSR-VTT/features/{}_val.hdf5".format(model)
    test_video_fpath = "data/MSR-VTT/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "data/MSR-VTT/metadata/train.json"
    val_metadata_fpath = "data/MSR-VTT/metadata/val.json"
    test_metadata_fpath = "data/MSR-VTT/metadata/test.json"


class VocabConfig:
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3 }
    embedding_size = 300
    pretrained = 'GloVe'


class MSVDLoaderConfig:
    train_caption_fpath = "data/MSVD/metadata/train.csv"
    val_caption_fpath = "data/MSVD/metadata/val.csv"
    test_caption_fpath = "data/MSVD/metadata/test.csv"
    min_count = 1
    max_caption_len = 15

    total_video_feat_fpath_tpl = "data/{}/features/{}.hdf5"
    split_video_feat_fpath_tpl = "data/{}/features/{}_{}.hdf5"
    frame_sample_len = 30

    split_negative_vids_fpath = "data/MSVD/metadata/neg_vids_{}.json"

    num_workers = 1


class MSRVTTLoaderConfig:
    train_caption_fpath = "data/MSR-VTT/metadata/train.json"
    val_caption_fpath = "data/MSR-VTT/metadata/val.json"
    test_caption_fpath = "data/MSR-VTT/metadata/test.json"
    min_count = 3
    max_caption_len = 15

    total_video_feat_fpath_tpl = "data/{}/features/{}.hdf5"
    split_video_feat_fpath_tpl = "data/{}/features/{}_{}.hdf5"
    frame_sample_len = 30

    split_negative_vids_fpath = "data/MSR-VTT/metadata/neg_vids_{}.json"

    num_workers = 2


class VisualEncoderConfig:
    app_feat, app_feat_size = 'ResNet101', 2048
    mot_feat, mot_feat_size = '3DResNext101', 2048
    feat_size = app_feat_size + mot_feat_size


class PhraseEncoderConfig:
    SA_num_layers = 1; assert SA_num_layers == 1
    SA_num_heads = 1; assert SA_num_heads == 1
    SA_dim_k = 32
    SA_dim_v = 32
    SA_dim_inner = 512
    SA_dropout = 0.1


class DecoderConfig:
    sem_align_hidden_size = 512
    sem_attn_hidden_size = 512
    rnn_num_layers = 1
    rnn_hidden_size = 512
    max_teacher_forcing_ratio = 1.0
    min_teacher_forcing_ratio = 1.0


class Config:
    seed = 0xd3853c

    corpus = 'MSVD'; assert corpus in [ 'MSVD', 'MSR-VTT' ]

    vocab = VocabConfig
    loader = {
        'MSVD': MSVDLoaderConfig,
        'MSR-VTT': MSRVTTLoaderConfig,
    }[corpus]
    vis_encoder = VisualEncoderConfig
    phr_encoder = PhraseEncoderConfig
    decoder = DecoderConfig

    """ Optimization """
    epochs = {
        'MSVD': 20,
        'MSR-VTT': 16,
    }[corpus]
    batch_size = {
        'MSVD': 200,
        'MSR-VTT': 50,
    }[corpus]
    lr = {
        'MSVD': 0.0007,
        'MSR-VTT': 0.0007,
    }[corpus]
    gradient_clip = 5.0 # None if not used
    PS_threshold = 0.2
    CA_lambda = 0.16

    """ Evaluation """
    metrics = [ 'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L' ]

    """ Log """
    exp_id = "SGN/{}".format(corpus)
    log_dpath = os.path.join("logs", exp_id)
    ckpt_fpath_tpl = os.path.join("checkpoints", exp_id, "{}.ckpt")

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/cross_entropy"
    tx_train_contrastive_attention_loss = "loss/train/contrastive_attention"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/cross_entropy"
    tx_val_contrastive_attention_loss = "loss/val/contrastive_attention"
    tx_lr = "params/lr"
    tx_teacher_forcing_ratio = "params/teacher_forcing_ratio"

