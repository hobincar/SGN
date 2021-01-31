from collections import defaultdict
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm

from loader.MSRVTT import MSRVTT
from loader.MSVD import MSVD
from models.decoder import Decoder
from models.semantic_grouping_network import SemanticGroupingNetwork as SGN
from models.transformer.Models import Encoder as PhraseEncoder
from models.visual_encoder import VisualEncoder
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [ [] for _ in range(self.num_losses) ]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [ 0. for _ in range(self.num_losses) ]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def build_loaders(config):
    if config.corpus == "MSVD":
        corpus = MSVD(config)
    elif config.corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, config.loader.min_count))
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab


def build_model(config, vocab):
    visual_encoder = VisualEncoder(
        app_feat=config.vis_encoder.app_feat,
        mot_feat=config.vis_encoder.mot_feat,
        app_input_size=config.vis_encoder.app_feat_size,
        mot_input_size=config.vis_encoder.mot_feat_size,
        app_output_size=config.vocab.embedding_size,
        mot_output_size=config.vocab.embedding_size)

    phrase_encoder = PhraseEncoder(
        len_max_seq=config.loader.max_caption_len + 2,
        d_word_vec=config.vocab.embedding_size,
        n_layers=config.phr_encoder.SA_num_layers,
        n_head=config.phr_encoder.SA_num_heads,
        d_k=config.phr_encoder.SA_dim_k,
        d_v=config.phr_encoder.SA_dim_v,
        d_model=config.vocab.embedding_size,
        d_inner=config.phr_encoder.SA_dim_inner,
        dropout=config.phr_encoder.SA_dropout)

    decoder = Decoder(
        num_layers=config.decoder.rnn_num_layers,
        vis_feat_size=2 * config.vocab.embedding_size,
        feat_len=config.loader.frame_sample_len,
        embedding_size=config.vocab.embedding_size,
        sem_align_hidden_size=config.decoder.sem_align_hidden_size,
        sem_attn_hidden_size=config.decoder.sem_attn_hidden_size,
        hidden_size=config.decoder.rnn_hidden_size,
        output_size=vocab.n_vocabs)

    model = SGN(visual_encoder, phrase_encoder, decoder, config.loader.max_caption_len, vocab,
                config.PS_threshold)
    return model


def parse_batch(batch):
    pos, neg = batch
    pos_vids, pos_vis_feats, pos_captions = pos
    neg_vids, neg_vis_feats, neg_captions = neg

    for model in pos_vis_feats:
        pos_vis_feats[model] = pos_vis_feats[model].cuda()
        neg_vis_feats[model] = neg_vis_feats[model].cuda()
    pos_captions = pos_captions.long().cuda()
    neg_captions = neg_captions.long().cuda()

    pos = ( pos_vids, pos_vis_feats, pos_captions )
    neg = ( neg_vids, neg_vis_feats, neg_captions )
    return pos, neg


def train(e, model, optimizer, train_iter, vocab, teacher_forcing_ratio, CA_lambda, gradient_clip):
    model.train()

    loss_checker = LossChecker(3)
    PAD_idx = vocab.word2idx['<PAD>']
    pgbar = tqdm(train_iter)
    for batch in pgbar:
        ( _, pos_vis_feats, pos_captions ), ( _, neg_vis_feats, neg_captions ) = parse_batch(batch)
        optimizer.zero_grad()
        output, contrastive_attention = model(
            pos_vis_feats, pos_captions, neg_vis_feats, neg_captions, teacher_forcing_ratio)
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        pos_captions[1:].contiguous().view(-1),
                                        ignore_index=PAD_idx)
        contrastive_attention_loss = F.binary_cross_entropy_with_logits(
            contrastive_attention.mean(dim=0), torch.cuda.FloatTensor([ 1, 0 ]))

        loss = cross_entropy_loss \
               + CA_lambda * contrastive_attention_loss
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item(), cross_entropy_loss.item(), contrastive_attention_loss.item())
        pgbar.set_description("[Epoch #{0}] loss: {2:.3f} = CE {3:.3f} + CA {1} * {4:.3f}".format(
            e, CA_lambda, *loss_checker.mean(last=10)))

    total_loss, cross_entropy_loss, contrastive_attention_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'contrastive_attention': contrastive_attention_loss,
    }
    return loss


def evaluate(model, val_iter, vocab, CA_lambda):
    model.eval()

    loss_checker = LossChecker(3)
    PAD_idx = vocab.word2idx['<PAD>']
    for batch in val_iter:
        ( _, pos_vis_feats, pos_captions ), ( _, neg_vis_feats, neg_captions ) = parse_batch(batch)
        output, contrastive_attention = model(
            pos_vis_feats, pos_captions, neg_vis_feats, neg_captions, teacher_forcing_ratio=0.)
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        pos_captions[1:].contiguous().view(-1),
                                        ignore_index=PAD_idx)
        contrastive_attention_loss = F.binary_cross_entropy_with_logits(
            contrastive_attention.mean(dim=0), torch.cuda.FloatTensor([ 1, 0 ]))

        loss = cross_entropy_loss \
               + CA_lambda * contrastive_attention_loss
        loss_checker.update(loss.item(), cross_entropy_loss.item(), contrastive_attention_loss.item())

    total_loss, cross_entropy_loss, contrastive_attention_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'contrastive_attention': contrastive_attention_loss,
    }
    return loss


def build_YOLO_iter(data_iter, batch_size):
    score_dataset = {}
    for batch in iter(data_iter):
        ( vids, feats, _ ), _ = parse_batch(batch)
        for i, vid in enumerate(vids):
            feat = {}
            for model in feats:
                feat[model] = feats[model][i]
            if vid not in score_dataset:
                score_dataset[vid] = feat

    score_iter = []
    vids = score_dataset.keys()
    feats = score_dataset.values()
    while len(vids) > 0:
        vids_batch = vids[:batch_size]
        feats_batch = defaultdict(lambda: [])
        for feat in feats[:batch_size]:
            for model, f in feat.items():
                feats_batch[model].append(f)
        for model in feats_batch:
            feats_batch[model] = torch.stack(feats_batch[model], dim=0)
        yield ( vids_batch, feats_batch )
        vids = vids[batch_size:]
        feats = feats[batch_size:]


def score(model, data_iter, vocab):
    def build_refs(data_iter):
        vid2idx = {}
        refs = {}
        for idx, (vid, captions) in enumerate(data_iter.captions.items()):
            vid2idx[vid] = idx
            refs[idx] = captions
        return refs, vid2idx

    model.eval()

    PAD_idx = vocab.word2idx['<PAD>']
    YOLO_iter = build_YOLO_iter(data_iter, batch_size=32)
    refs, vid2idx = build_refs(data_iter)

    hypos = {}
    for vids, feats in tqdm(YOLO_iter, desc='score'):
        captions = model.describe(feats)
        captions = [ idxs_to_sentence(caption, vocab.idx2word, vocab.word2idx['<EOS>']) for caption in captions ]
        for vid, caption in zip(vids, captions):
            hypos[vid2idx[vid]] = [ caption ]
    scores = calc_scores(refs, hypos)
    return scores, refs, hypos, vid2idx


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def save_checkpoint(ckpt_fpath, epoch, model, optimizer):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save(model.state_dict(), ckpt_fpath)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

