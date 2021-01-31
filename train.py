from __future__ import print_function

from tensorboardX import SummaryWriter
import torch
from config import Config as C

from utils import build_loaders, build_model, train, evaluate, score, get_lr, save_checkpoint, \
                  count_parameters, set_random_seed


def log_train(C, summary_writer, e, loss, lr, teacher_forcing_ratio, scores=None):
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_train_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_train_contrastive_attention_loss, loss['contrastive_attention'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    summary_writer.add_scalar(C.tx_teacher_forcing_ratio, teacher_forcing_ratio, e)
    print("[TRAIN] loss: {} (= CE {} + CA {})".format(
        loss['total'], loss['cross_entropy'], loss['contrastive_attention']))
    if scores is not None:
      for metric in C.metrics:
          summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
      print("scores: {}".format(scores))


def log_val(C, summary_writer, e, loss, scores=None):
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_val_cross_entropy_loss, loss['cross_entropy'], e)
    summary_writer.add_scalar(C.tx_val_contrastive_attention_loss, loss['contrastive_attention'], e)
    print("[VAL] loss: {} (= CE {} + CA {})".format(
        loss['total'], loss['cross_entropy'], loss['contrastive_attention']))
    if scores is not None:
        for metric in C.metrics:
            summary_writer.add_scalar("VAL SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))


def get_teacher_forcing_ratio(max_teacher_forcing_ratio, min_teacher_forcing_ratio, epoch, max_epoch):
    x = 1 - float(epoch - 1) / (max_epoch - 1)
    a = max_teacher_forcing_ratio - min_teacher_forcing_ratio
    b = min_teacher_forcing_ratio
    return a * x + b


def main():
    set_random_seed(C.seed)

    summary_writer = SummaryWriter(C.log_dpath)

    train_iter, val_iter, test_iter, vocab = build_loaders(C)

    model = build_model(C, vocab)
    print("#params: ", count_parameters(model))
    model = model.cuda()

    optimizer = torch.optim.Adamax(model.parameters(), lr=C.lr, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, C.epochs, eta_min=0, last_epoch=-1)

    best_val_scores = { 'CIDEr': -1. }
    for e in range(1, C.epochs + 1):
        print()
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)

        """ Train """
        teacher_forcing_ratio = get_teacher_forcing_ratio(C.decoder.max_teacher_forcing_ratio,
                                                          C.decoder.min_teacher_forcing_ratio,
                                                          e, C.epochs)
        train_loss = train(e, model, optimizer, train_iter, vocab, teacher_forcing_ratio,
                           C.CA_lambda, C.gradient_clip)
        log_train(C, summary_writer, e, train_loss, get_lr(optimizer), teacher_forcing_ratio)
        lr_scheduler.step()

        """ Validation """
        val_loss = evaluate(model, val_iter, vocab, C.CA_lambda)
        val_scores, _, _, _ = score(model, val_iter, vocab)
        log_val(C, summary_writer, e, val_loss, val_scores)

        if val_scores['CIDEr'] > best_val_scores['CIDEr']:
            best_val_scores = val_scores
            best_epoch = e
            best_model = model

        print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
        save_checkpoint(ckpt_fpath, e, model, optimizer)

    """ Test """
    test_scores, _, _, _ = score(best_model, test_iter, vocab)
    for metric in C.metrics:
        summary_writer.add_scalar("BEST SCORE/{}".format(metric), test_scores[metric], best_epoch)
    best_ckpt_fpath = C.ckpt_fpath_tpl.format("best")
    save_checkpoint(best_ckpt_fpath, best_epoch, best_model, optimizer)


if __name__ == "__main__":
    main()

