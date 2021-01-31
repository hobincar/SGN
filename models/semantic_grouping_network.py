import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.transformer.Constants import PAD as SelfAttention_PAD


class SemanticGroupingNetwork(nn.Module):
    def __init__(self, vis_encoder, phr_encoder, decoder, max_caption_len, vocab, PS_threshold):
        super(SemanticGroupingNetwork, self).__init__()
        self.vis_encoder = vis_encoder
        self.phr_encoder = phr_encoder
        self.decoder = decoder
        self.max_caption_len = max_caption_len
        self.vocab = vocab
        self.PS_threshold = PS_threshold

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.vocab.embedding_weights), freeze=False,
                                                      padding_idx=self.vocab.word2idx['<PAD>'])

    def get_rnn_init_hidden(self, batch_size, num_layers, hidden_size):
        return (
            torch.zeros(num_layers, batch_size, hidden_size).cuda(),
            torch.zeros(num_layers, batch_size, hidden_size).cuda())

    def forward_visual_encoder(self, vis_feats):
        app_feats = vis_feats[self.vis_encoder.app_feat]
        mot_feats = vis_feats[self.vis_encoder.mot_feat]
        vis_feats = self.vis_encoder(app_feats, mot_feats)
        return vis_feats

    def forward_decoder(self, batch_size, vocab_size, pos_vis_feats, pos_captions, neg_vis_feats, neg_captions,
                        teacher_forcing_ratio):
        caption_EOS_table = pos_captions == self.vocab.word2idx['<EOS>']
        caption_PAD_table = pos_captions == self.vocab.word2idx['<PAD>']
        caption_end_table = ~(~caption_EOS_table * ~caption_PAD_table)

        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.hidden_size)
        outputs = Variable(torch.zeros(self.max_caption_len + 2, batch_size, vocab_size)).cuda()

        caption_lens = torch.zeros(batch_size).cuda().long()
        contrastive_attention_list = []
        output = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<SOS>']))
        for t in range(1, self.max_caption_len + 2):
            embedded = self.embedding(output.view(1, -1)).squeeze(0)
            if t == 1:
                embedded_list = embedded[:, None, :]
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            elif t == 2:
                embedded_list = embedded[:, None, :]
                caption_lens += 1
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            else:
                embedded_list = torch.cat([ embedded_list, embedded[:, None, :]  ], dim=1)
                caption_lens += ((output.long().squeeze() != self.vocab.word2idx['<PAD>']) * \
                                 (output.long().squeeze() != self.vocab.word2idx['<EOS>'])).long()
                src_pos = torch.arange(1, t).repeat(batch_size, 1).cuda()
                src_pos[src_pos > caption_lens[:, None]] = SelfAttention_PAD
            phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
            phr_attns = phr_attns[0]

            if t >= 2:
                A = torch.bmm(phr_attns, phr_attns.transpose(1, 2))
                A_mask = torch.eye(t-1, t-1).cuda().bool()
                A.masked_fill_(A_mask, 0)
                A_sum = A.sum(dim=2)

                indices = (A >= self.PS_threshold).nonzero() # Obtain indices of phrase pairs that
                                                                             # are highly overlapped with each other
                indices = indices[indices[:, 1] < indices[:, 2]] # Leave only the upper triangle to prevent duplication

                phr_masks = torch.zeros_like(A_sum).bool()
                if len(indices) > 0:
                    redundancy_masks = torch.zeros_like(phr_masks).long()
                    indices_b = indices[:, 0]
                    indices_i = indices[:, 1]
                    indices_j = indices[:, 2]
                    indices_ij = torch.stack(( indices_i, indices_j ), dim=1)
                    A_sum_i = A_sum[indices_b, indices_i]
                    A_sum_j = A_sum[indices_b, indices_j]
                    A_sum_ij = torch.stack(( A_sum_i, A_sum_j ), dim=1)
                    _, i_or_j = A_sum_ij.max(dim=1)
                    i_or_j = i_or_j.bool()
                    indices_i_or_j = torch.zeros_like(indices_b)
                    indices_i_or_j[i_or_j] = indices_j[i_or_j]
                    indices_i_or_j[~i_or_j] = indices_i[~i_or_j]
                    redundancy_masks[indices_b, indices_i_or_j] = 1 # Mask phrases that are more redundant
                                                                    # than their counterpart
                    phr_masks = redundancy_masks > 0.5
            else:
                phr_masks = None

            output, hidden, ( sem_align_weights, _ ), ( sem_align_logits, _ ) = self.decoder(
                embedded, hidden, pos_vis_feats, phr_feats, phr_masks)

            # Calculate the Contrastive Attention loss
            if t >= 2:
                pos_sem_align_logits = sem_align_logits
                _, _, neg_sem_align_logits = self.decoder.semantic_alignment(phr_feats, neg_vis_feats)
                pos_align_logit = pos_sem_align_logits.sum(dim=2)
                neg_align_logit = neg_sem_align_logits.sum(dim=2)
                pos_align_logit = pos_align_logit[~caption_end_table[t-1]]
                neg_align_logit = neg_align_logit[~caption_end_table[t-1]]
                align_logits = torch.stack([ pos_align_logit, neg_align_logit ], dim=2)
                phr_masks_for_logits = phr_masks[~caption_end_table[t-1]]
                align_logits = align_logits.view(-1, 2)[~phr_masks_for_logits.view(-1)]
                contrastive_attention_list.append(align_logits)

            # Early stop
            if torch.all(caption_end_table[t]).item():
                break

            # Choose the next word
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(pos_captions.data[t] if is_teacher else top1).cuda()

        contrastive_attention = torch.cat(contrastive_attention_list, dim=0)
        return outputs, contrastive_attention

    def forward(self, pos_vis_feats, pos_captions, neg_vis_feats, neg_captions, teacher_forcing_ratio=0.):
        batch_size = pos_vis_feats.values()[0].size(0)
        vocab_size = self.decoder.output_size

        pos_vis_feats = self.forward_visual_encoder(pos_vis_feats)
        neg_vis_feats = self.forward_visual_encoder(neg_vis_feats)
        captions, CA_logits = self.forward_decoder(batch_size, vocab_size, pos_vis_feats, pos_captions,
                                                   neg_vis_feats, neg_captions, teacher_forcing_ratio)
        return captions, CA_logits

    def describe(self, vis_feats):
        batch_size = vis_feats.values()[0].size(0)
        vocab_size = self.decoder.output_size

        vis_feats = self.forward_visual_encoder(vis_feats)
        captions = self.beam_search(batch_size, vocab_size, vis_feats)
        return captions

    def beam_search(self, batch_size, vocab_size, vis_feats, width=5):
        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.hidden_size)

        input_list = [ torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<SOS>']) ]
        hidden_list = [ hidden ]
        cum_prob_list = [ torch.ones(batch_size).cuda() ]
        cum_prob_list = [ torch.log(cum_prob) for cum_prob in cum_prob_list ]
        EOS_idx = self.vocab.word2idx['<EOS>']

        output_list = [ [[]] for _ in range(batch_size) ]
        for t in range(1, self.max_caption_len + 2):
            beam_output_list = []
            normalized_beam_output_list = []
            beam_hidden_list = ( [], [] )
            next_output_list = [ [] for _ in range(batch_size) ]

            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                caption_list = [ output_list[b][i] for b in range(batch_size) ]
                if t == 1:
                    words_list = input.transpose(0, 1)
                else:
                    words_list = torch.cuda.LongTensor(caption_list)

                embedded_list = self.embedding(words_list)
                if t == 1:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                elif t == 2:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                else:
                    caption_lens = torch.cuda.LongTensor([ [ idx.item() for idx in caption ].index(EOS_idx) if EOS_idx in [ idx.item() for idx in caption ] else t-1 for caption in caption_list ])
                    src_pos = torch.arange(1, t).repeat(batch_size, 1).cuda()
                    src_pos[src_pos > caption_lens[:, None]] = 0
                phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
                phr_attns = phr_attns[0]

                if t >= 2:
                    A = torch.bmm(phr_attns, phr_attns.transpose(1, 2))
                    A_mask = torch.eye(t-1, t-1).cuda().bool()
                    A.masked_fill_(A_mask, 0)
                    A_sum = A.sum(dim=2)

                    indices = (A >= self.PS_threshold).nonzero()
                    indices = indices[indices[:, 1] < indices[:, 2]] # Leave only the upper triangle to prevent duplication

                    phr_masks = torch.zeros_like(A_sum).bool()
                    if len(indices) > 0:
                        redundancy_masks = torch.zeros_like(phr_masks).long()
                        indices_b = indices[:, 0]
                        indices_i = indices[:, 1]
                        indices_j = indices[:, 2]
                        indices_ij = torch.stack(( indices_i, indices_j ), dim=1)
                        A_sum_i = A_sum[indices_b, indices_i]
                        A_sum_j = A_sum[indices_b, indices_j]
                        A_sum_ij = torch.stack(( A_sum_i, A_sum_j ), dim=1)
                        _, i_or_j = A_sum_ij.max(dim=1)
                        i_or_j = i_or_j.bool()
                        indices_i_or_j = torch.zeros_like(indices_b)
                        indices_i_or_j[i_or_j] = indices_j[i_or_j]
                        indices_i_or_j[~i_or_j] = indices_i[~i_or_j]
                        redundancy_masks[indices_b, indices_i_or_j] = 1 # Mask phrases that are more redundant
                                                                        # than their counterpart
                        phr_masks = redundancy_masks > 0.5
                else:
                    phr_masks = None

                embedded = self.embedding(input.view(1, -1)).squeeze(0)
                output, next_hidden, _, _ = self.decoder(embedded, hidden, vis_feats, phr_feats, phr_masks)

                EOS_mask = [ 1 if EOS_idx in [ idx.item() for idx in caption ] else 0 for caption in caption_list ]
                EOS_mask = torch.cuda.BoolTensor(EOS_mask)
                output[EOS_mask] = 0.

                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [ [ idx.item() for idx in caption ].index(EOS_idx) + 1 if EOS_idx in [ idx.item() for idx in caption ] else t for caption in caption_list ]
                caption_lens = torch.cuda.FloatTensor(caption_lens)
                normalizing_factor = ((5 + caption_lens) ** 1.6) / ((5 + 1) ** 1.6)
                normalized_output = output / normalizing_factor[:, None]
                normalized_beam_output_list.append(normalized_output)
                beam_hidden_list[0].append(next_hidden[0])
                beam_hidden_list[1].append(next_hidden[1])
            beam_output_list = torch.cat(beam_output_list, dim=1) # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:, :width] # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size # ( 100, width )

            topk_output_list = [ topk_output_index[:, i] for i in range(width) ] # width * ( 100, )
            topk_hidden_list = (
                [ [] for _ in range(width) ],
                [ [] for _ in range(width) ]) # 2 * width * (1, 100, 512)
            topk_cum_prob_list = [ [] for _ in range(width) ] # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    topk_hidden_list[0][k].append(beam_hidden_list[0][bi][:, i, :])
                    topk_hidden_list[1][k].append(beam_hidden_list[1][bi][:, i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [ oi ])
            output_list = next_output_list

            input_list = [ topk_output.unsqueeze(0) for topk_output in topk_output_list ] # width * ( 1, 100 )
            hidden_list = (
                [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[0] ],
                [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[1] ]) # 2 * width * ( 1, 100, 512 )
            hidden_list = [ ( hidden, context ) for hidden, context in zip(*hidden_list) ]
            cum_prob_list = [ torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in topk_cum_prob_list ] # width * ( 100, )

        SOS_idx = self.vocab.word2idx['<SOS>']
        outputs = [ [ SOS_idx ] + o[0] for o in output_list ]
        return outputs

