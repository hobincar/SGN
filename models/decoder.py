import torch
import torch.nn as nn

from models.attention import SemanticAlignment, SemanticAttention


class Decoder(nn.Module):
    def __init__(self, num_layers, vis_feat_size, feat_len, embedding_size, sem_align_hidden_size,
                 sem_attn_hidden_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.vis_feat_size = vis_feat_size
        self.feat_len = feat_len
        self.embedding_size = embedding_size
        self.sem_align_hidden_size = sem_align_hidden_size
        self.sem_attn_hidden_size = sem_attn_hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.semantic_alignment = SemanticAlignment(
            query_size=self.embedding_size,
            feat_size=self.vis_feat_size,
            bottleneck_size=self.sem_align_hidden_size)
        self.semantic_attention = SemanticAttention(
            query_size=self.hidden_size,
            key_size=self.embedding_size + self.vis_feat_size,
            bottleneck_size=self.sem_attn_hidden_size)

        self.rnn = nn.LSTM(
            input_size=self.vis_feat_size + self.embedding_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def get_last_hidden(self, hidden):
        last_hidden = hidden[0]
	last_hidden = last_hidden.view(self.num_layers, 1, last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1), last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden

    def forward(self, embedded, hidden, vis_feats, phr_feats, phr_masks):
        last_hidden = self.get_last_hidden(hidden)
        semantic_group_feats, semantic_align_weights, semantic_align_logits = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=vis_feats)
        feat, semantic_attn_weights, semantic_attn_logits = self.semantic_attention(
            query=last_hidden,
            keys=semantic_group_feats,
            values=semantic_group_feats,
            masks=phr_masks)

        feat = torch.cat((
            feat,
            embedded), dim=1)
        output, hidden = self.rnn(feat[None, :, :], hidden)

        output = output.squeeze(0)
        output = self.out(output)
        output = torch.log_softmax(output, dim=1)
        return output, hidden, ( semantic_align_weights, semantic_attn_weights ), \
               ( semantic_align_logits, semantic_attn_logits )

