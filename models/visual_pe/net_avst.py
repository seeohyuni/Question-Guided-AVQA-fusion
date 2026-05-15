import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, maxlen=512):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(maxlen, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_length = x.size(1)
        if seq_length > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_length} exceeds positional encoding "
                f"maxlen {self.pe.size(1)}."
            )
        pe = self.pe[:, :seq_length, :].to(dtype=x.dtype)
        return self.dropout(x + pe)


class AVQA_Fusion_Net(nn.Module):

    def __init__(
        self,
        audio_dim=1024,
        num_answers=12,
        max_time_steps=512,
        positional_dropout=0.0,
    ):
        super(AVQA_Fusion_Net, self).__init__()

        self.fc_a1 = nn.Linear(audio_dim, 512)
        self.fc_a2 = nn.Linear(512, 512)
        self.visual_pos_enc = PositionalEncoding(
            d_model=512,
            dropout=positional_dropout,
            maxlen=max_time_steps,
        )

        # 1. Cross attention: query=f_q, key/value=f_v.
        self.attn_vq1 = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear_vq1_1 = nn.Linear(512, 512)
        self.dropout_vq1_1 = nn.Dropout(0.1)
        self.linear_vq1_2 = nn.Linear(512, 512)
        self.dropout_vq1_2 = nn.Dropout(0.1)
        self.norm_vq1 = nn.LayerNorm(512)

        # 3. Cross attention: query=f_VQ, key/value=v'.
        self.attn_vq2 = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear_vq2_1 = nn.Linear(512, 512)
        self.dropout_vq2_1 = nn.Dropout(0.1)
        self.linear_vq2_2 = nn.Linear(512, 512)
        self.dropout_vq2_2 = nn.Dropout(0.1)
        self.norm_vq2 = nn.LayerNorm(512)

        # 4. Cross attention: query=f_q, key/value=f_VQ''.
        self.attn_q_vq = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear_qvq_1 = nn.Linear(512, 512)
        self.dropout_qvq_1 = nn.Dropout(0.1)
        self.linear_qvq_2 = nn.Linear(512, 512)
        self.dropout_qvq_2 = nn.Dropout(0.1)
        self.norm_qvq = nn.LayerNorm(512)

        # 1. Cross attention: query=f_q, key/value=f_a.
        self.attn_aq1 = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear_aq1_1 = nn.Linear(512, 512)
        self.dropout_aq1_1 = nn.Dropout(0.1)
        self.linear_aq1_2 = nn.Linear(512, 512)
        self.dropout_aq1_2 = nn.Dropout(0.1)
        self.norm_aq1 = nn.LayerNorm(512)

        # 3. Cross attention: query=f_AQ, key/value=a'.
        self.attn_aq2 = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear_aq2_1 = nn.Linear(512, 512)
        self.dropout_aq2_1 = nn.Dropout(0.1)
        self.linear_aq2_2 = nn.Linear(512, 512)
        self.dropout_aq2_2 = nn.Dropout(0.1)
        self.norm_aq2 = nn.LayerNorm(512)

        # 4. Cross attention: query=f_q, key/value=f_AQ''.
        self.attn_q_aq = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.linear_qaq_1 = nn.Linear(512, 512)
        self.dropout_qaq_1 = nn.Dropout(0.1)
        self.linear_qaq_2 = nn.Linear(512, 512)
        self.dropout_qaq_2 = nn.Dropout(0.1)
        self.norm_qaq = nn.LayerNorm(512)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_answers),
        )


    def forward(self, audio, visual_posi, question):
        """
        audio:       [B, T, audio_dim]
        visual_posi: [B, T, 512, H, W]
        question:    [B, 512] or [B, 1, 512]
        """
        f_q = question.float()
        if f_q.dim() > 2:
            f_q = f_q.squeeze(1)

        B, T, visual_channels, H, W = visual_posi.size()

        # Audio feature.
        f_a = F.relu(self.fc_a1(audio))
        f_a = self.fc_a2(f_a)  # [B, T, 512]

        _, _, audio_channels = f_a.size()
        if visual_channels != audio_channels:
            raise ValueError(
                "visual_posi channel dimension must match the projected audio/question "
                f"dimension 512, but got {visual_channels}."
            )

        C = audio_channels
        BT = B * T
        if f_q.size(-1) != C:
            raise ValueError(
                f"question feature dimension must be {C}, but got {f_q.size(-1)}."
            )

        if T > self.visual_pos_enc.pe.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds positional encoding "
                f"maxlen {self.visual_pos_enc.pe.size(1)}."
            )
        visual_time_pe = self.visual_pos_enc.pe[:, :T, :].to(
            device=visual_posi.device,
            dtype=visual_posi.dtype,
        )
        visual_time_pe = visual_time_pe.unsqueeze(-1).unsqueeze(-1)
        visual_posi = self.visual_pos_enc.dropout(visual_posi + visual_time_pe)

        # Original visual feature f_v.
        f_v = visual_posi.reshape(BT, C, H, W)      # [B*T, C, H, W]
        f_v_seq = f_v.flatten(2).permute(2, 0, 1)  # [H*W, B*T, C]
        f_v_seq = F.normalize(f_v_seq, dim=-1)

        # Original question feature expanded to each visual time step.
        f_q_bt = f_q.unsqueeze(1).expand(B, T, C)
        f_q_bt = f_q_bt.reshape(BT, C).unsqueeze(0)  # [1, B*T, C]

        # 1. Visual original feature f_v and question original feature f_q:
        # query=f_q, key/value=f_v => f_VQ.
        vq1_query = f_q_bt
        f_VQ_att = self.attn_vq1(
            vq1_query,
            f_v_seq,
            f_v_seq,
            attn_mask=None,
            key_padding_mask=None,
        )[0]  # [1, B*T, C]
        src = self.linear_vq1_2(
            self.dropout_vq1_1(F.relu(self.linear_vq1_1(f_VQ_att)))
        )
        f_VQ = vq1_query + self.dropout_vq1_2(src)
        f_VQ = self.norm_vq1(f_VQ)

        # 2. Channel attention:
        # GAP -> ReLU -> Sigmoid -> channel-wise multiplication with original f_v.
        channel_att = F.adaptive_avg_pool2d(f_v, 1)  # [B*T, C, 1, 1]
        channel_att = F.relu(channel_att)
        channel_att = torch.sigmoid(channel_att)

        v_prime = f_v * channel_att                 # [B*T, C, H, W]
        v_prime_seq = v_prime.flatten(2).permute(2, 0, 1)
        v_prime_seq = F.normalize(v_prime_seq, dim=-1)  # [H*W, B*T, C]

        # 3. f_VQ and v':
        # query=f_VQ, key/value=v' => f_VQ''.
        vq2_query = f_VQ
        f_VQ_2_att = self.attn_vq2(
            vq2_query,
            v_prime_seq,
            v_prime_seq,
            attn_mask=None,
            key_padding_mask=None,
        )[0]  # [1, B*T, C]
        src = self.linear_vq2_2(
            self.dropout_vq2_1(F.relu(self.linear_vq2_1(f_VQ_2_att)))
        )
        f_VQ_2 = vq2_query + self.dropout_vq2_2(src)
        f_VQ_2 = self.norm_vq2(f_VQ_2)

        # 4. f_VQ'' and f_q:
        # query=f_q, key/value=f_VQ'' => final visual-question feature.
        f_VQ_2_time = f_VQ_2.squeeze(0).reshape(B, T, C)
        f_VQ_2_time = f_VQ_2_time.permute(1, 0, 2)  # [T, B, C]

        f_q_seq = f_q.unsqueeze(0)                  # [1, B, C]
        qvq_query = f_q_seq
        visual_feat_att = self.attn_q_vq(
            qvq_query,
            f_VQ_2_time,
            f_VQ_2_time,
            attn_mask=None,
            key_padding_mask=None,
        )[0]  # [1, B, C]
        src = self.linear_qvq_2(
            self.dropout_qvq_1(F.relu(self.linear_qvq_1(visual_feat_att)))
        )
        visual_feat = qvq_query + self.dropout_qvq_2(src)
        visual_feat = self.norm_qvq(visual_feat)
        visual_feat = visual_feat.squeeze(0)        # [B, C]

        # 1. Audio original feature f_a and question original feature f_q:
        # query=f_q, key/value=f_a => f_AQ.
        f_a_seq = f_a.permute(1, 0, 2)              # [T, B, C]
        f_a_seq = F.normalize(f_a_seq, dim=-1)

        aq1_query = f_q_seq
        f_AQ_att = self.attn_aq1(
            aq1_query,
            f_a_seq,
            f_a_seq,
            attn_mask=None,
            key_padding_mask=None,
        )[0]  # [1, B, C]
        src = self.linear_aq1_2(
            self.dropout_aq1_1(F.relu(self.linear_aq1_1(f_AQ_att)))
        )
        f_AQ = aq1_query + self.dropout_aq1_2(src)
        f_AQ = self.norm_aq1(f_AQ)

        # 2. Audio channel attention:
        # temporal GAP -> ReLU -> Sigmoid -> channel-wise multiplication with original f_a.
        audio_channel_att = f_a.mean(dim=1, keepdim=True)  # [B, 1, C]
        audio_channel_att = F.relu(audio_channel_att)
        audio_channel_att = torch.sigmoid(audio_channel_att)

        a_prime = f_a * audio_channel_att                  # [B, T, C]
        a_prime_seq = a_prime.permute(1, 0, 2)             # [T, B, C]
        a_prime_seq = F.normalize(a_prime_seq, dim=-1)

        # 3. f_AQ and a':
        # query=f_AQ, key/value=a' => f_AQ''.
        aq2_query = f_AQ
        f_AQ_2_att = self.attn_aq2(
            aq2_query,
            a_prime_seq,
            a_prime_seq,
            attn_mask=None,
            key_padding_mask=None,
        )[0]  # [1, B, C]
        src = self.linear_aq2_2(
            self.dropout_aq2_1(F.relu(self.linear_aq2_1(f_AQ_2_att)))
        )
        f_AQ_2 = aq2_query + self.dropout_aq2_2(src)
        f_AQ_2 = self.norm_aq2(f_AQ_2)

        # 4. f_AQ'' and f_q:
        # query=f_q, key/value=f_AQ'' => final audio-question feature.
        qaq_query = f_q_seq
        audio_feat_att = self.attn_q_aq(
            qaq_query,
            f_AQ_2,
            f_AQ_2,
            attn_mask=None,
            key_padding_mask=None,
        )[0]  # [1, B, C]
        src = self.linear_qaq_2(
            self.dropout_qaq_1(F.relu(self.linear_qaq_1(audio_feat_att)))
        )
        audio_feat = qaq_query + self.dropout_qaq_2(src)
        audio_feat = self.norm_qaq(audio_feat)
        audio_feat = audio_feat.squeeze(0)          # [B, C]

        fused = torch.cat((visual_feat, audio_feat), dim=-1)
        out_qa = self.classifier(fused)

        return out_qa
