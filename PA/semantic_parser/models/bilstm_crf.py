import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRFParser(nn.Module):
    """
    BiLSTM-CRF for BIO tagging + a simple relation head.

    - Token embeddings -> BiLSTM -> emissions -> CRF (BIO tags)
    - Span pooling -> pairwise features -> MLP for relation classification
    """
    def __init__(self, vocab_size, bio_label_count, rel_label_count, emb_dim=128, hid_dim=256, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.emissions = nn.Linear(hid_dim, bio_label_count)
        self.crf = CRF(bio_label_count, batch_first=True)

        # Relation head: concat(h, t, h*t) -> MLP
        self.rel_mlp = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, rel_label_count)
        )

    def encode(self, input_ids, mask):
        x = self.emb(input_ids)
        h, _ = self.lstm(x)
        h = self.drop(h)
        return h

    def forward(self, input_ids, mask, gold_bio=None, rel_pairs=None):
        """
        Args:
            input_ids: [B, T]
            mask:      [B, T]  (1 for valid, 0 for pad)
            gold_bio:  [B, T]  int64
            rel_pairs: list of lists, length B
                       Each inner list has tuples: (h_start, h_end, t_start, t_end, rel_id)
        Returns:
            loss, emissions (for decoding), token_repr (for span pooling)
        """
        H = self.encode(input_ids, mask)
        emissions = self.emissions(H)
        loss = None

        if gold_bio is not None:
            loss_crf = -self.crf(emissions, gold_bio, mask=mask.bool(), reduction='mean')
            loss = loss_crf if loss is None else loss + loss_crf

        # Relation classification (toy): mean-pool span representations
        if rel_pairs is not None:
            feats, gts = [], []
            B = input_ids.size(0)
            for b in range(B):
                pairs = rel_pairs[b] if rel_pairs[b] is not None else []
                for (hs, he, ts, te, rel_id) in pairs:
                    hs = max(hs, 0); he = max(he, 0); ts = max(ts, 0); te = max(te, 0)
                    # clamp to sequence length
                    T = H.size(1)
                    hs, he = min(hs, T-1), min(he, T-1)
                    ts, te = min(ts, T-1), min(te, T-1)
                    h_span = H[b, hs:he+1].mean(dim=0) if he >= hs else H[b, hs]
                    t_span = H[b, ts:te+1].mean(dim=0) if te >= ts else H[b, ts]
                    z = torch.cat([h_span, t_span, h_span * t_span], dim=-1)
                    feats.append(z)
                    gts.append(rel_id)

            if feats:
                feats = torch.stack(feats, dim=0)
                logits = self.rel_mlp(feats)
                y = torch.tensor(gts, dtype=torch.long, device=logits.device)
                loss_rel = nn.CrossEntropyLoss()(logits, y)
                loss = loss + loss_rel if loss is not None else loss_rel

        return loss, emissions, H

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask.bool())
