#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from models.bilstm_crf import BiLSTMCRFParser
from utils.vocab import Vocab
from utils.data_loader import read_conllu, batchify
from utils.metrics import bio_f1

# ======= 标签集合（不包含 <pad>/<unk>）=======
BIO_TAGS = ["O", "B-NP", "I-NP", "B-ATTR", "I-ATTR"]
REL_TAGS = ["NONE", "on", "under", "left_of", "right_of", "near"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vocabs(datasets):
    """词表含 <pad>/<unk>；标签表不含，严格等于上面的 BIO_TAGS/REL_TAGS"""
    tok_vocab = Vocab(add_base_specials=True)     # tokens 需要 PAD/UNK
    bio_vocab = Vocab(add_base_specials=False)    # 标签表不要 PAD/UNK
    rel_vocab = Vocab(add_base_specials=False)

    for t in BIO_TAGS:
        bio_vocab.add(t)
    for r in REL_TAGS:
        rel_vocab.add(r)

    for ds in datasets:
        for sent in ds:
            for w in sent["tokens"]:
                tok_vocab.add(w)

    return tok_vocab, bio_vocab, rel_vocab


def has_positive(ex):
    return any(tag != "O" for tag in ex.get("bio", []))


def ratio_non_o(ds):
    total = sum(len(ex["bio"]) for ex in ds) or 1
    pos = sum(1 for ex in ds for t in ex["bio"] if t != "O")
    return pos / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/sample")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # ------- 配置 -------
    cfg = yaml.safe_load(Path(args.config).read_text())
    lr = float(cfg.get("lr", 3e-3))
    dropout = float(cfg.get("dropout", 0.0))
    max_len = int(cfg.get("max_len", 128))
    lambda_rel = float(cfg.get("lambda_rel", 0.0))  # 先设 0.0 只训 BIO，收敛后再开

    # ------- 读数据 -------
    data_dir = Path(args.data_dir)
    train = read_conllu(data_dir / "train.conllu")
    print(train[0])
    dev   = read_conllu(data_dir / "dev.conllu")

    # 过滤全 O（训练集），并上采样含正样本的句子
    pos_train = [ex for ex in train if has_positive(ex)]
    print(f"[INFO] train total={len(train)}, positives={len(pos_train)}, dev={len(dev)}")
    print(f"[INFO] non-O ratio -> train={ratio_non_o(train):.3f} (pos={ratio_non_o(pos_train):.3f})  dev={ratio_non_o(dev):.3f}")

    # 上采样：把含正样本的句子再复制一份（可按需调倍数）
    train = train + pos_train
    print(f"[INFO] after oversample, train size={len(train)}")

    # ------- 词表 -------
    tok_vocab, bio_vocab, rel_vocab = build_vocabs([train, dev])
    print(f"[INFO] label sizes -> BIO: {len(bio_vocab)}  REL: {len(rel_vocab)}  (expect 5 / 6或7)")

    # ------- 模型 -------
    model = BiLSTMCRFParser(
        vocab_size=len(tok_vocab),
        bio_label_count=len(bio_vocab),
        rel_label_count=len(rel_vocab),
        emb_dim=int(cfg.get("emb_dim", 128)),
        hid_dim=int(cfg.get("hid_dim", 256)),
        dropout=dropout,
    )
    opt = optim.Adam(model.parameters(), lr=lr)

    # ------- 损失：类别加权 CE（降低 O 权重） -------
    O_W, NONO_W = 0.2, 1.0
    cls_weights = torch.ones(len(bio_vocab), dtype=torch.float)
    o_idx = bio_vocab.id("O")
    for i in range(len(bio_vocab)):
        cls_weights[i] = NONO_W
    cls_weights[o_idx] = O_W
    ce_loss_fn = nn.CrossEntropyLoss(weight=cls_weights, reduction="sum")

    # ------- 训练策略：Warm-up CE -> 线性引入 CRF -------
    warmup_epochs = int(cfg.get("warmup_epochs", 10))  # 前 N 个 epoch 只用 CE
    ramp_epochs   = int(cfg.get("ramp_epochs", 10))    # 之后 M 个 epoch 线性提高 CRF 权重
    beta_crf_max  = float(cfg.get("beta_crf_max", 1.0))
    alpha_ce = 1.0

    # ------- 训练 -------
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train)

        for i in range(0, len(train), int(cfg.get("batch_size", 16))):
            batch = train[i:i + int(cfg.get("batch_size", 16))]
            pack = batchify(batch, tok_vocab, bio_vocab, rel_vocab, max_len=max_len)

            # 前向：拿 emissions 做 token 级 CE（不传 gold 给 CRF）
            _, emissions, _ = model(pack["ids"], pack["mask"], None, None)

            # token CE
            B, T, C = emissions.shape
            logits = emissions.view(B * T, C)
            gold   = pack["bio"].view(B * T)
            mask   = pack["mask"].view(B * T).float()
            ce = ce_loss_fn(logits, gold)
            valid = mask.sum().clamp(min=1.0)
            ce = ce / valid

            # 计算当下 CRF 权重
            if epoch < warmup_epochs:
                beta_crf = 0.0
            else:
                step = min(epoch - warmup_epochs + 1, ramp_epochs)
                beta_crf = beta_crf_max * (step / max(1, ramp_epochs))

            # 关系损失（可选）
            rel_loss = 0.0
            if lambda_rel > 0.0 and any(len(x) > 0 for x in pack["rel_pairs"]):
                # 这里的关系头损失已经包含在 model 内部 forward 返回的 loss 中，
                # 我们单独再走一次只为拿到 CRF 序列损失；更简单些直接忽略关系损失也行
                pass

            # 加入 CRF 序列损失
            if beta_crf > 0:
                crf_loss, _, _ = model(pack["ids"], pack["mask"], pack["bio"], None)
                total = alpha_ce * ce + beta_crf * crf_loss
            else:
                total = alpha_ce * ce

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); opt.zero_grad()

        # ------- Dev 评估 -------
        model.eval()
        all_pred, all_gold = [], []
        non_o_gold = non_o_pred = 0
        with torch.no_grad():
            for i in range(0, len(dev), int(cfg.get("eval_batch_size", 32))):
                batch = dev[i:i + int(cfg.get("eval_batch_size", 32))]
                pack = batchify(batch, tok_vocab, bio_vocab, rel_vocab, max_len=max_len)

                # warm-up 用 argmax；之后用 CRF 解码
                if epoch < warmup_epochs:
                    _, emissions, _ = model(pack["ids"], pack["mask"], None, None)
                    argmax = emissions.argmax(-1)  # [B,T]
                    paths = []
                    for b in range(argmax.size(0)):
                        L = int(pack["mask"][b].sum().item())
                        paths.append(argmax[b, :L].tolist())
                else:
                    _, emissions, _ = model(pack["ids"], pack["mask"], None, None)
                    paths = model.decode(emissions, pack["mask"])

                for b, p in enumerate(paths):
                    L = int(pack["mask"][b].sum().item())
                    gold = [bio_vocab.token(int(x)) for x in pack["bio"][b][:L].tolist()]
                    pred = [bio_vocab.token(int(x)) for x in p[:L]]
                    all_gold.append(gold); all_pred.append(pred)
                    non_o_gold += sum(t != "O" for t in gold)
                    non_o_pred += sum(t != "O" for t in pred)

        f1 = bio_f1(all_pred, all_gold)
        print(f"[DEBUG] dev non-O gold={non_o_gold}  pred={non_o_pred}  beta_crf={beta_crf:.2f}")
        print(f"Epoch {epoch+1}: dev BIO F1={f1:.3f}")
        model.train()

    # ------- 保存产物 -------
    torch.save(model.state_dict(), "parser.ckpt")
    art = Path("artifacts"); art.mkdir(exist_ok=True)
    (art / "tok_vocab.txt").write_text("\n".join(tok_vocab.itos), encoding="utf-8")
    (art / "bio_tags.txt").write_text("\n".join(BIO_TAGS), encoding="utf-8")
    (art / "rel_tags.txt").write_text("\n".join(REL_TAGS), encoding="utf-8")
    print(f"Saved model to parser.ckpt and artifacts/. [BIO={len(BIO_TAGS)} / REL={len(REL_TAGS)}]")


if __name__ == "__main__":
    main()
