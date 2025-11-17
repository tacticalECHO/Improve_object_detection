import re
from typing import List, Dict, Any
from pathlib import Path

def _flush_block(sents, cur_text, cur_tokens, cur_bio, cur_rels):
    if not cur_tokens:
        return
    # 对齐 BIO 长度到 tokens
    if not cur_bio:
        cur_bio = ["O"] * len(cur_tokens)
    elif len(cur_bio) != len(cur_tokens):
        want = len(cur_tokens)
        if len(cur_bio) > want:
            print(f"[WARN] BIO too long ({len(cur_bio)}->{want}); truncated: {cur_text[:60]}...")
            cur_bio = cur_bio[:want]
        else:
            print(f"[WARN] BIO too short ({len(cur_bio)}->{want}); padded with O: {cur_text[:60]}...")
            cur_bio = cur_bio + ["O"] * (want - len(cur_bio))

    sents.append({
        "text": cur_text,
        "tokens": cur_tokens,
        "bio": cur_bio,
        "rels": cur_rels
    })

def read_conllu(path: Path) -> List[Dict[str, Any]]:
    """
    预期块格式：
      # text = ...
      # BIO  = ...
      # RELS = (h_s,h_e,t_s,t_e,label) ; ...
      1\tAl\tAl\tX
      ...
    块之间建议有空行；若没有，遇到新的 '# text =' 也会自动 flush 前一块。
    """
    sents = []
    cur_tokens, cur_bio, cur_rels, cur_text = [], [], [], ""

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            _flush_block(sents, cur_text, cur_tokens, cur_bio, cur_rels)
            cur_tokens, cur_bio, cur_rels, cur_text = [], [], [], ""
            continue

        if line.startswith("# text ="):
            if cur_tokens or cur_bio or cur_rels:
                _flush_block(sents, cur_text, cur_tokens, cur_bio, cur_rels)
                cur_tokens, cur_bio, cur_rels = [], [], []
            cur_text = line.split("=", 1)[1].strip()
            continue

        if line.startswith("# BIO"):
            cur_bio = line.split("=", 1)[1].strip().split()
            continue

        if line.startswith("# RELS"):
            rels_str = line.split("=", 1)[1].strip()
            pairs = re.findall(r"\((\d+),(\d+),(\d+),(\d+),([^)]+)\)", rels_str)
            cur_rels = [(int(a), int(b), int(c), int(d), e.strip()) for a, b, c, d, e in pairs]
            continue

        # token 行（优先按制表符）
        if line[0].isdigit():
            parts = line.split("\t")
            if len(parts) >= 2:
                cur_tokens.append(parts[1])
            else:
                parts = line.split()
                if len(parts) >= 2:
                    cur_tokens.append(parts[1])

    _flush_block(sents, cur_text, cur_tokens, cur_bio, cur_rels)
    return sents
from typing import List, Dict, Any

def batchify(examples: List[Dict[str, Any]], tok_vocab, bio_vocab, rel_vocab, max_len=128):
    import torch

    B = len(examples)
    T = min(max(len(ex["tokens"]) for ex in examples), max_len)

    PAD_ID = 0                       # 一般 <pad> = 0
    O_ID   = bio_vocab.id("O")       # 正确的 O id

    ids  = torch.full((B, T), PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    bio  = torch.full((B, T), O_ID,  dtype=torch.long)
    rel_pairs = []

    for i, ex in enumerate(examples):
        toks = ex["tokens"][:T]
        L = len(toks)
        for j, w in enumerate(toks):
            ids[i, j] = tok_vocab.id(w)
            mask[i, j] = 1

        # BIO 补齐到 L
        tags = (ex.get("bio") or [])[:T]
        if len(tags) < L:
            tags = tags + ["O"] * (L - len(tags))
        for j, t in enumerate(tags[:L]):
            bio[i, j] = bio_vocab.id(t)

        # RELS：期望 0-based；越界/非法 span 跳过。
        pairs_raw = ex.get("rels", []) or []
        # 如果发现 span ≥ L 且都在 [1, L] 内，判定为 1-based，整体减 1
        if any(max(hs, he, ts, te) >= L for (hs, he, ts, te, _) in pairs_raw):
            if pairs_raw and max(hs for hs, _, _, _, _ in pairs_raw) <= L and \
               max(he for _, he, _, _, _ in pairs_raw) <= L and \
               max(ts for _, _, ts, _, _ in pairs_raw) <= L and \
               max(te for _, _, _, te, _ in pairs_raw) <= L:
                print("[INFO] RELS look 1-based; converting to 0-based for this sample.")
                pairs_raw = [(hs-1, he-1, ts-1, te-1, r) for (hs, he, ts, te, r) in pairs_raw]

        pairs = []
        for (hs, he, ts, te, r) in pairs_raw:
            if hs > he or ts > te:
                continue
            if not (0 <= hs < L and 0 <= he < L and 0 <= ts < L and 0 <= te < L):
                continue
            if r not in rel_vocab.stoi:
                # 你也可以映射到 NONE： rid = rel_vocab.id("NONE")
                continue
            pairs.append((hs, he, ts, te, rel_vocab.id(r)))
        rel_pairs.append(pairs)

    return {"ids": ids, "mask": mask, "bio": bio, "rel_pairs": rel_pairs}
