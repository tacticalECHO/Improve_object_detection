#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, random

# ---------- IO ----------
def read_conllu(path):
    sents, sent = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            if not line:
                if sent: sents.append(sent); sent=[]
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            sent.append({
                "id": int(cols[0]),
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "head": int(cols[6]) if cols[6].isdigit() else 0,
                "deprel": cols[7]
            })
    if sent: sents.append(sent)
    return sents

def to_conllu_block(tokens, bio, rels):
    lines=[]
    lines.append("# text = " + " ".join(tokens))
    lines.append("# BIO = " + " ".join(bio))
    rel_txt = " ; ".join(f"({hs},{he},{ts},{te},{lab})" for (hs,he,ts,te,lab) in rels)
    lines.append("# RELS = " + rel_txt)
    for i,w in enumerate(tokens, start=1):
        lines.append(f"{i}\t{w}\t{w}\tX")
    lines.append("")
    return "\n".join(lines)

# ---------- helpers ----------
def index_sent(sent):
    by_id = {t["id"]: t for t in sent}
    children = {t["id"]: [] for t in sent}
    for t in sent:
        h = t["head"]
        if h in children:
            children[h].append(t["id"])
    return by_id, children

def children_with(by_id, children, node_id, relset=None, upos=None):
    out=[]
    for ch in children.get(node_id, []):
        t=by_id[ch]
        if (relset is None or t["deprel"].split(":")[0] in relset) and (upos is None or t["upos"] in upos):
            out.append(ch)
    return out

def has_case_word(by_id, children, node_id, words):
    ws = set(w.lower() for w in words)
    for ch in children.get(node_id, []):
        t=by_id[ch]
        if t["deprel"].split(":")[0] == "case" and t["form"].lower() in ws:
            return t["form"].lower()
    return None

# ---------- conversion ----------
def build_np_spans(sent):
    """
    以名词为头：向左吸收 det/amod 形成 NP span，输出 [(start,end)]（0基闭区间）
    """
    by_id, children = index_sent(sent)
    noun = {"NOUN","PROPN"}
    spans=[]
    for t in sent:
        if t["upos"] not in noun:
            continue
        i = t["id"] - 1
        # 向左吸收其孩子中的 det/amod（在表面序上可能不在左，但 EWT 常在左）
        dets = [by_id[ch]["id"]-1 for ch in children_with(by_id, children, t["id"], {"det"})]
        amods= [by_id[ch]["id"]-1 for ch in children_with(by_id, children, t["id"], {"amod"})]
        cand = dets + amods + [i]
        s,e = min(cand), max(cand)
        spans.append((s,e))
    # 合并重叠/相邻
    spans.sort()
    merged=[]
    for s,e in spans:
        if not merged or s>merged[-1][1]+1:
            merged.append([s,e])
        else:
            merged[-1][1]=max(merged[-1][1], e)
    return [(s,e) for s,e in merged]

def bio_from_spans(T, np_spans, amod_positions):
    bio = ["O"]*T
    # 属性标 B-ATTR
    for i in amod_positions:
        if 0<=i<T:
            bio[i]="B-ATTR"
    # NP spans -> B/I
    for s,e in np_spans:
        for k in range(s,e+1):
            bio[k] = "I-NP" if k>s else "B-NP"
    return bio

def extract_relations(sent, np_spans):
    """
    on/under/near: head(noun) --nmod/obl--> tail(noun) (tail has case=on/under/near)
    left_of/right_of: head(noun) --nmod/obl--> left/right (NOUN/ADJ) ; left/right --nmod(of)--> tail(noun)
    """
    by_id, children = index_sent(sent)
    noun = {"NOUN","PROPN"}
    rels=[]
    # map token index -> which NP span id
    span_id_by_tok = {}
    for si,(s,e) in enumerate(np_spans):
        for k in range(s,e+1):
            span_id_by_tok[k]=si

    # ---- prepositions ----
    PREP_REL = {"on":"on","under":"under","near":"near"}
    for t in sent:
        if t["deprel"].split(":")[0] in {"nmod","obl"} and t["upos"] in noun:
            prep = has_case_word(by_id, children, t["id"], PREP_REL.keys())
            if prep:
                tail_i = t["id"]-1
                head_i = by_id[t["head"]]["id"]-1 if t["head"] in by_id else None
                if head_i is None or head_i not in span_id_by_tok or tail_i not in span_id_by_tok:
                    continue
                hs,he = np_spans[span_id_by_tok[head_i]]
                ts,te = np_spans[span_id_by_tok[tail_i]]
                rels.append((hs,he,ts,te, PREP_REL[prep]))

    # ---- left/right of ----
    for t in sent:
        if t["form"].lower() in {"left","right"} and t["upos"] in {"NOUN","ADJ"}:
            lr = t["form"].lower()
            # head noun?
            head = by_id.get(t["head"])
            if not head or head["upos"] not in noun:
                continue
            # tail via nmod with case=of
            tail_id = None
            for ch in children.get(t["id"], []):
                ch_t = by_id[ch]
                if ch_t["deprel"].split(":")[0] == "nmod" and has_case_word(by_id, children, ch, {"of"}):
                    if ch_t["upos"] in noun:
                        tail_id = ch; break
            if tail_id is None:
                continue
            head_i = head["id"]-1
            tail_i = tail_id-1
            if head_i not in span_id_by_tok or tail_i not in span_id_by_tok:
                continue
            hs,he = np_spans[span_id_by_tok[head_i]]
            ts,te = np_spans[span_id_by_tok[tail_i]]
            rels.append((hs,he,ts,te, "left_of" if lr=="left" else "right_of"))
    return rels

def convert_sent(sent):
    T = len(sent)
    tokens = [t["form"] for t in sent]
    by_id, children = index_sent(sent)
    # amod positions (attributes)
    amod_pos = []
    for t in sent:
        if t["deprel"] == "amod":
            amod_pos.append(t["id"]-1)
    np_spans = build_np_spans(sent)
    bio = bio_from_spans(T, np_spans, amod_pos)
    rels = extract_relations(sent, np_spans)
    return tokens, bio, rels

# ---------- pipeline ----------
def convert_file(inp, outp, limit=None, keep_only_pos=False, keep_only_rel=False, shuffle=False, seed=0):
    sents = read_conllu(inp)
    if shuffle:
        random.Random(seed).shuffle(sents)
    kept=0
    with open(outp, "w", encoding="utf-8") as fo:
        for s in sents:
            tokens, bio, rels = convert_sent(s)
            if keep_only_pos and all(t=="O" for t in bio):
                continue
            if keep_only_rel and not rels:
                continue
            fo.write(to_conllu_block(tokens, bio, rels))
            kept += 1
            if limit and kept >= limit:
                break
    print(f"[OK] wrote {outp} with {kept} sentences")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", required=True)
    ap.add_argument("--dev_in",   required=True)
    ap.add_argument("--test_in",  required=True)
    ap.add_argument("--out_dir", default="data/ud_ewt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--keep_only_pos", action="store_true",
                    help="仅保留含非O(B/I/ATTR)的句子（强推，避免全O）")
    ap.add_argument("--keep_only_rel", action="store_true",
                    help="仅保留含关系的句子（更强筛选）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kw = dict(limit=args.limit or None,
              keep_only_pos=args.keep_only_pos,
              keep_only_rel=args.keep_only_rel,
              shuffle=args.shuffle,
              seed=args.seed)
    convert_file(args.train_in, os.path.join(args.out_dir, "train.conllu"), **kw)
    convert_file(args.dev_in,   os.path.join(args.out_dir, "dev.conllu"),   **kw)
    convert_file(args.test_in,  os.path.join(args.out_dir, "test.conllu"),  **kw)
