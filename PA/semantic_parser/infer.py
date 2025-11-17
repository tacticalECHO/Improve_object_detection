#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F

import spacy
from spacy.tokens import Doc

from models.bilstm_crf import BiLSTMCRFParser
from utils.vocab import Vocab
from utils.data_loader import batchify

# ----------------- IO & Vocab -----------------

def read_lines(p: Path) -> List[str]:
    return [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def load_vocab_txt(p: Path) -> Vocab:
    v = Vocab()
    toks = read_lines(p)
    v.itos = toks
    v.stoi = {t: i for i, t in enumerate(toks)}
    return v

def simple_tokenize(text: str) -> List[str]:
    # 与训练一致：空格切分
    return text.strip().split()

# ----------------- BIO 解码与实体/属性 -----------------

def has_named_tags(tagset: List[str], *names) -> bool:
    s = set(tagset)
    return all(n in s for n in names)

def collect_spans_from_bio(tokens: List[str], tags: List[str], tagset: List[str]):
    ents, attrs = [], []
    if has_named_tags(tagset, "B-NP", "I-NP") or any("NP" in t for t in tagset):
        def collect(prefix):
            spans = []
            start = -1
            for i, tag in enumerate(tags + ["O"]):
                if tag.startswith(f"B-{prefix}"):
                    if start != -1:
                        spans.append((start, i - 1))
                    start = i
                elif tag.startswith(f"I-{prefix}"):
                    continue
                else:
                    if start != -1:
                        spans.append((start, i - 1))
                        start = -1
            return [{"text": " ".join(tokens[s:e+1]), "start": s, "end": e} for s, e in spans]
        ents = collect("NP")
        if any("ATTR" in t for t in tagset):
            attrs = collect("ATTR")
    else:
        start = -1
        for i, tag in enumerate(tags + ["O"]):
            if tag != "O":
                if start == -1:
                    start = i
            else:
                if start != -1:
                    ents.append({"text": " ".join(tokens[start:i]), "start": start, "end": i - 1})
                    start = -1
    return ents, attrs

def infer_attrs_from_np(tokens: List[str], entities: List[Dict[str, Any]]):
    DET = {"a","an","the"}
    attrs = []
    for e in entities:
        s, t = e["start"], e["end"]
        span = tokens[s:t+1]
        if not span: continue
        k = 1 if span[0].lower() in DET else 0
        if k < len(span)-1:
            val = " ".join(span[k:-1]).strip()
            if val:
                head = span[-1]
                attrs.append({"head": head, "type": "attr", "value": val, "span_token": [s+k, s+len(span)-2]})
                e["text"] = head
        else:
            e["text"] = span[-1]
    return attrs

def filter_pseudo_entities(entities: List[Dict[str, Any]]):
    BLOCK = {"left","right","of","on","under","near","to","the","a","an",",",".",";",":","'","\""}
    out=[]
    for e in entities:
        if e["text"].lower() in BLOCK: continue
        out.append(e)
    return out

# ----------------- 关系：MLP（可选） -----------------

def span_repr(H: torch.Tensor, s: int, e: int) -> torch.Tensor:
    s = max(int(s), 0); e = max(int(e), 0)
    T = H.size(1); s = min(s, T-1); e = min(e, T-1)
    if e >= s: return H[0, s:e+1, :].mean(dim=0)
    return H[0, s, :]

def score_relations_mlp(model: BiLSTMCRFParser, H: torch.Tensor,
                        entities: List[Dict[str, Any]], rel_tags: List[str],
                        topk=None, thr=None):
    if len(entities) < 2: return []
    feats, idx = [], []
    for i,h in enumerate(entities):
        hv = span_repr(H, h["start"], h["end"])
        for j,t in enumerate(entities):
            if i==j: continue
            tv = span_repr(H, t["start"], t["end"])
            z = torch.cat([hv, tv, hv*tv], dim=-1)
            feats.append(z); idx.append((i,j))
    if not feats: return []
    logits = model.rel_mlp(torch.stack(feats, 0))
    probs = F.softmax(logits, dim=-1)
    scores, ids = probs.max(dim=-1)
    NONE = {"NONE","none","O","no_relation","no-rel","no_rel"}
    out=[]
    for (i,j), lab, sc in zip(idx, ids.tolist(), scores.tolist()):
        rel = rel_tags[lab] if lab < len(rel_tags) else f"REL_{lab}"
        if rel in NONE: continue
        if thr is not None and sc < thr: continue
        out.append({"head": entities[i]["text"], "rel": rel, "tail": entities[j]["text"], "score": float(sc)})
    if topk:
        byh = {}
        for r in out: byh.setdefault(r["head"], []).append(r)
        trimmed=[]
        for h,lst in byh.items():
            lst.sort(key=lambda x:-x["score"]); trimmed.extend(lst[:topk])
        return trimmed
    return out

# ----------------- 依存图辅助：token 对齐 -----------------

def build_char_spans_from_tokens(tokens: List[str]) -> List[Tuple[int,int]]:
    """
    把空格拼接的 tokens 生成 char 范围：[start,end)；用于对齐到 spaCy 的 char 级。
    """
    spans=[]
    off=0
    for tok in tokens:
        s=off; e=off+len(tok)
        spans.append((s,e))
        off = e+1  # 加一个空格
    return spans

def align_entity_to_doc(tokens: List[str], entities: List[Dict[str,Any]], doc: Doc) -> Dict[int, List[int]]:
    """
    返回: ent_index -> list(spacy_token_idx) 覆盖此实体（按 char overlap）
    """
    text_joined = " ".join(tokens)
    tok_spans = build_char_spans_from_tokens(tokens)
    ent_char_spans = []
    for e in entities:
        s_tok, e_tok = e["start"], e["end"]
        cs = tok_spans[s_tok][0]
        ce = tok_spans[e_tok][1]
        ent_char_spans.append((cs, ce))

    mapping = {}
    for ei, (cs, ce) in enumerate(ent_char_spans):
        covered = []
        for i, w in enumerate(doc):
            ws, we = w.idx, w.idx+len(w)
            if not (we <= cs or ws >= ce):  # overlap
                covered.append(i)
        mapping[ei] = covered
    return mapping

def map_doc_token_to_entity(ent2doc: Dict[int,List[int]]) -> Dict[int,int]:
    """
    反向映射: doc_token_idx -> ent_index（优先覆盖最多的实体；并保持最早出现的实体）
    """
    doc2ent={}
    # 统计每个 doc token 被哪些实体覆盖
    coverage={}
    for ei, lst in ent2doc.items():
        for d in lst:
            coverage.setdefault(d, []).append(ei)
    for d, eis in coverage.items():
        doc2ent[d] = eis[0]
    return doc2ent

# ----------------- 依存规则抽关系（支持从句） -----------------

def deps_relations(doc, tokens, entities):
    """
    同时支持 UD 与 spaCy 依存：
      1) on/under/near
         - UD:   tail(noun) --(nmod/obl with case=prep)--> head(noun)
         - spaCy: head(noun) --prep(on/under/near)--> pobj(noun)=tail
         - 以及相对从句里：head 的 relcl 子树中的 prep(near)->pobj
      2) left/right of
         - UD:   head(noun) --(nmod/obl)--> lr(NOUN/ADJ); lr --nmod(case=of)--> tail(noun)
         - spaCy: head(noun) --prep(to)--> pobj(left/right) --prep(of)--> pobj=tail
    """
    if not entities:
        return []

    ent2doc = align_entity_to_doc(tokens, entities, doc)
    doc2ent = map_doc_token_to_entity(ent2doc)
    def ent_txt(di):
        ei = doc2ent.get(di, None)
        return entities[ei]["text"] if ei is not None else None

    noun_like = {"NOUN","PROPN"}
    PREP = {"on","under","near"}
    out = []
    seen = set()

    # ---- 1) on/under/near ----
    for t in doc:
        base = t.dep_.split(":")[0]

        # (A) UD 风格：tail(noun) --(nmod/obl with case in PREP)--> head(noun)
        if base in {"nmod","obl"} and t.pos_ in noun_like and t.head.pos_ in noun_like:
            prep = None
            for ch in t.children:
                if ch.dep_.split(":")[0] == "case" and ch.lower_ in PREP:
                    prep = ch.lower_; break
            if prep:
                h_txt = ent_txt(t.head.i); t_txt = ent_txt(t.i)
                if h_txt and t_txt and (h_txt, prep, t_txt) not in seen:
                    seen.add((h_txt, prep, t_txt))
                    out.append({"head": h_txt, "rel": prep, "tail": t_txt, "score": 1.0})

        # (B) spaCy 风格：head(noun) --prep(w)--> pobj(noun)=tail  + 重绑&relcl修正
        if t.pos_ in noun_like:
            head = t

            # 先收集 head 的所有 prep→pobj，供后续参考/重绑
            preps = []
            for ch in head.children:
                if ch.dep_ == "prep":
                    pobj = next((gc for gc in ch.children if gc.dep_ == "pobj" and gc.pos_ in noun_like), None)
                    if pobj:
                        preps.append((ch, pobj))

            SUPPORT  = {"on", "under"}               # 承载类介词
            PREP     = {"on","under","near"}         # 我们关注的介词
            NEARLIKE = {"near"}                      # 需要重绑的邻近类

            # 先正常抽取 head ─prep→ pobj
            for ch, pobj in preps:
                w = ch.lower_
                if w in PREP:
                    h_txt = ent_txt(head.i); t_txt = ent_txt(pobj.i)
                    if h_txt and t_txt and (h_txt, w, t_txt) not in seen:
                        seen.add((h_txt, w, t_txt))
                        out.append({"head": h_txt, "rel": w, "tail": t_txt, "score": 1.0})

            # 选择一个“承载” pobj（如 table），优先选出现在 near 之前的最近一个
            base_obj = None
            for ch2, pobj2 in preps:
                if ch2.lower_ in SUPPORT:
                    if base_obj is None or pobj2.i > base_obj.i:
                        base_obj = pobj2

            # ⭐ 对 near 做“后移重绑”：有 base_obj 时，把 near 绑定到 base_obj（如 table）
            for ch, pobj in preps:
                if ch.lower_ in NEARLIKE:
                    if base_obj is not None and base_obj.i < ch.i:  # near 出现在承载名词之后
                        h2_txt = ent_txt(base_obj.i)
                        t2_txt = ent_txt(pobj.i)
                        if h2_txt and t2_txt and (h2_txt, ch.lower_, t2_txt) not in seen:
                            seen.add((h2_txt, ch.lower_, t2_txt))
                            out.append({"head": h2_txt, "rel": ch.lower_, "tail": t2_txt, "score": 1.0})
                    else:
                        # 没有承载名词，就按原样绑定到 head
                        h_txt = ent_txt(head.i); t_txt = ent_txt(pobj.i)
                        if h_txt and t_txt and (h_txt, ch.lower_, t_txt) not in seen:
                            seen.add((h_txt, ch.lower_, t_txt))
                            out.append({"head": h_txt, "rel": ch.lower_, "tail": t_txt, "score": 1.0})

            # (C) 定语从句：若存在 base_obj，则把 relcl 子树里的 near 也改绑给 base_obj
            for rc in head.children:
                if rc.dep_ == "relcl":
                    for ch in rc.subtree:
                        if ch.dep_ == "prep" and ch.lower_ in PREP:
                            pobj = next((gc for gc in ch.children if gc.dep_ == "pobj" and gc.pos_ in noun_like), None)
                            if not pobj:
                                continue
                            if base_obj is not None and ch.lower_ in NEARLIKE:
                                h_txt = ent_txt(base_obj.i)   # 把 near 绑到 table
                            else:
                                h_txt = ent_txt(head.i)
                            t_txt = ent_txt(pobj.i)
                            if h_txt and t_txt and (h_txt, ch.lower_, t_txt) not in seen:
                                seen.add((h_txt, ch.lower_, t_txt))
                                out.append({"head": h_txt, "rel": ch.lower_, "tail": t_txt, "score": 1.0})
    # ---- 2) left/right of ----
    for t in doc:
        # UD 版本
        if t.lower_ in {"left","right"} and t.pos_ in {"NOUN","ADJ"}:
            head = t.head
            if head.pos_ in noun_like:
                tail = None
                for ch in t.children:
                    if ch.dep_.split(":")[0] == "nmod":
                        has_of = any(gc.dep_.split(":")[0]=="case" and gc.lower_=="of" for gc in ch.children)
                        if has_of and ch.pos_ in noun_like:
                            tail = ch; break
                if tail:
                    h_txt = ent_txt(head.i); t_txt = ent_txt(tail.i)
                    rel = "left_of" if t.lower_=="left" else "right_of"
                    if h_txt and t_txt and (h_txt, rel, t_txt) not in seen:
                        seen.add((h_txt, rel, t_txt))
                        out.append({"head": h_txt, "rel": rel, "tail": t_txt, "score": 1.0})

        # spaCy 版本：head --prep(to)--> pobj(left/right) --prep(of)--> pobj=tail
        if t.pos_ in noun_like:
            head = t
            for p_to in head.children:
                if p_to.dep_ == "prep" and p_to.lower_ == "to":
                    lr = next((gc for gc in p_to.children if gc.dep_ == "pobj" and gc.lower_ in {"left","right"}), None)
                    if lr is None:
                        continue
                    p_of = next((c for c in lr.children if c.dep_ == "prep" and c.lower_ == "of"), None)
                    if p_of is None:
                        continue
                    tail = next((gc for gc in p_of.children if gc.dep_ == "pobj" and gc.pos_ in noun_like), None)
                    if tail:
                        h_txt = ent_txt(head.i); t_txt = ent_txt(tail.i)
                        rel = "left_of" if lr.lower_ == "left" else "right_of"
                        if h_txt and t_txt and (h_txt, rel, t_txt) not in seen:
                            seen.add((h_txt, rel, t_txt))
                            out.append({"head": h_txt, "rel": rel, "tail": t_txt, "score": 1.0})

    return out

def merge_relations(rule_rels, mlp_rels):
    have = {(r["head"], r["rel"], r["tail"]) for r in rule_rels}
    return rule_rels + [r for r in mlp_rels if (r["head"], r["rel"], r["tail"]) not in have]

# ----------------- 主流程 -----------------
# parser_infer.py 里增加：

def run_parser(
    text: str,
    ckpt: str = "./semantic_parser/parser.ckpt",
    artifacts_dir: str = "./semantic_parser/artifacts",
    rel_topk: int = 0,
    rel_thr: float = 0.0,
    use_mlp: bool = True,
) -> Dict[str, Any]:
    # ---- spaCy 依存解析器
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("spaCy 模型未安装：请先运行 `python -m spacy download en_core_web_sm`")

    artifacts = Path(artifacts_dir)
    ckpt_state = torch.load(ckpt, map_location="cpu")

    # 1) 标签集合
    bio_path = artifacts / "bio_tags.txt"
    rel_path = artifacts / "rel_tags.txt"
    if bio_path.exists():
        BIO_TAGS = read_lines(bio_path)
    else:
        bio_n = ckpt_state["emissions.weight"].shape[0]
        BIO_TAGS = [f"T{i}" for i in range(bio_n)]
    if rel_path.exists():
        REL_TAGS = read_lines(rel_path)
    else:
        rel_n = ckpt_state["rel_mlp.2.weight"].shape[0]
        REL_TAGS = [f"R{i}" for i in range(rel_n)]

    # 2) 词表
    tok_path = artifacts / "tok_vocab.txt"
    if not tok_path.exists():
        raise FileNotFoundError(f"未找到 {tok_path}")
    tok_vocab = load_vocab_txt(tok_path)

    # 3) 模型
    model = BiLSTMCRFParser(
        vocab_size=len(tok_vocab),
        bio_label_count=len(BIO_TAGS),
        rel_label_count=len(REL_TAGS),
    )
    model.load_state_dict(ckpt_state, strict=True)
    model.eval()

    # 4) 准备 batch
    tokens = simple_tokenize(text)
    ex = {"tokens": tokens, "bio": ["O"] * len(tokens), "rels": []}
    pack = batchify([ex], tok_vocab, Vocab(), Vocab(), max_len=128)

    with torch.no_grad():
        _, emissions, H = model(pack["ids"], pack["mask"], None, None)
        path = model.decode(emissions, pack["mask"])[0]

    bio_tags_seq = [BIO_TAGS[i] if i < len(BIO_TAGS) else f"T{i}" for i in path[:len(tokens)]]

    entities, attributes_raw = collect_spans_from_bio(tokens, bio_tags_seq, BIO_TAGS)
    entities = filter_pseudo_entities(entities)

    # NP 内剥离属性，并把实体标准化为名词头
    attrs_from_np = infer_attrs_from_np(tokens, entities)

    # 属性就近挂载（来自 BIO 的 ATTR）
    attributes = []
    for a in attributes_raw:
        best_i, best_d = -1, 10**9
        for i, e in enumerate(entities):
            d = min(abs(a["start"] - e["start"]), abs(a["end"] - e["end"]))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            attributes.append({
                "head": entities[best_i]["text"],
                "type": "attr",
                "value": a["text"],
                "span_token": [a["start"], a["end"]],
            })
    attributes.extend(attrs_from_np)

    # 依存关系
    doc = nlp(" ".join(tokens))
    dep_rels = deps_relations(doc, tokens, entities)

    # MLP 关系（可选）
    if not use_mlp:
        relations = dep_rels
    else:
        mlp_rels = score_relations_mlp(
            model, H, entities=entities, rel_tags=REL_TAGS,
            topk=rel_topk if rel_topk > 0 else None,
            thr=rel_thr if rel_thr > 0 else None
        )
        relations = merge_relations(dep_rels, mlp_rels)

    out = {
        "tokens": tokens,
        "entities": [{"text": e["text"], "span_token": [e["start"], e["end"]]} for e in entities],
        "attributes": attributes,
        "relations": relations
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="parser.ckpt")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--rel_topk", type=int, default=0)
    ap.add_argument("--rel_thr", type=float, default=0.0)
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--no_mlp", action="store_true", help="仅依存规则，不用 MLP 关系头")
    args = ap.parse_args()

    # ---- spaCy 依存解析器
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("spaCy 模型未安装：请先运行 `python -m spacy download en_core_web_sm`")

    artifacts = Path(args.artifacts_dir)
    ckpt_state = torch.load(args.ckpt, map_location="cpu")

    # 1) 标签集合
    bio_path = artifacts / "bio_tags.txt"
    rel_path = artifacts / "rel_tags.txt"
    if bio_path.exists():
        BIO_TAGS = read_lines(bio_path)
    else:
        bio_n = ckpt_state["emissions.weight"].shape[0]
        BIO_TAGS = [f"T{i}" for i in range(bio_n)]
    if rel_path.exists():
        REL_TAGS = read_lines(rel_path)
    else:
        rel_n = ckpt_state["rel_mlp.2.weight"].shape[0]
        REL_TAGS = [f"R{i}" for i in range(rel_n)]

    # 2) 词表
    tok_path = artifacts / "tok_vocab.txt"
    if not tok_path.exists():
        raise FileNotFoundError(f"未找到 {tok_path}")
    tok_vocab = load_vocab_txt(tok_path)

    # 3) 模型
    model = BiLSTMCRFParser(
        vocab_size=len(tok_vocab),
        bio_label_count=len(BIO_TAGS),
        rel_label_count=len(REL_TAGS),
    )
    model.load_state_dict(ckpt_state, strict=True)
    model.eval()

    # 4) 准备 batch
    tokens = simple_tokenize(args.text)
    ex = {"tokens": tokens, "bio": ["O"] * len(tokens), "rels": []}
    pack = batchify([ex], tok_vocab, Vocab(), Vocab(), max_len=128)

    with torch.no_grad():
        _, emissions, H = model(pack["ids"], pack["mask"], None, None)
        path = model.decode(emissions, pack["mask"])[0]

    bio_tags_seq = [BIO_TAGS[i] if i < len(BIO_TAGS) else f"T{i}" for i in path[:len(tokens)]]

    entities, attributes_raw = collect_spans_from_bio(tokens, bio_tags_seq, BIO_TAGS)
    entities = filter_pseudo_entities(entities)

    # NP 内剥离属性，并把实体标准化为名词头
    attrs_from_np = infer_attrs_from_np(tokens, entities)

    # 属性就近挂载（来自 BIO 的 ATTR）
    attributes=[]
    for a in attributes_raw:
        best_i, best_d = -1, 10**9
        for i,e in enumerate(entities):
            d = min(abs(a["start"]-e["start"]), abs(a["end"]-e["end"]))
            if d < best_d: best_d, best_i = d, i
        if best_i >= 0:
            attributes.append({
                "head": entities[best_i]["text"],
                "type": "attr",
                "value": a["text"],
                "span_token": [a["start"], a["end"]],
            })
    attributes.extend(attrs_from_np)

    # 依存关系（支持从句）
    doc = nlp(" ".join(tokens))
    dep_rels = deps_relations(doc, tokens, entities)

    # MLP 关系（可选）
    if args.no_mlp:
        relations = dep_rels
    else:
        mlp_rels = score_relations_mlp(
            model, H, entities=entities, rel_tags=REL_TAGS,
            topk=args.rel_topk if args.rel_topk>0 else None,
            thr=args.rel_thr if args.rel_thr>0 else None
        )
        relations = merge_relations(dep_rels, mlp_rels)

    out = {
        "tokens": tokens,
        "entities": [{"text": e["text"], "span_token": [e["start"], e["end"]]} for e in entities],
        "attributes": attributes,
        "relations": relations
    }
    if args.pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
