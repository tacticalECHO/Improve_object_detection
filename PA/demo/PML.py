#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rft_quick_demo.py
前端：semantic_parser.infer.run_parser(text)  -> 结构体 {entities/attributes/relations}
后端：Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward -> 依据结构体文字化的 prompt 返回框
输出：可视化图片 + JSON 框坐标
"""

import os, re, json, argparse, pathlib, sys
from typing import List, Dict, Any

import torch
from PIL import Image, ImageDraw, ImageFont

# ====== 你的工程根路径 & 解析器 ======
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append("semantic_parser")   # 确保能 import 到你的解析器包

from semantic_parser import infer as sem_infer  # 你的解析器，提供 run_parser(text)

# ====== Visual-RFT / Qwen2-VL ======
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info   # 来自 Qwen 官方示例工具

MODEL_NAME = "Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward"


# ---------------- 解析模型输出里的 boxes ----------------

def parse_boxes_from_raw_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    从 raw_text 中解析出 [{label, bbox}] 列表。
    优先找 ```json ... ``` 代码块，里面应该是一个 JSON 数组。
    若找不到，再在全文里找第一个 JSON 数组。
    """
    # 1) 优先找 ```json ... ``` 代码块
    m = re.search(r"```json(.*?)```", raw_text, flags=re.S | re.I)
    if m:
        block = m.group(1).strip()
        try:
            arr = json.loads(block)
            if isinstance(arr, list):
                out = []
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    lab = item.get("label") or item.get("name") or "object"
                    bbox = item.get("bbox") or item.get("box")
                    if isinstance(bbox, list) and len(bbox) == 4:
                        out.append({"label": str(lab), "bbox": bbox})
                if out:
                    return out
        except Exception:
            pass

    # 2) 兜底：在整段文本里找第一个 JSON 数组
    for m in re.finditer(r'\[[\s\S]*?\]', raw_text):
        candidate = m.group(0)
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                out = []
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    lab = item.get("label") or item.get("name") or "object"
                    bbox = item.get("bbox") or item.get("box")
                    if isinstance(bbox, list) and len(bbox) == 4:
                        out.append({"label": str(lab), "bbox": bbox})
                if out:
                    return out
        except Exception:
            continue

    return []  # 实在解析不到就返回空


# ---------------- 根据 result.json 再画一遍（可选工具） ----------------

def draw_from_result_json(image_path, result_json_path, out_path=None, title_text=None):
    """
    image_path: 原始图片路径
    result_json_path: 你保存的结果 json（里面有 query/prompt/raw_text/detections 等）
    out_path: 输出图片路径（默认在原图旁边加 .vis2.jpg）
    title_text: 左上角写一句提示，比如 query（可选）
    """
    from pathlib import Path

    image_path = Path(image_path)
    result_json_path = Path(result_json_path)

    # 1) 读图和 JSON
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    with result_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data.get("detections") or []
    # 如果 detections 是空，就从 raw_text 里再解析一次
    if not detections:
        raw_text = data.get("raw_text", "")
        detections = parse_boxes_from_raw_text(raw_text)

    if not detections:
        raise ValueError("在 detections 和 raw_text 里都没有解析出 bbox，画不出框。")

    # 2) 在图上画框
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        label = det.get("label", "object")
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])

        # 你的 prompt 说的是 0~1000，所以先按这个来，同时也兼容 0~1 / 像素
        maxv = max(x1, y1, x2, y2)
        if maxv <= 1.01:
            # 0~1
            sx1, sy1, sx2, sy2 = x1 * W, y1 * H, x2 * W, y2 * H
        elif maxv <= 1010:
            # 0~1000
            sx1, sy1, sx2, sy2 = x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H
        else:
            # 像素坐标
            sx1, sy1, sx2, sy2 = x1, y1, x2, y2

        draw.rectangle([sx1, sy1, sx2, sy2], outline="green", width=4)
        draw.text((sx1 + 2, sy1 + 2), label, fill="red", font=font)

    if title_text is None:
        title_text = data.get("query", "") or data.get("prompt", "")
    if title_text:
        draw.text((10, 10), title_text, fill="blue", font=font)

    # 3) 保存
    if out_path is None:
        out_path = image_path.with_suffix(".vis_from_json.jpg")
    out_path = Path(out_path)
    image.save(out_path)
    print(f"[OK] 已保存可视化结果到: {out_path}")
    return str(out_path)


# ---------------- 结构体 -> Prompt ----------------

def struct_to_prompt(struct: Dict[str, Any]) -> str:
    """
    把结构体转成 Visual-RFT 友好的自然语言提示。
    entities: [{'text': 'cup'}, ...]
    attributes: [{'head':'cup','type':'color','value':'red'}, ...]
    relations: [{'head':'cup','rel':'on','tail':'table'}, ...]
    """
    ents = [e["text"] for e in struct.get("entities", []) if e.get("text")]

    # 把属性合到名词前
    def decorate(name: str) -> str:
        adjs = []
        for a in struct.get("attributes", []):
            if a.get("head", "").lower() == name.lower():
                val = (a.get("value") or "").strip()
                if val:
                    adjs.append(val)
        return (" ".join(adjs) + " " + name).strip()

    ent_phrase = ", ".join(decorate(n) for n in ents) if ents else "the target objects"

    rels = []
    for r in struct.get("relations", []):
        rel = (r.get("rel") or "").replace("_", " ")
        rels.append(f'{r.get("head")} {rel} {r.get("tail")}')
    rel_phrase = "; ".join(rels)

    # 让 LVLM 倾向产出 JSON（框+标签）
    prompt = (
        f"Locate {ent_phrase}. "
        f"Relations: {rel_phrase}. "
        f"Return bounding boxes for each mentioned object as a JSON list, "
        f'where each item is {{"label": "<name>", "bbox": [x1,y1,x2,y2]}}. '
        f"The bbox coordinates should be normalized to the range [0, 1000]."
    )
    return prompt


def draw_boxes(image: Image.Image, dets: List[Dict[str, Any]], out_path: str, title: str = ""):
    W, H = image.size
    im = image.copy()
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()

    for det in dets:
        lab = det["label"]
        x1, y1, x2, y2 = det["bbox"]

        # 兼容 0~1 / 0~1000 / 像素
        mx = max(x1, y1, x2, y2)
        if mx <= 1.01:        # 0~1
            sx1, sy1, sx2, sy2 = x1 * W, y1 * H, x2 * W, y2 * H
        elif mx <= 1010:      # 0~1000
            sx1, sy1, sx2, sy2 = x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H
        else:                 # 像素
            sx1, sy1, sx2, sy2 = x1, y1, x2, y2

        draw.rectangle([sx1, sy1, sx2, sy2], outline="lime", width=4)
        draw.text((sx1 + 2, sy1 + 2), lab, fill="red", font=font)

    if title:
        draw.text((10, 10), title, fill="blue", font=font)
    im.save(out_path)


# ---------------- 推理主流程 ----------------

@torch.no_grad()
def run_pipeline(image_path: str, query_text: str,
                 model_name: str = MODEL_NAME,
                 max_new_tokens: int = 512,
                 device: str | None = None) -> Dict[str, Any]:
    """
    1) 前端：语义解析器 -> 结构体
    2) 结构体 -> 文本 prompt
    3) 后端：Visual-RFT Qwen2-VL 生成 -> 解析出框
    4) 画框 + 返回 JSON
    """
    # 1) 前端：语义解析
    struct = sem_infer.run_parser(query_text)
    # 兜底：如果没抽到 entities，最少把词面塞进来（可选）
    if not struct.get("entities"):
        struct["entities"] = [{"text": w} for w in query_text.split()]

    prompt = struct_to_prompt(struct)

    # 2) 模型 & 处理器
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # 3) 构造多模态消息（Qwen 官方风格）
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": "You are a helpful vision assistant that returns JSON boxes."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    vision_inputs = process_vision_info(messages)

    # ✅ 兼容两种实现：dict 或 tuple / list
    if isinstance(vision_inputs, dict):
        images = vision_inputs.get("images", None)
        videos = vision_inputs.get("videos", None)
    elif isinstance(vision_inputs, (list, tuple)):
        # 老版本 qwen_vl_utils：通常返回 (images, videos) 或只返回 images
        if len(vision_inputs) == 2:
            images, videos = vision_inputs
        else:
            images, videos = vision_inputs, None
    else:
        # 极端兜底：直接当作 images
        images, videos = vision_inputs, None

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt"
    ).to(model.device)

    # 4) 生成
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # 5) 解析框（用 parse_boxes_from_raw_text）
    detections = parse_boxes_from_raw_text(out_text)

    # 6) 可视化 & 输出
    vis_path = str(pathlib.Path(image_path).with_suffix(".vis_rft.jpg"))
    if detections:
        draw_boxes(image, detections, vis_path, title=query_text)
    else:
        print("[WARN] 没解析出任何 bbox，仍然保存原图。")
        image.save(vis_path)

    result = {
        "query": query_text,
        "prompt": prompt,
        "raw_text": out_text,
        "detections": detections,
        "vis_image": vis_path
    }
    res_path = str(pathlib.Path(image_path).with_suffix(".vis_rft.json"))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved: {vis_path}\n[OK] saved: {res_path}")
    return result


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--text", required=True, help="用于解析的语句，如: 'find the red cup on the table near the window'")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    run_pipeline(args.image, args.text, model_name=args.model, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
