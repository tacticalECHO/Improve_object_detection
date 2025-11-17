import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from PIL import Image, ImageDraw, ImageFont
import re
import ollama
import time

torch.manual_seed(1234)

"""
This script processes images in a specified directory for LLM-Visual-RFT, extracting noun chunks from a text file and generating
 bounding boxes for them using a pre-trained model. It also saves the processed images with bounding boxes 
 and logs the processing time and details.
"""
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def prepare_inputs(img_path, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output the thinking process in <think> </think> and your grounding box. Following \"<think> thinking process </think>\n<answer>(x1,y1),(x2,y2)</answer>)\" format."}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda")
def filter_inclusive_chunks(chunks):
    filtered = []
    for chunk in sorted(chunks, key=len, reverse=True):  # we sort by length to ensure longer chunks are checked first
        if not any(chunk in longer_chunk for longer_chunk in filtered):
            filtered.append(chunk)
    return filtered

# load model and processor
model_name = "deepseek-r1:7b"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward", device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained("Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward")

# image folder path
image_dir = "./test_un/"

# traverse all image files in the folder and its subfolders
for root, dirs, files in os.walk(image_dir):
    # 检查是否存在对应的文本文件
    text_file_path = os.path.join(root, "a.txt")
    if not os.path.exists(text_file_path):
        print(f"No input_text.txt found in {root}, skipping...")
        continue

    # 从文本文件中读取输入文本
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        input_text = text_file.read().strip()
        print(f"Input text from {text_file_path}: {input_text}")

    for file in files:
        if file.lower().endswith(('.jpg')) and not file.startswith("processed_"):
            image_path = os.path.join(root, file)

            # check if already processed
            processed_file_name = f"processed_{file}"
            processed_file_path = os.path.join(root, processed_file_name)
            if os.path.exists(processed_file_path):
                print(f"Skipping already processed image: {processed_file_name}")
                continue

            print(f"Processing image: {image_path}")

            # record start time
            start_time = time.time()

            # build prompt
            prompt = (
                "A conversation between User and Assistant. The user asks a question, and the Assistant find all the noun chunks in it. The chunk is like a noun phrase, e.g., the chunk in sentence:'what is the child in green doing?' is 'the child in green'. the chunk in sentence:'what is the chld in green shirt doing?' is 'the child in green shirt'."
                "if one answer has include other answers, just return the answer which is the most inclusive. e.g. the chunk in sentence:'what is the child in green doing?' is 'the child in green'. DO NOT answer with 'the child in green' and 'the child', only answer 'the child in green'."
                "answer are enclosed within <answer> </answer> tags, respectively, i.e., "
                "find all matching noun chunks in the sentence. "
                "<answer> answer here </answer>"    
                "if there is more than one answer, separate them with a new <answer> tag."
                "the user asks:{}"
            ).format(input_text)

            # call the model to generate response
            response_ol = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            response_ol = response_ol.message.content

            # extract and process the response
            infos = []
            matches_ol = re.findall(r'<answer>(.*?)</answer>', response_ol)
            matches_ol=list(set(matches_ol))  # de-duplicate
            matches_ol = filter_inclusive_chunks(matches_ol)  # filter inclusive chunks
            print(matches_ol)
            for chunk in matches_ol:
                inputs = prepare_inputs(image_path, "{}. Output thinking process as detail as possibile and describe it (what it is doing?)".format(chunk))
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print(response)
                think_pattern = r"<think>(.*?)</think>"
                think_matches = re.findall(think_pattern, response, re.DOTALL)
                think_text = think_matches[-1].strip() if think_matches else "No think process found"
                pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                matches = re.findall(pattern, response)
                x1, y1, x2, y2 = map(int, matches[0])
                infos.append((chunk, (x1, y1, x2, y2), think_text))
                print(infos)

            # draw the image
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            w, h = image.size

            # set font for text
            font = ImageFont.truetype("arial.ttf", 10)  # you can change the font and size here

            for info in infos:
                chunk, matches, think = info
                x1, y1, x2, y2 = map(int, matches)
                box_r1 = [int(x1) / 1000, int(y1) / 1000, int(x2) / 1000, int(y2) / 1000]
                
                # draw the bounding box
                draw.rectangle([box_r1[0] * w, box_r1[1] * h, box_r1[2] * w, box_r1[3] * h], outline="green", width=5)
                
                # draw the text
                text_bbox_chunk = font.getbbox(chunk) 
                text_bbox_think = font.getbbox(think)
                text_size_chunk = (text_bbox_chunk[2] - text_bbox_chunk[0], text_bbox_chunk[3] - text_bbox_chunk[1])
                text_size_think = (text_bbox_think[2] - text_bbox_think[0], text_bbox_think[3] - text_bbox_think[1])
                
                # draw the chunk text background
                chunk_bg_x1 = box_r1[0] * w
                chunk_bg_y1 = box_r1[1] * h - text_size_chunk[1] - 5  
                chunk_bg_x2 = chunk_bg_x1 + text_size_chunk[0] + 10  
                chunk_bg_y2 = box_r1[1] * h
                draw.rectangle([chunk_bg_x1, chunk_bg_y1, chunk_bg_x2, chunk_bg_y2], fill="white")
                
                
                # draw the think text
                draw.text((chunk_bg_x1 + 5, chunk_bg_y1 + 2), chunk, font=font, fill="red")
                #draw.text((think_bg_x1 + 5, think_bg_y1 + 2), think, font=font, fill="blue")
            # save the processed image
            image.save(processed_file_path)
            print(f"Processed image saved to: {processed_file_path}")

            # save the processing time to file
            end_time = time.time()
            processing_time = end_time - start_time
            
            # save the processing time, thought text, and bounding box to file
            time_log_path = os.path.join(root, "processing_time.txt")
            with open(time_log_path, "a", encoding="utf-8") as time_log_file:
                time_log_file.write(f"{file}: {processing_time:.2f} seconds\n")
                for info in infos:
                    chunk, (x1, y1, x2, y2), think_text = info
                    time_log_file.write(f"Chunk: {chunk}\n")
                    time_log_file.write(f"Think Text: {think_text}\n")
                    time_log_file.write(f"BBox: ({x1}, {y1}), ({x2}, {y2})\n")
                time_log_file.write("\n")  # add a blank line to separate different images' information