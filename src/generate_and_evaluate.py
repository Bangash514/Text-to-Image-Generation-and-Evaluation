# BEN 2026-11-01
# Change aim: Generate images from a prompt list and score prompt-image agreement with CLIP.
# File: generate_and_evaluate.py
"""Generate one image per prompt with a text-to-image model, then compute a CLIP
similarity score (cosine similarity x 100) between each image and its prompt.

Run from the project root on a GPU machine (for example Google Colab):
    python src/generate_and_evaluate.py
Outputs: results/images/*.png, results/scores.csv, results/summary.txt
Check the licence of the chosen model before using or sharing generated images."""
import argparse
import os

import numpy as np
import pandas as pd


# BEN 2026-11-01
# Change aim: Read the prompt list from a text file, one prompt per line.
# File: generate_and_evaluate.py
# Function: load_prompts
def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# BEN 2026-11-01
# Change aim: Generate one image per prompt with a fixed seed so results can be repeated.
# File: generate_and_evaluate.py
# Function: generate_images
def generate_images(prompts, model_id, steps, guidance, base_seed, out_dir):
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype).to(device)
    os.makedirs(out_dir, exist_ok=True)
    images = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(base_seed + i)
        img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                   generator=gen).images[0]
        img.save(os.path.join(out_dir, f"img_{i:02d}.png"))
        images.append(img)
        print(f"[{i + 1}/{len(prompts)}] {prompt}")
    return images


# BEN 2026-11-01
# Change aim: Compute the CLIP cosine similarity (x100) between each image and its prompt.
# File: generate_and_evaluate.py
# Function: clip_scores
def clip_scores(images, prompts, clip_id):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(clip_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_id)
    scores = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            inp = proc(text=[prompt], images=img, return_tensors="pt", padding=True,
                       truncation=True).to(device)
            out = model(**inp)
            img_e = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_e = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            scores.append(float((img_e * txt_e).sum()) * 100.0)
    return scores


# BEN 2026-11-01
# Change aim: Run generation and evaluation, then write the scores table and a summary.
# File: generate_and_evaluate.py
# Function: main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts/prompts.txt")
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    images = generate_images(prompts, args.model, args.steps, args.guidance, args.seed,
                             os.path.join(args.out, "images"))
    scores = clip_scores(images, prompts, args.clip)

    df = pd.DataFrame({"prompt": prompts, "seed": [args.seed + i for i in range(len(prompts))],
                       "clip_score": scores})
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "scores.csv"), index=False, float_format="%.2f")
    summary = (f"model={args.model}\nclip={args.clip}\nsteps={args.steps} guidance={args.guidance}\n"
               f"prompts={len(prompts)}\nmean_clip_score={np.mean(scores):.2f}\n"
               f"std_clip_score={np.std(scores):.2f}\n")
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
