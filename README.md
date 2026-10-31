# Text-to-Image-Generation-and-Evaluation

> **Status: work in progress.** The generation and evaluation script is written but has not been run yet, so there are no results to report.

## Goal
A learning project on text-to-image generation and how to evaluate it. It is based on the ideas in the paper below, which proposes generating images step by step with a program-like plan and evaluating them with interpretable checks (objects, counts, spatial relations) instead of a single score.

Planned work:
1. Generate images from a fixed set of prompts with an open text-to-image model.
2. Evaluate prompt-image agreement with CLIP score, plus simple object-presence and counting checks.
3. Report a results table and example images, with the model, settings and random seeds used.

## Reference
- Cho J., Zala A., Bansal M., *Visual Programming for Text-to-Image Generation and Evaluation*, NeurIPS 2023. arXiv:2305.15328, https://arxiv.org/abs/2305.15328

This repository is an independent learning project. It is not the authors' code and is not affiliated with them.

## Quick start
```bash
pip install -r requirements.txt
python src/generate_and_evaluate.py
```
Run from the project root on a GPU machine such as Google Colab (T4). It writes images to `results/images/`, per-prompt scores to `results/scores.csv` and a summary to `results/summary.txt`. The CLIP score here is the cosine similarity between image and prompt embeddings, multiplied by 100.

## Planned structure
```
src/generate_and_evaluate.py   generation + CLIP scoring
prompts/prompts.txt            fixed prompt list (10 prompts)
results/                       scores and example images (after a run)
README.md
requirements.txt
```

## Progress
- [x] Choose model (`stabilityai/sd-turbo`) and prompt set
- [x] Generation and CLIP-score script written (not yet run on real models)
- [ ] Run it and record results
- [ ] Object/count checks
- [ ] Results and limitations

## Notes
Check the licence of any model or dataset before use or redistribution. Generated images are not included unless the model licence allows it.

## Author
BEN, Software Engineer. GitHub: [Bangash514](https://github.com/Bangash514)
