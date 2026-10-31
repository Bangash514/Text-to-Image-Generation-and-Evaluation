# Text-to-Image-Generation-and-Evaluation
Visual Programming for Text-to-Image Generation and Evaluation
# Text-to-Image-Generation-and-Evaluation

> **Status: work in progress.** This repository currently contains only this README. There is no code and there are no results yet.

## Goal
A learning project on text-to-image generation and how to evaluate it. It is based on the ideas in the paper below, which proposes generating images step by step with a program-like plan and evaluating them with interpretable checks (objects, counts, spatial relations) instead of a single score.

Planned work:
1. Generate images from a fixed set of prompts with an open text-to-image model.
2. Evaluate prompt-image agreement with CLIP score, plus simple object-presence and counting checks.
3. Report a results table and example images, with the model, settings and random seeds used.

## Reference
- Cho J., Zala A., Bansal M., *Visual Programming for Text-to-Image Generation and Evaluation*, NeurIPS 2023. arXiv:2305.15328, https://arxiv.org/abs/2305.15328

This repository is an independent learning project. It is not the authors' code and is not affiliated with them.

## Planned structure
```
src/        generation and evaluation scripts
prompts/    fixed prompt list
results/    metrics table and example images
README.md
requirements.txt
```

## Progress
- [ ] Choose model and prompt set
- [ ] Generation script
- [ ] CLIP-score evaluation
- [ ] Object/count checks
- [ ] Results and limitations

## Notes
Check the licence of any model or dataset before use or redistribution. Generated images are not included unless the model licence allows it.

## Author
BEN, Software Engineer. GitHub: [Bangash514](https://github.com/Bangash514)
