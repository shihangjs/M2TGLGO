This project performs end-to-end processing of histological image patches, including:
1. Nuclear Segmentation using [Hover-Net](https://github.com/vqdang/hover_net)
2. Feature Extraction from segmented nuclei
3. Model Training and Evaluation
---
## Pipeline Overview
### Step 1: Nuclear Segmentation
Use a pretrained Hover-Net model to segment nuclei from image patches.
#### Input:
- Image patches in: `/data/bc/{DATASET}/patches/`
#### Output:
- Segmentation masks saved to: `/dlpfc/{bc}/segment/`
#### Run Segmentation Script (`batch_infer.sh`):
```bash
#!/bin/bash
datasets=(
    A1 A2 A3 A4 A5 A6
    B1 B2 B3 B4 B5 B6
)

for dataset in "${datasets[@]}"; do
    python3 run_infer.py \
        --gpu='3' \
        --nr_types=6 \
        --type_info_path=type_info.json \
        --batch_size=32 \
        --model_mode=fast \
        --model_path=./pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
        --nr_inference_workers=8 \
        --nr_post_proc_workers=16 \
        tile \
        --input_dir="/data/dlpfc/${dataset}/patches" \
        --output_dir="/dlpfc/${dataset}/segment" \
        --mem_usage=0.1 \
        --draw_dot \
        --save_qupath
done
```
### Step 2: Feature Extraction
After segmentation, extract morphological, spatial, or biological features using scripts in the preprocess/ folder.
### Step 3: Model Training and Evaluation
