## Batch Patch Segmentation Using Hover-Net

In this project, we use [Hover-Net](https://github.com/vqdang/hover_net) to perform nuclear instance segmentation on image patches from multiple datasets. To streamline batch processing, we wrote the following shell script to iterate through each dataset and run the `run_infer.py` inference script:

```bash
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
