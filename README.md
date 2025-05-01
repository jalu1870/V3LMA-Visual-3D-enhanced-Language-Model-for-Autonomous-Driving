# V3LMA: Visual 3D-enhanced Language Model for Autonomous Driving 

Evaluation code for the corresponding [paper (currently processing)]() 🔬

## Build Docker Image
```bash
docker build -t qwen .
```

## Run Docker Container with GPU Support
```bash
docker run --gpus all -it --rm \
  -v /home/<username>/:/mnt/ \ 
  qwen bash
```

```bash
cd vlm_scene_understanding/preprocess/
```

## Download Grounded_SAM
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```

## Download SAM Checkpoints
```bash
cd Grounded-SAM-2/checkpoints 
bash download_ckpts.sh
```


## Navigate Back
```bash
cd ../..
```

## Download LingoQA Dataset
Download from their GitHub repository: [LingoQA](https://github.com/wayveai/LingoQA)

## Extract Desired Datasets
Extract the necessary datasets after downloading.

## Download Traffic Light Detection Checkpoint
```bash
cd traffic-light-detection/model_weights 
bash download_weights.sh
```
alternatively: Download from: [KIT Sync and Share](https://bwsyncandshare.kit.edu/s/iabWB5k3q3LKzRz)

## Download Yolo11x Checkpoint
Download from: [YOLO](https://docs.ultralytics.com/de/models/yolo11/#supported-tasks-and-modes) and place it in preprocess/
or
```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
```

## Data Preprocessing
```bash
python process.py --dataset_path "LingoQA/evaluation/images/val" \
--dataset_parquet_path "LingoQA/evaluation/val.parquet" \
--output_path "path at which to store the processed dataset, a .parquet file"
```


## Inference
To run inference over a variety of combination configurations:
```bash
python inference.py --model_name ("Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-2B-Instruct","lmms-lab/llava-onevision-qwen2-7b-ov","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-7B-Instruct","combination" or another base model alternatively: "combination") \
--val_data_path output_path(where the created .parquet dataset file is located) \
--llm_prompt_for_vision False
--llm_name ("Qwen/Qwen2.5-7B-Instruct" or "Qwen/Qwen2.5-2B-Instruct")
--vlm_name ("lmms-lab/llava-onevision-qwen2-7b-ov" or "Qwen/Qwen2.5-1.5B-Instruct" or "Qwen/Qwen2.5-7B-Instruct")
--mode ("standard" - loops over all configurations for the model, "best_only" loops over best configurations in earlier inference runs which were saved to "out/" and evaluated, "on_checkpoints" loops over all chackpoints for runs saved in "runs/")
```

## Evaluation
```bash
cd evaluate
python evaluation.py --dataset_parquet_path "../lingo/LingoQA/evaluation/val.parquet"
```
results are saved to "out/eval_result.json" an overview to "evaluate/outputs.xlsx"

## Training
```bash
torchrun --nproc_per_node=<num gpus> --rdzv_backend=c10d train.py 
--model_name ("Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-2B-Instruct","lmms-lab/llava-onevision-qwen2-7b-ov","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-7B-Instruct","combination" or another base model alternatively: "combination") \
--llm_name (base llm if model_name is "combination", then either: "Qwen/Qwen2.5-7B-Instruct" or "Qwen/Qwen2.5-2B-Instruct")
--vlm_name (base vlm if model_name is "combination", then either: "lmms-lab/llava-onevision-qwen2-7b-ov" or "Qwen/Qwen2.5-1.5B-Instruct" or "Qwen/Qwen2.5-7B-Instruct")
--train_data_path "LingoQA/evaluation/train.parquet" \
--resume False \
--pretrain_path none \
--use_lora True \
--lr 5e-5


## 📖 How to Cite

If you use this code or data in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{rivera2025scenariounderstandingtrafficscenes,
      title={Scenario Understanding of Traffic Scenes Through Large Visual Language Models}, 
      author={Esteban Rivera and Jannik Lübberstedt and Nico Uhlemann and Markus Lienkamp},
      year={2025},
      eprint={2501.17131},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.17131}, 
}
