from datasets import load_dataset
import os
import torch
torch.cuda.empty_cache()
from PIL import Image
from torch.optim import AdamW
import datetime
import pickle
import prepare_models
import json
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np

from peft import PeftModel
from peft import LoraConfig, get_peft_model
import collate_functions

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch

def get_combination_configs():
    # load a list of different configurations to process multiple


    models = ["combination"]
    layers_weights_to_merge = [[]]
    #
    weights = [[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]]
    # head_weights = [[0.5,0.5]]
    weights = [[0.1,0.9]]

    layers_features_to_merge = [[i for i in range(20,28)] + [-1],[25, 26, 27, -1],[-1]]
    # layers_features_to_merge = [],[i for i in range(10,28)] + [-1]
    head_weights = [[0.1,0.9],[0.9,0.1],[0.5,0.5],[0.3,0.7],[0.7,0.3]]
    # head_weights = [[0.1,0.9],[0.9,0.1]]
    head_weights = [[0.1,0.9]]
    

    configs = []
    for model in models:
        for layer_weights in layers_weights_to_merge:
            for weight in weights:
                print(model,layer_weights,weight[0])
                for layer_features in layers_features_to_merge:
                    for head_weight in head_weights:
                        configs.append([model,layer_weights,weight,layer_features,weight,head_weight])
    print(len(configs))
    return configs

lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # No bias in LoRA layers
    target_modules=[f"llm.model.layers.{i}.self_attn.q_proj" for i in range(0,28)] +  # LLM q_proj layers (for all 28 layers)
                   [f"llm.model.layers.{i}.self_attn.v_proj" for i in range(0,28)] +
                   [f"llm.model.layers.{i}.self_attn.k_proj" for i in range(0,28)] + 
                   #[f"visual.merger.mlp.{i}" for i in [0,2]],
                   ["combined_head"],#,  # LLM v_proj layers (for all 28 layers)
    # task_type="CAUSAL_LM"  # We are working on language models
    task_type="CAUSAL_LM"
)
def create_lora_model(model):
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

from torch.utils.data import DataLoader
    
total_steps = 100000  # Total number of training steps
warmup_steps = 1000  # Number of warmup steps

import math
from torch.optim.lr_scheduler import LambdaLR
def lr_lambda(step):
    if step < warmup_steps:
        # Warmup: Linear increase
        return float(step) / warmup_steps
    else:
        # Cosine Annealing after warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

def print_gpu_memory_usage():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
        print(f"  Total:     {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")

# Create the learning rate scheduler
def load_model(model_name,config,local_rank,resume,lr, pretrained_weights,use_lora):
    # loads the model with the defined configuration and resumes training if set
    
    model_components = None
    llm_name = model_name.split("_")[1]
    vlm_name = model_name.split("_")[2]
    model_name_, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,feature_weights,merge_head_weights,get_all_vlm_features_first,apply_on_entire_state,sum_weight_feature,learn_feature_weights,use_same_index = config
    if("combination" in model_name):
        version = "new"
        
        if(version == "new"):
            print(config)
            from combine_models import Combination
            model_components = [Combination(llm_name,vlm_name,merge_head_weights=merge_head_weights,merge_layer_weights_layers=[],
                                        merge_layer_weights_weights=merge_layer_weights_weights,get_all_vlm_features_first=get_all_vlm_features_first,
                                        merge_feature_layers=merge_feature_layers,feature_weights=feature_weights,mode="train",
                                        apply_on_entire_state=apply_on_entire_state,sum_weight_feature=sum_weight_feature,
                                        local_rank=local_rank,learn_feature_weights=learn_feature_weights,get_all_llm_features_first=False,use_same_index=use_same_index)]

        
        else:
            # llm_name, vlm_name,merge_head_weights,merge_layer_weights_layers,merge_layer_weights_weights
            from combine_models_old import Combination
            model_components = [Combination(llm_name,vlm_name,merge_head_weights=merge_head_weights,merge_layer_weights_layers=[],
                                        merge_layer_weights_weights=merge_layer_weights_weights,merge_feature_layers=merge_feature_layers,
                                        feature_weights=feature_weights,get_all_vlm_features_first=True,
                                        apply_on_entire_state=True,sum_weight_feature=True)]
    else:
        model_components = prepare_models.preprocess(model_name)
    from safetensors.torch import load_file
    print(model_components)
    
    model_components[0].to(local_rank)
    print(resume)
    
    if(resume):
        if(".pth" in pretrained_weights):
            print("load checkpoint")
            model_components[0] = create_lora_model(model_components[0])

            model_components[0].to(local_rank)
            checkpoint = torch.load(pretrained_weights,map_location=f"cuda:{local_rank}")
            model_components[0].load_state_dict(checkpoint['model_state_dict'])

            optimizer = AdamW(model_components[0].parameters(), weight_decay=0.1, lr=lr)
            scheduler = LambdaLR(optimizer, lr_lambda)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            del checkpoint
        else:
            print("load adapter")
            model_components[0] = PeftModel.from_pretrained(model_components[0], pretrained_weights)  # Apply adapter to the base model
            for name, param in model_components[0].named_parameters():
                if "lora" not in name:  # Exclude the specific parameter 'lora'
                    param.requires_grad = False
                    # print("" + name)
                else:
                    param.requires_grad = True
                    print(name)
            model_components[0].to(local_rank)
            optimizer = AdamW(model_components[0].parameters(), weight_decay=0.1, lr=lr)
            scheduler = LambdaLR(optimizer, lr_lambda)

    else:
        if(use_lora):
            model_components[0] = create_lora_model(model_components[0])
        else:
            for name, param in model_components[0].named_parameters():
                # if "combined_head" not in name or "lora" in name:  # Exclude the specific parameter 'lora'
                if "vlm.visual" in name:  # Exclude the specific parameter 'lora'
                    param.requires_grad = False
                    print("no_grad: " + name)
                else:
                    param.requires_grad = True
                    # print(name)

        model_components[0].to(local_rank)
        optimizer = AdamW(model_components[0].parameters(), weight_decay=0.1, lr=lr)
        scheduler = LambdaLR(optimizer, lr_lambda)

    
    trainable_params = [param for param in model_components[0].parameters() if param.requires_grad]
    num_trainable_params = sum(param.numel() for param in trainable_params)

    if(local_rank == 0):
        print(model_components[0])

        print(f"Number of trainable parameters: {num_trainable_params}")
    
    return model_components, optimizer, scheduler
from tqdm import tqdm
def compute_gradient_norms(model):
    # Compute the gradient norm for LoRA parameters
    lora_params = [p for name, p in model.named_parameters() if 'lora' in name]
    lora_gradients = [p.grad for p in lora_params if p.grad is not None]
    lora_grad_norm = torch.norm(torch.stack([torch.norm(g).to(torch.float32) for g in lora_gradients]), p=2).item() if lora_gradients else 0

    # Compute the overall gradient norm
    all_params = [p for p in model.parameters() if p.grad is not None]
    all_grad_norm = torch.norm(torch.stack([torch.norm(g).to(torch.float32) for g in all_params]), p=2).item() if all_params else 0

    return lora_grad_norm, all_grad_norm
def setup():
    # Initialize the distributed environment.
    dist.init_process_group("nccl", init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def train(model_name,config,parallel_,resume, lr, pretrained_weights, train_data_path,use_lora):

    if(parallel_):
        # setup parallel training
        setup()
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    model_components, optimizer, scheduler = load_model(model_name,config,local_rank,resume,lr, pretrained_weights,use_lora)

    if(parallel_):
        # prepare model for multi gpu training
        model_components[0] = DDP(model_components[0], device_ids=[local_rank],find_unused_parameters=True)
        dist.barrier()
        

    # Training loop
    import numpy as np
    
    train_dataset = load_dataset('parquet', data_files=train_data_path, split="train")
    
    
    train_batch_size = 1
    if(parallel_):
        # prepare data for multi gpu training
        sampler = DistributedSampler(train_dataset)
        num_gpus = torch.cuda.device_count()
        accumulation_steps = 8 / num_gpus / train_batch_size

        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler,collate_fn=lambda batch: collate_functions.collate_fn(batch, model_components, mode="train",model_name=model_name,llm_prompt_for_vision=False))
           
    
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,collate_fn=lambda batch: collate_functions.collate_fn(batch, model_components, mode="train",model_name=model_name,llm_prompt_for_vision=False))
    
        accumulation_steps = 8

    
    
    run_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if(parallel_ == False or rank == 0):
        # save config
        os.makedirs(f"runs/{run_path}/",exist_ok=True)
        with open(f'runs/{run_path}/config_.json', 'w') as json_file:
            data = {
                "config":list(config),
                "parallel":parallel_,
                "resume":resume, 
                "lr":lr, 
                "pretrained_weights":pretrained_weights,
                
                "lora_r":lora_config.r,
                "lora_alpha":lora_config.lora_alpha,
                "lora_bias":lora_config.bias,
                "lora_modules":list(lora_config.target_modules),
            }
            json.dump(data, json_file, indent=4)

    max_grad_norm = 5.0
    model = model_components[0]
    # set padding side
    try:
        model.module.base_model.model.vlm_processor.padding_side="left"
        model.module.base_model.model.tokenizer_llm.padding_side="left"
    except:
        model.module.vlm_processor.padding_side="left"
        model.module.tokenizer_llm.padding_side="left"
    
    losses = []
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        losses_per_epoch = []
        datas = []

        for step_nr, batch in enumerate(loop):

            if("combination" in model_name):
                #transfer to gpu
                batch["model_inputs_vlm"] = {key: value.to(local_rank) for key, value in batch["model_inputs_vlm"].items()}
                batch["model_inputs_llm"] = {key: value.to(local_rank) for key, value in batch["model_inputs_llm"].items()}
                batch.pop("video_path")
                batch.pop("prompt_llm")
                batch.pop("prompt_vlm")
            else:
                batch = {key: value.to(local_rank) for key, value in batch.items()}
                
            labels = batch.pop("labels",None)
            
            outputs = model(**batch,labels=labels,return_dict=True)
            
            if(isinstance(outputs,list)):
                loss = outputs[0].loss
            else:
                loss = outputs.loss
            # Backward pass
            loss.backward()
            # print_gpu_memory_usage()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if (step_nr // train_batch_size + 1) % accumulation_steps == 0:
                optimizer.step()  # Update model weights using accumulated gradients
                optimizer.zero_grad()  # Zero gradients after the update
                scheduler.step()
                print(scheduler.get_last_lr())
                
            if parallel_:
                dist.barrier()
            if (step_nr % 10000) == 0 and step_nr > 0 and (parallel_ == False or rank == 0):
                os.makedirs(f'runs/{run_path}',exist_ok=True)
                    
                if (step_nr % 20000) == 0:
                    #save checkpoint
                    checkpoint = {
                        
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                        'epoch': epoch,  # Optional: track the current epoch
                        'batch_idx': step_nr,
                    }

                    torch.save(checkpoint, f'runs/{run_path}/checkpoint_{step_nr + 1}_epoch_{epoch+1}.pth')
                losses.append(losses_per_epoch)
                losses_per_epoch = []
                with open(f'runs/{run_path}/losses_{epoch + 1}.pkl', 'wb') as f:
                    pickle.dump(losses, f)
            


            loop.set_postfix(loss=loss.item())
            
            losses_per_epoch.append(loss.item())
            del batch
            del loss
        
        if parallel_ == False or rank == 0:
            losses.append(losses_per_epoch)
    # save final model        
    if parallel_ == False or rank == 0:
            os.makedirs(f'runs/{run_path}',exist_ok=True)
            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'epoch': epoch,  # Optional: track the current epoch
            }

            torch.save(checkpoint, f'runs/{run_path}/checkpoint_{1}_epoch.pth')
            losses.append(losses_per_epoch)
            with open(f'runs/{run_path}/losses_{1}.pkl', 'wb') as f:
                pickle.dump(losses, f)

    if parallel_:
        dist.barrier()

        
    if parallel_:
        dist.destroy_process_group()

import argparse

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description="available arguments are: model_name, train_data_path, resume, pretrain_path, use_lora, lr")

    parser.add_argument('--model_name', type=str, help='name of the model to use can be: "Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-2B-Instruct","lmms-lab/llava-onevision-qwen2-7b-ov","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-7B-Instruct","combination"')
    parser.add_argument('--train_data_path', type=str, help='path to the generated dataset .parquet file for the train set')
    parser.add_argument('--resume', type=str, help='if training should be resumed',default=False)
    parser.add_argument('--pretrain_path', type=str, help='path to the checkpoint file',default="none")
    parser.add_argument('--use_lora', type=str, help='if lora should be used for training',default=True)
    parser.add_argument('--lr', type=str, help='learning rate', default=5e-5)
    parser.add_argument('--llm_name', type=str, help='llm to apply with the combination')
    parser.add_argument('--vlm_name', type=str, help='vlm to apply with the combination')
    
    
    args = parser.parse_args()
    use_lora = args.use_lora
    
    random_seed = 42
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    

    ## other options
    # llm_name = "Qwen/Qwen2.5-7B-Instruct"#"Qwen/Qwen2.5-7B-Instruct"#""#"Qwen/Qwen2-7B-Instruct"#"Qwen/Qwen2-7B-Instruct"#"Qwen/Qwen2-7B-Instruct-AWQ"#
    # vlm_name = "Qwen/Qwen2-VL-7B-Instruct"#"lmms-lab/LLaVA-Video-7B-Qwen2"#"lmms-lab/llava-onevision-qwen2-7b-ov"#

    llm_name =  args.llm_name 
    vlm_name =  args.vlm_name

    models = [args.model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = get_combination_configs()
    
    trans_dict = {}
    config = configs[0]
    print(config)
    
    if(len(config) == 1):
            model_name = config[0]
    else:
        model_name, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,feature_weights,merge_head_weights = config

        
    model_name = "combination_{}_{}".format(llm_name,vlm_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file_json = f'out/results_{timestamp}.json'
    
    parallel_ = True
    pretrained_weights = args.pretrain_path
    # resume = bool(args.resume)
    if args.resume.lower() == 'true':
        resume = True
    elif args.resume.lower() == 'false':
        resume = False

    lr=args.lr#5e-5
    lr = float(lr)
    

    print("lr: ",lr)
    #get_all_vlm_features_first,apply_on_entire_state,sum_weight_feature,learn_feature_weights,use_same_index
    config = config + [True,True,False,False,False]
    
    print(timestamp)
    print(resume)
    
    train(model_name,config,parallel_,resume,lr,pretrained_weights,args.train_data_path,use_lora)

