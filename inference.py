
from datasets import load_dataset
import os
import torch
from PIL import Image
from torch.optim import AdamW
import datetime
import pickle
import prepare_models
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import utils
import random
import numpy as np
from peft import LoraConfig, get_peft_model
from combine_models import Combination
import collate_functions


import torch.distributed as dist
import torch

def get_combination_configs(models,vlm_name,llm_name):


    # models = ["combination"]
    layers_weights_to_merge = [[]]
    #
    weights = [[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]]

    layers_features_to_merge = [[25, 26, 27, -1],[i for i in range(20,28)] + [-1],[-1],]
    head_weights = [[0.1,0.9],[0.9,0.1],[0.3,0.7],[0.7,0.3]]
    
    sum_weights = [False,True]
    apply_on_entire = [True,False]
    get_vlm_feat_firsts = [True,False]
    get_llm_feat_firsts = [False]

    ## 
    configs = []
    for model in models:
        if("combination" in model):
            model_name = "combination_{}_{}".format(vlm_name,llm_name)
        else:
            model_name = model
        for layer_weights in layers_weights_to_merge:
            for weight in weights:
                for layer_features in layers_features_to_merge:
                    for head_weight in head_weights:
                        for sum_ in sum_weights:
                            for apply_ in apply_on_entire:
                                for get_vlm_feat_first in get_vlm_feat_firsts:
                                    for get_llm_feat_first in get_llm_feat_firsts:
                                        configs.append([model_name,layer_weights,weight,layer_features,weight,head_weight,sum_,apply_,get_vlm_feat_first,get_llm_feat_first])
                        
    print(len(configs))
    return configs

lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # No bias in LoRA layers
    target_modules=[f"model.layers.{i}.self_attn.q_proj" for i in range(15,28)] +  # LLM q_proj layers (for all 28 layers)
                   [f"model.layers.{i}.self_attn.v_proj" for i in range(15,28)] + 
                   [f"model.layers.{i}.self_attn.k_proj" for i in range(15,28)] + 
                   ["combined_head"],  # LLM v_proj layers (for all 28 layers)
    task_type="CAUSAL_LM"  # We are working on language models
)
def create_lora_model(model,lora_config):

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

from torch.utils.data import DataLoader

def evaluation_(batch,model_components,gpu_id,model_name,datas,val_data_path,adapter_model_dir,output_file_json,descr="",version="new"):

    # performs the inference run on the batch for the desired model

    labels = batch.pop("labels",None)

    batch["do_sample"] = False
    if("combination" in model_name):
        model = model_components[0]
        model.eval()
        processor = model.vlm_processor
        tokenizer_llm = model.tokenizer_llm
        processor.padding_side="left"
        tokenizer_llm.padding_side="left"
        batch_gpu = {}
        batch_gpu["model_inputs_vlm"] = {}
        for key, value in batch["model_inputs_vlm"].items():
            if(type(value) != list):
                batch_gpu["model_inputs_vlm"][key] = value.to(gpu_id)
            else:
                batch_gpu["model_inputs_vlm"][key] = value
        batch_gpu["model_inputs_llm"] = {key: value.to(gpu_id) for key, value in batch["model_inputs_llm"].items()}
        
        prompt_llm=batch["prompt_llm"]
        prompt_vlm=batch["prompt_vlm"]
        with torch.no_grad():
            
            batch_gpu["model_inputs_llm"]["max_new_tokens"] = 300
            batch_gpu["model_inputs_vlm"]["max_new_tokens"] = batch_gpu["model_inputs_llm"]["max_new_tokens"]
            return_dict = True
            output_hidden_states = True
            if(version == "new"):
                
                outputs = model.generate_(kwargs_llm=batch_gpu["model_inputs_llm"],
                                         kwargs_vlm=batch_gpu["model_inputs_vlm"],
                                         max_new_tokens=batch_gpu["model_inputs_llm"]["max_new_tokens"])

            else:
                outputs = model.generate(kwargs_llm=batch["model_inputs_llm"],kwargs_vlm=batch["model_inputs_vlm"],return_dict=return_dict,
                                        output_hidden_states=output_hidden_states,
                                            merge_feature_layers=model.merge_feature_layers,feature_weights=model.feature_weights)
            input_ids_trimmed = [
            out_ids[:len(in_ids)] for in_ids, out_ids in zip(batch["model_inputs_llm"]["input_ids"], outputs)
            ]
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["model_inputs_llm"]["input_ids"], outputs)
            ]
            input_llm = ["".join(tokenizer_llm.batch_decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)) for generated_id in input_ids_trimmed]
            response_llm = ["".join(tokenizer_llm.batch_decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)) for generated_id in generated_ids_trimmed]
    
    elif("Qwen2-VL" in model_name):
        model,processor = model_components
        prompt_vlm=batch["prompt_vlm"]
        
        batch_gpu = {}
        batch_gpu["model_inputs_vlm"] = {}
        batch_gpu["model_inputs_vlm"] = {key: value.to(0) for key, value in batch["model_inputs_vlm"].items()}


        batch_gpu = batch_gpu["model_inputs_vlm"]
        with torch.no_grad():
            outputs = model.generate(**batch_gpu,max_new_tokens=300)
        input_ids_trimmed = [
        out_ids[:len(in_ids)] for in_ids, out_ids in zip(batch_gpu["input_ids"], outputs)
        ]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_gpu["input_ids"], outputs)
        ]
        input_llm = ["".join(processor.batch_decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)) for generated_id in input_ids_trimmed]
        response_llm = ["".join(processor.batch_decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)) for generated_id in generated_ids_trimmed]
    elif("Qwen2.5-7B-Instruct" in model_name or "Qwen2.5-1.5B-Instruct" in model_name):
        model,processor = model_components
        prompt_llm=batch["prompt_llm"]
        prompt_vlm = [""] * 5
        
        batch_gpu = {}
        batch_gpu["model_inputs_llm"] = {}
        batch_gpu["model_inputs_llm"] = {key: value.to(0) for key, value in batch["model_inputs_llm"].items()}


        batch_gpu = batch_gpu["model_inputs_llm"]
        with torch.no_grad():
            outputs = model.generate(**batch_gpu,max_new_tokens=300)
        input_ids_trimmed = [
        out_ids[:len(in_ids)] for in_ids, out_ids in zip(batch_gpu["input_ids"], outputs)
        ]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_gpu["input_ids"], outputs)
        ]
        input_llm = ["".join(processor.batch_decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)) for generated_id in input_ids_trimmed]
        response_llm = ["".join(processor.batch_decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)) for generated_id in generated_ids_trimmed]
    elif("LLaVA" in model_name):
        model,processor,tokenizer = model_components
        prompt_vlm=batch["prompt_vlm"]
        
        batch_gpu = {}
        batch_gpu["model_inputs_vlm"] = {}
        for key, value in batch["model_inputs_vlm"].items():
            if(isinstance(value,list)):
                batch_gpu["model_inputs_vlm"][key] = value
            else:

                batch_gpu["model_inputs_vlm"][key] = value.to(0)
                
        batch_gpu = batch_gpu["model_inputs_vlm"]
        
        with torch.no_grad():
            outputs = model.generate(
                batch_gpu["input_ids"],
                attention_mask = batch_gpu["attention_mask"],
                images=batch_gpu["images"],
                modalities=batch_gpu["modalities"],
                do_sample=False,
                temperature=0,
                max_new_tokens=300,
            )
        response_llm = tokenizer.batch_decode(outputs, skip_special_tokens=True)#.strip()
        input_llm = ["","","","",""]


    if("combination" in model_name):
        data = [{
            "model_name": "combination_" + vlm_name + "_" + llm_name,
            "question": question,
            "scene_token": scene,
            "task_token": id_,
            "llm_prompt": llm_prompt,
            "vlm_prompt": vlm_prompt,
            "result": result,
            "merge_feature_layers":model_components[0].merge_feature_layers,
            "feature_weights":model_components[0].feature_weights,
            "merge_head_weights":model_components[0].merge_head_weights,
            "merge_layer_weights_layers":model_components[0].merge_layer_weights_layers,
            "merge_layer_weights_weights":model_components[0].merge_layer_weights_weights,
            "apply_on_entire_state":model_components[0].apply_on_entire_state,
            "sum_weight_feature":model_components[0].sum_weight_feature,
            # "mode":model_components[0].mode,
            "descr":descr,
            "get_all_vlm_features_first":model_components[0].get_all_vlm_features_first,
            "get_all_llm_features_first":model_components[0].get_all_llm_features_first,
            "next_comb_temp":False,
            "learn_feature_weights":model_components[0].learn_feature_weights,
            "use_same_index":model_components[0].use_same_index,
            "val_data_path":val_data_path,
            "save_path":adapter_model_dir,
            "sample":model_components[0].sample
        } for question, scene, id_, llm_prompt, vlm_prompt, result in zip(input_llm, batch["scene_id"],batch["task_id"], prompt_llm,prompt_vlm,response_llm)]
    else:
        data = [{
            "model_name": model_name,
            "question": question,
            "scene_token": scene,
            "task_token": id_,
            "vlm_prompt": vlm_prompt,
            "result": result,
            "descr":descr,
            "next_comb_temp":False,
            "val_data_path":val_data_path,
            "save_path":adapter_model_dir
        } for question, scene, id_, vlm_prompt, result in zip(input_llm, batch["scene_id"],batch["task_id"],prompt_vlm,response_llm)]
     
     
    if("train" not in descr):
        if(adapter_model_dir != "none"):
            output_file_json = adapter_model_dir + ".json"
        with open(output_file_json, 'w') as json_file:
            datas = datas + data
            json.dump(datas, json_file, indent=4)
            
    return datas, input_llm, labels, response_llm
    
import math
def print_gpu_memory_usage():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
        print(f"  Total:     {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")

from tqdm import tqdm


def get_val_loss(model_components,model_name,adapter_model_dir,val_data_path):
    
    
    val_dataset = load_dataset('parquet', data_files=val_data_path, split="train")#[:100]
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False,collate_fn=lambda batch: collate_functions.collate_fn(batch, model_components, mode="val_loss",llm_prompt_for_vision=False,model_name=model_name))
    losses = []

    for batch in tqdm(val_dataloader):
        if("combination" in model_name):
            # print(batch)
            batch["model_inputs_vlm"] = {key: value.to("cuda:0") for key, value in batch["model_inputs_vlm"].items()}
            batch["model_inputs_llm"] = {key: value.to("cuda:0") for key, value in batch["model_inputs_llm"].items()}
            batch.pop("video_path")
            batch.pop("prompt_llm")
            batch.pop("prompt_vlm")
        else:
            batch = {key: value.to("cuda:0") for key, value in batch.items()}
        
        labels = batch.pop("labels",None)
        # Forward pass
        
        model = model_components[0]
        model.eval()
        with torch.no_grad():
            outputs = model(**batch,labels=labels,return_dict=True)
        
            if(isinstance(outputs,list)):
                loss = outputs[0].loss
            else:
                loss = outputs.loss
                
            losses.append(loss.item())
        del batch
        del loss
        del outputs
        # break
    path_ = "/".join(adapter_model_dir.split("/")[:-1])
    print(sum(losses)/len(losses))
    save_path = ""
    print(adapter_model_dir)
    if(".pth" in adapter_model_dir):
        save_path = adapter_model_dir.replace("runs/","").replace("pth","").replace("/","_")
    else:
        save_path = adapter_model_dir.replace("runs/","").replace("pth","").replace("/","_")
        save_path = 0#"losses.pkl"
        # path_ = 
    path_ = f'{path_}/losses_{save_path}.pkl'

    with open(path_.replace("/losses","losses"), 'wb') as f:
        pickle.dump(losses, f)
    
    return

def inference(model_components,gpu_id,model_name,val_data_path,version,adapter_model_dir,output_file_json,llm_prompt_for_vision):
    #iterates over the validation batches

    val_dataset = load_dataset('parquet', data_files=val_data_path, split="train[:1000]")#[:100]
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False,collate_fn=lambda batch: collate_functions.collate_fn(batch, model_components, mode="val",model_name=model_name,llm_prompt_for_vision=llm_prompt_for_vision))

    datas = []
    scores = []
    for batch in tqdm(val_dataloader):
        datas, input_llm, labels, response_llm = evaluation_(batch,model_components,gpu_id,model_name,datas,val_data_path,adapter_model_dir,output_file_json,version=version)

        del batch
    del model_components
from peft import PeftModel


def convert_pth_to_adapter(model_components,adapter_model_dir):

    model_components[0] = create_lora_model(model_components[0])
    checkpoint = torch.load(adapter_model_dir,map_location=f"cuda:0")
    model_components[0].load_state_dict(checkpoint['model_state_dict'])
    corrected_path = adapter_model_dir.replace(".pth","")
    model_components[0].save_pretrained(corrected_path)
    print(corrected_path)

def get_saved_peft(model_name,dirs,base_dir):
    # finds all saved checkpoints in the "run" directory
    files = []
    c = 0
    for dir_ in dirs:
        if("." in dir_):
            continue
        val_files = os.listdir(base_dir + dir_)
        for val_file in val_files:
            if("checkpoint" not in val_file or ".pkl" in val_file or ".json" in val_file or ".pth" not in val_file):
                continue
                
            model_dir = base_dir + dir_ + "/"
            adapter_model_dir = model_dir + "/" + val_file
            
            model_components = None
            if("combination" in model_name):
                version = "new"
                
                if(version == "new"):
                    get_all_vlm_features_first=False
                    apply_on_entire_state=True
                    sum_weight_feature=True
                    try:
                        # print(model_dir + "config_.json")
                        with open(model_dir + "config_.json", 'r') as file:
                            conf_ = json.load(file)
                        conf_ = conf_["config"] 
                        # print(conf_)
                        get_all_vlm_features_first = get_all_vlm_features_first
                        apply_on_entire_state = conf_[7]
                        sum_weight_feature = conf_[8]
                        feature_weights = conf_[2]
                        merge_feature_layers = conf_[3]
                        merge_head_weights = conf_[2]
                        # print(conf_[2:9])
                        c += 1
                        files.append(adapter_model_dir)
                    except:
                        continue
    # print(c)
    return files
def find_key_with_array_value(dictionary, array):
    for key, value in dictionary.items():
        # print(value,array)
        if value == array:
            return key
    return None  # Return None if array is not found as a value

import argparse


def load_checkpoint_config(adapter_model_dir,configs):
    # loads the config for a checkpoint to load the correct model configuration
    try:
        model_dir = "/".join(adapter_model_dir.split("/")[:-1])
        print(model_dir + "config_.json")
        with open(model_dir + "config_.json", 'r') as file:
            data = json.load(file)
        conf_ = data["config"]

        # print(conf_)
        get_all_vlm_features_first = conf_[6]
        apply_on_entire_state = conf_[7]
        sum_weight_feature = conf_[8]
        feature_weights = conf_[2]
        merge_feature_layers = conf_[3]
        merge_head_weights = conf_[2]
        
        dat_ = {dat:data[dat] for dat in data if dat != "others"}


        configs_ = {}
        # print(configs)
        for file in configs:
            print(configs,file)
            for dat in configs[file]:
                # print(configs[file][dat])
                if dat != "others":

                    if file not in configs_:
                        configs_[file] = {}
                    configs_[file][dat] = configs[file][dat]
                    

        contained = find_key_with_array_value(configs_, dat_)
        
        if(contained is None):
            dat_ = {dat:data[dat] for dat in data if dat != "pretrained_weights"}
            configs[adapter_model_dir] = data
            configs[adapter_model_dir]["others"] = []
        else:
            configs[contained]["others"].append(adapter_model_dir) 
    except Exception as e:
        print(f"{adapter_model_dir}: {e}")
        print("C"+5)
        # continue
        get_all_vlm_features_first = True
        apply_on_entire_state = True
        sum_weight_feature = True
        feature_weights = [0.9,0.1]
        merge_feature_layers = [0.9,0.1]
        merge_head_weights = [25,26,27,-1]
    return get_all_vlm_features_first, apply_on_entire_state, sum_weight_feature, feature_weights,merge_feature_layers,merge_head_weights, configs
def load_checkpoint(model_components,adapter_model_dir,use_pth):
    # loads a checkpoint and the corresponding model
    if(use_pth):
        try:
            lora_config_ = LoraConfig.from_pretrained(adapter_model_dir.replace(".pth","").replace("checkpoint_","fine_tuned_model_"))
        except:
                
            with open(adapter_model_dir.replace(adapter_model_dir.split("/")[-1],"config_.json"), 'r') as file:
                data__ = json.load(file)
            print(data__)
            lora_config_ = LoraConfig(
                r = data__["lora_r"],
                lora_alpha = data__["lora_alpha"],
                bias = data__["lora_bias"],
                target_modules = data__["lora_modules"],
                task_type = lora_config.task_type,
                lora_dropout=lora_config.lora_dropout
            )
        # model_components[0] = create_lora_model(model_components[0],lora_config_)
        try:
            checkpoint = torch.load(adapter_model_dir)#,map_location=f"cuda:0")
        except Exception as e:
            print(e)
        use_lora = False
        for name in checkpoint['model_state_dict']:
            # print(name)
            if("base_model" in name):
                use_lora = True
                break
        if(use_lora):
            model_components[0] = create_lora_model(model_components[0],lora_config_)
        keys = model_components[0].load_state_dict(checkpoint['model_state_dict'],strict=False)
        print(keys)
        del checkpoint
    else:
        # adapter_model_dir = base_dir# + dir_
        print("load adapter")
        model_components[0] = PeftModel.from_pretrained(model_components[0], adapter_model_dir)  # Apply adapter to the base model
    return model_components[0]


def inference_on_checkpoints(model_name,llm_name,vlm_name,val_data_path,args):
    #performs inference on all the checkpoints in the run directory

    base_dir = f"runs/"
    if(args.arg1 == "yes"):
        print("reverse")
        dirs = os.listdir(base_dir)[::-1]
    else:
        dirs = os.listdir(base_dir)

    # print(len(dirs))
    dirs = [dir_ for dir_ in dirs if "." not in dir_]

    saved_pefts = get_saved_peft(model_name,dirs,base_dir)#[::-1]
    
    print("len_saved_pefts: ", len(saved_pefts))
    print(saved_pefts)

    #50001,
    # if(run_nr < 60):
    #     continue
    gpu_id = 0
    c = 0
    # val_data_mostfreq#_per_object
    val_files = ["val_data"]
    
    train_runs = ["20250214_234138"]
    from_config = True
    configs = {}
    for train_run in train_runs:
        for adapter_model_dir in saved_pefts:
            print(adapter_model_dir)
            for val_file in val_files:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                print(timestamp)
                # output_file = f'out/results_{timestamp}.txt'
                output_file_json = f'out/results_{timestamp}.json'
                try:
                    with open(output_file_json, 'r') as file:
                        data__ = json.load(file)
                    output_file_json = output_file_json.replace(".json","_1.json")
                except:
                    pass

                print("model_name: ",model_name)
                if(model_name[0] == "combination"):
                    model_name = model_name[0] + "_" + llm_name + "_" + vlm_name

                print("adapter: " + adapter_model_dir)
                resume = True
                
                
                if("model_epoch" in adapter_model_dir):
                    epoch_notin = True
                    epochs = list(np.arange(0,1000, 10)) + list(np.arange(10))
                    print(epochs)
                    file_name = adapter_model_dir.split("/")[-1]
                    print(file_name)
                    for epoch in epochs:
                        if(f"epoch_{epoch}_epoch" in file_name):
                            epoch_notin = False
                            break
                    if(epoch_notin):
                        continue
                    
                model_components = None

                #loading of different models
                if("combination" in model_name):
                    version = "new"
                    if(version == "new"):
                        get_all_vlm_features_first=True
                        apply_on_entire_state=True
                        sum_weight_feature=True
                        if(from_config and resume):
                            get_all_vlm_features_first, apply_on_entire_state, sum_weight_feature, feature_weights,merge_feature_layers,merge_head_weights, configs = load_checkpoint_config(adapter_model_dir,configs)
                            
                        model_components = [Combination(llm_name,vlm_name,merge_head_weights=merge_head_weights,merge_layer_weights_layers=[],
                                                    merge_layer_weights_weights=None,get_all_vlm_features_first=get_all_vlm_features_first,
                                                    merge_feature_layers=merge_feature_layers,feature_weights=feature_weights,mode="val",
                                                    apply_on_entire_state=apply_on_entire_state,sum_weight_feature=sum_weight_feature,get_all_llm_features_first=False,
                                                    local_rank=0,use_same_index=False)]

            
                else:
                    model_components = prepare_models.preprocess(model_name)
                use_pth = True if ".pth" in adapter_model_dir else False
                if(resume):
                    model_components[0] = load_checkpoint(model_components,adapter_model_dir,use_pth)
                else:
                    adapter_model_dir = "none"
                model_components[0].to(gpu_id)

                inference(model_components,gpu_id,model_name,val_data_path,version,adapter_model_dir,output_file_json,False)
                del model_components
                torch.cuda.empty_cache()

def loop_predefined_configs(models,llm_name,vlm_name,configs,llm_prompt_for_vision,args,val_data_path,sample,nr_repeats):

    ### loops over all configurations that are input as "configs" and repeats them "nr_repeats" times
    for run_nr, config in enumerate(configs):
        for repeat in range(nr_repeats):
            # if(run_nr < 3):
            #     continue
            # if(run_nr != 2 and run_nr != 3 and run_nr != 5 and run_nr != 9 and run_nr != 28):
            #     continue
            print(run_nr,config)
            # continue
            print(config)
            use_same_index = None
            if(len(config) == 1):
                model_name = config[0]
            elif(len(config) == 3):
                model_name, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,feature_weights,merge_head_weights = config[0]
                apply_on_entire_state, sum_weight_feature, get_all_vlm_features_first, next_comb_temp,learn_feature_weights,use_same_index = config[1]
                llm_name,vlm_name = config[2]
            elif(len(config) == 6):
                (model_name, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,
                feature_weights,merge_head_weights) = config
            else:
                (model_name, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,
                feature_weights,merge_head_weights,sum_weight_feature,apply_on_entire_state,get_all_vlm_features_first,get_all_llm_features_first) = config
                
                if(use_same_index is None):
                    use_same_index = False
            #
            datas = []
            train_ = False
            print(model_name)
            if("_" in model_name):
                model_parts = model_name.split("_")
                llm_name = model_parts[2]
                vlm_name = model_parts[1]

            if("combination" == model_name):
            # if("combination" in model_name):
                model_name = "combination_{}_{}".format(llm_name,vlm_name)
            
            print(model_name)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(timestamp)
            # output_file = f'out/results_{timestamp}.txt'
            output_file_json = f'out/results_{timestamp}.json'
            try:
                with open(output_file_json, 'r') as file:
                    data__ = json.load(file)
                output_file_json = output_file_json.replace(".json","_1.json")
            except:
                pass
            
            if("combination" in model_name):

                model_components = [Combination(llm_name,vlm_name,merge_head_weights=merge_head_weights,merge_layer_weights_layers=merge_layer_weights_layers,
                                    merge_layer_weights_weights=merge_layer_weights_weights,get_all_vlm_features_first=get_all_vlm_features_first,
                                    merge_feature_layers=merge_feature_layers,feature_weights=feature_weights,mode="val",
                                    apply_on_entire_state=apply_on_entire_state,sum_weight_feature=sum_weight_feature,get_all_llm_features_first=False,
                                    sample=sample,local_rank=0,use_same_index=use_same_index)]
            else:
                model_components = prepare_models.preprocess(model_name)
            # print(model_components)
            gpu_id = 0
            model_components[0].to(gpu_id)
            
            adapter_model_dir = "none"
            version = "new"

            inference(model_components,gpu_id,model_name,val_data_path,version,adapter_model_dir,output_file_json,llm_prompt_for_vision)
            del model_components
            torch.cuda.empty_cache()
        break
    
def get_missing_configs(base_path,configs):
    #returns the configs for which no inference file was evaluated, yet
    results = []
    with open(base_path + "eval_result.json", 'r') as file:
        results = json.load(file)  # 'indent' for pretty-printing
    
    data = []
    a = 0
    for file_name in results:

        
        try:
            with open(base_path + file_name, 'r') as file:
                file_data = json.load(file)
        except Exception as e:
            print(e)
            continue
        nr_samples = results[file_name]["nr_samples"]
        if(nr_samples != 100):
            continue
        first_info = file_data[0]
        # print(first_info["model_name"])

        if("combination" in first_info["model_name"]):
            
            model_name = first_info["model_name"]
            merge_feature_layers = first_info["merge_feature_layers"]
            feature_weights = first_info["feature_weights"]
            merge_head_weights = first_info["merge_head_weights"]
            merge_layer_weights_layers = first_info["merge_layer_weights_layers"]
            merge_layer_weights_weights = first_info["merge_layer_weights_weights"]
            
            apply_on_entire_state = first_info["apply_on_entire_state"]
            sum_weight_feature = first_info["sum_weight_feature"]
            get_all_vlm_features_first = first_info["get_all_vlm_features_first"]
            next_comb_temp = first_info["next_comb_temp"]
            learn_feature_weights = first_info["learn_feature_weights"]
            use_same_index = first_info["use_same_index"]
            arr = [model_name, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,
            feature_weights,merge_head_weights,sum_weight_feature,apply_on_entire_state,get_all_vlm_features_first]
            if(arr not in data):
                data.append(arr)
            # model,layer_weights,weight,layer_features,weight,head_weight,apply_,sum_,get_feat_firsts
        else:
            model_name = first_info["model_name"]
        a += 1
    print(a)
    
    print(len(data))
    missing_configs = []
    for config in configs:
        if(config not in data):
            print(config)
            missing_configs.append(config)
    return missing_configs
def load_best_configs(base_path):
    #returns the best overall configs
    
    path = base_path + "eval_result.json"

    with open(path, 'r') as file:
        data = json.load(file)

    configs = []
    for file_name in data:
        if(data[file_name]["lingo_mean"] < 0.44 or data[file_name]["nr_samples"] < 500):
            continue

        with open(base_path + file_name, 'r') as file:
            file_data = json.load(file)
        first_info = file_data[0]

        if("combination" in first_info["model_name"]):
            model_name = first_info["model_name"]
            print(model_name)
            if("1.5B" not in model_name):
                continue
            merge_feature_layers = first_info["merge_feature_layers"]
            feature_weights = first_info["feature_weights"]
            merge_head_weights = first_info["merge_head_weights"]
            merge_layer_weights_layers = first_info["merge_layer_weights_layers"]
            merge_layer_weights_weights = first_info["merge_layer_weights_weights"]
            
            apply_on_entire_state = first_info["apply_on_entire_state"]
            sum_weight_feature = first_info["sum_weight_feature"]
            get_all_vlm_features_first = first_info["get_all_vlm_features_first"]
            next_comb_temp = first_info["next_comb_temp"]
            learn_feature_weights = first_info["learn_feature_weights"]
            use_same_index = first_info["use_same_index"]
        else:
            model_name = first_info["model_name"]

        config = [(model_name, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,feature_weights,merge_head_weights), 
        (apply_on_entire_state, sum_weight_feature, get_all_vlm_features_first, next_comb_temp,learn_feature_weights,use_same_index),(llm_name,vlm_name)]
        if(config not in configs):
            configs.append(config)
            
    print("len_best: ",len(configs))
    return configs

if __name__ == "__main__":
    # Load VQA dataset

    parser = argparse.ArgumentParser(description="inference arguments are model_name, val_data_path, llm_prompt_for_vision.")

    # Add arguments
    parser.add_argument('--model_name', type=str, help='name of the model to use can be: "Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-2B-Instruct","lmms-lab/llava-onevision-qwen2-7b-ov","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-7B-Instruct","combination"')
    parser.add_argument('--val_data_path', type=str, help='path to the generated dataset .parquet file for the validation set')
    parser.add_argument('--llm_prompt_for_vision', type=str, help='use the llm prompt also for the LVLM? binary',default=False)
    parser.add_argument('--llm_name', type=str, help='llm to apply with the combination')
    parser.add_argument('--vlm_name', type=str, help='vlm to apply with the combination')
    parser.add_argument('--mode', type=str, help='inference mode to apply, "standard" loops over all configurations for the given model, "best_only" runs inference only on the best configurations that have ben tested already, "on_checkpoints" loops over all chackpoints for runs saved in "runs/" ')


    args = parser.parse_args()
    random_seed = 42
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    # llm_name = "Qwen/Qwen2.5-7B-Instruct"#"Qwen/Qwen2.5-7B-Instruct"#""#"Qwen/Qwen2-7B-Instruct"#"Qwen/Qwen2-7B-Instruct"#"Qwen/Qwen2-7B-Instruct-AWQ"#
    # vlm_name = "Qwen/Qwen2-VL-7B-Instruct"#"lmms-lab/llava-onevision-qwen2-7b-ov"#"lmms-lab/LLaVA-Video-7B-Qwen2"#
    # llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # vlm_name = "Qwen/Qwen2-VL-2B-Instruct"
    models = [args.model_name]
    val_data_path = args.val_data_path
    llm_prompt_for_vision = args.llm_prompt_for_vision
    llm_name =  args.llm_name 
    vlm_name =  args.vlm_name
    sample = False
    nr_repeats = 1


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if(args.mode == "standard"):
        configs = get_combination_configs(models,vlm_name,llm_name)
        loop_predefined_configs(models,llm_name,vlm_name,configs,llm_prompt_for_vision,args,val_data_path,sample,nr_repeats)

    elif(args.mode == "best_only"):
        base_path = "out/"
        configs = load_best_configs(base_path)
        loop_predefined_configs(models,llm_name,vlm_name,configs,llm_prompt_for_vision,args,val_data_path,sample,nr_repeats)

    elif(args.mode == "on_checkpoints"):
        inference_on_checkpoints(models,llm_name,vlm_name,args)




    # configs = get_missing_configs("workspace/LLaVA-NeXT/out/",configs)


    
    