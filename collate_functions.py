import torch

import prepare_models
def collate_fn_eval(examples):
    questions = [example["question"] for example in examples]
    results = [example["result"] for example in examples]
    scene_ids = [example["scene_token"] for example in examples]
    task_ids = [example["task_token"] for example in examples]
    run_names = [example["run_name"] for example in examples]

    return_dict = {
        "questions":questions,
        "results":results,
        "scene_ids":scene_ids,
        "task_ids":task_ids,
        "run_names":run_names,
    }
    return return_dict


def collate_fn_llava(examples,model_components,mode,model_name):
    if(len(model_components) == 1):
        model = model_components[0]
        processor = model.vlm_processor
        tokenizer_llm = model.tokenizer_llm
    elif(len(model_components) == 2):
        model,processor = model_components
    else:
        model,processor,tokenizer_llm = model_components
    video_paths = [example["video_path"] for example in examples]
    prompt_llm = [example["prompt_llm"].replace("\u00a0","") for example in examples]
    prompt_vlm = [example["prompt_vlm"].replace("\u00a0","") for example in examples]
    labels = [example["labels"] for example in examples]
    scene_ids = [example["scene_id"] for example in examples]
    task_ids = [example["task_id"] for example in examples]
    if(mode != "train"):
        video_paths = video_paths[::2]
        prompt_llm = prompt_llm[::2]
        prompt_vlm = prompt_vlm[::2]
        labels = labels[::2]
        scene_ids = scene_ids[::2]
        task_ids = task_ids[::2]
    examples = {
        "video_path":video_paths,
        "prompt_llm":prompt_llm,
        "prompt_vlm":prompt_vlm,
        "labels":labels,
    }
    processor.padding_side="left"
    tokenizer_llm.padding_side="left"
    add_generation_prompt = True
    videos = [utils.load_video(video_path, 16, 1, force_sample=True) for video_path in video_paths]
    videos_preprocessed = processor.preprocess(videos, return_tensors="pt")["pixel_values"].cuda().half()
    videos_preprocessed = [videos_preprocessed]
    
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instrucitons = [f"The video lasts for {video[2]:.2f} seconds, and {len(video_preprocessed[0])} frames are uniformly sampled from it. \
                        These frames are located at {video[1]}.Please answer the following questions related to this video." for video,video_preprocessed in zip(videos,videos_preprocessed)]
    question = [DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + prompt_vlm_ for prompt_vlm_,time_instruciton in zip(prompt_vlm,time_instrucitons)]
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    vlm_input_ids = tokenizer_image_token(prompt_question, model.tokenizer_vlm, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.vlm.device)

    vlm_data = {"input_ids":vlm_input_ids,
                "do_sample":False,
                "attention_mask":torch.ones(vlm_input_ids.shape[1]).to(torch.bfloat16),
        #    "images":[vid.to(torch.bfloat16) for vid in video],
        "images":videos_preprocessed,#[vid for vid in video],
        "modalities":["video"]}
        
    messages_llm = [
        [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt_llm}
        ] for prompt_llm in examples["prompt_llm"]
    ]
    text_llm = [tokenizer_llm.apply_chat_template(
        prompt_llm,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    ) for prompt_llm in messages_llm]
    llm_data = tokenizer_llm(text_llm, return_tensors="pt",padding=True)
    inputs = {}
    inputs["model_inputs_vlm"] = {
        "input_ids":vlm_data.pop("input_ids"),
        "attention_mask":vlm_data.pop("attention_mask"),
        "images":vlm_data.pop("images")
    }
    inputs["model_inputs_llm"] = {
        "input_ids":llm_data["input_ids"],
        "attention_mask":llm_data["attention_mask"],
    }
    inputs["video_path"] = video_paths
    inputs["prompt_llm"] = prompt_llm
    inputs["prompt_vlm"] = prompt_vlm

    inputs["scene_id"] = scene_ids
    inputs["task_id"] = task_ids
    # print(time.time()-curr)
    return inputs

def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_fn(examples,model_components,mode,model_name,llm_prompt_for_vision):
    
    if("combination" in model_name):
        model = model_components[0]
        if(mode == "train"):
            try:
                processor = model.module.base_model.model.vlm_processor
                tokenizer_llm = model.module.base_model.model.tokenizer_llm
            except:
                processor = model.vlm_processor
                tokenizer_llm = model.tokenizer_llm

        else:
            # print(model)
            try:
                processor = model.base_model.model.vlm_processor
                tokenizer_llm = model.base_model.model.tokenizer_llm
            except:

                processor = model.vlm_processor
                tokenizer_llm = model.tokenizer_llm
    else:
        model = model_components[0]
        processor = model_components[1]
        try:
            tokenizer_llm = model_components[2]
        except:
            tokenizer_llm = None


    video_paths = [example["video_path"] for example in examples]
    # print(video_paths)
    prompt_llm = [example["prompt_llm"].replace("\u00a0","") for example in examples]
    prompt_vlm = [example["prompt_vlm"].replace("\u00a0","") for example in examples]
    labels = [example["labels"] for example in examples]
    scene_ids = [example["scene_id"] for example in examples]
    task_ids = [example["task_id"] for example in examples]
    if(mode != "train" and mode != "val_loss"):
        video_paths = video_paths[::2]
        prompt_llm = prompt_llm[::2]
        prompt_vlm = prompt_vlm[::2]
        labels = labels[::2]
        scene_ids = scene_ids[::2]
        task_ids = task_ids[::2]
    


    examples = {
        "video_path":video_paths,
        "prompt_llm":prompt_llm,
        "prompt_vlm":prompt_vlm,
        "labels":labels,
    }
    
    
    if("LLaVA" in model_name):
        vlm_data, add_generation_prompt = prepare_models.LLaVA_Video_7B_Qwen2(model,examples,mode,processor,tokenizer_llm,llm_prompt_for_vision)
    elif("Qwen2_VL" in model_name or "Qwen2-VL" in model_name):
        vlm_data, add_generation_prompt = prepare_models.Qwen2_VL_7B_Instruct(model,examples,mode,processor,tokenizer_llm,llm_prompt_for_vision)
    else:
        vlm_data = None
        add_generation_prompt = True

    # print(model_name)
    # llm_name = "Qwen/Qwen2.5-7B-Instruct"#"Qwen/Qwen2.5-7B-Instruct"#""#"Qwen/Qwen2-7B-Instruct"#"Qwen/Qwen2-7B-Instruct"#"Qwen/Qwen2-7B-Instruct-AWQ"#
    # vlm_name = "Qwen/Qwen2-VL-7B-Instruct
    # Qwen/Qwen2.5-1.5B-Instruc
    if("combination" in model_name or "Qwen2.5-7B-Instruct" in model_name or "Qwen2.5-1.5B-Instruct" in model_name):
        
        if(tokenizer_llm is None):
            tokenizer_llm = processor
        if(mode == "val"):
            tokenizer_llm.padding_side="left"
        if(mode != "train" and mode != "val_loss"):
            if(llm_prompt_for_vision):
                messages_llm = [
                    [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": prompt_llm}
                    ] for prompt_llm in examples["prompt_vlm"]
                ]
            else:
                messages_llm = [
                    [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": prompt_llm}
                    ] for prompt_llm in examples["prompt_llm"]
                ]
            text_llm = [tokenizer_llm.apply_chat_template(
                prompt_llm,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            ) for prompt_llm in messages_llm]
        else:
            messages_llm = [
                [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt_llm},
                    {"role": "assistant","content": labels}
                ] for prompt_llm,labels in zip(examples["prompt_llm"],examples["labels"])
            ]
                
            text_llm = [tokenizer_llm.apply_chat_template(
                prompt_llm,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            ) for prompt_llm in messages_llm]
        llm_data = tokenizer_llm(text_llm, return_tensors="pt",padding=True)

    if("combination" in model_name):
        
        inputs = {}
        inputs["model_inputs_vlm"] = vlm_data
        inputs["model_inputs_llm"] = {
            "input_ids":llm_data["input_ids"],
            "attention_mask":llm_data["attention_mask"],
        }
        inputs["video_path"] = video_paths
        inputs["prompt_llm"] = prompt_llm
        inputs["prompt_vlm"] = prompt_vlm
    elif("VL" in model_name or "Video" in model_name):
        inputs = {}
        inputs["model_inputs_vlm"] = vlm_data
        inputs["video_path"] = video_paths
        if(llm_prompt_for_vision):
            inputs["prompt_vlm"] = prompt_llm
        else:
            inputs["prompt_vlm"] = prompt_vlm
        
    else:
        inputs = {}
        if(llm_prompt_for_vision):
            inputs["prompt_llm"] = prompt_vlm
        inputs["model_inputs_llm"] = llm_data
        # inputs["video_path"] = video_paths

    if(mode == "train" or mode == "val_loss"):
        if("model_inputs_llm" in inputs):
            input_ids_lists = inputs["model_inputs_llm"]['input_ids'].tolist()
            # print("LLM in")
        else:
            input_ids_lists = inputs["model_inputs_vlm"]['input_ids'].tolist()
        # assert len(messages) == len(input_ids_lists)
        # print(input_ids_lists)
        # print("C"+5)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            labels_list.append(label_ids)
            # tokens = find_assistant_content_sublist_indexes(ids_list)
            # print(label_ids, tokens,len(inputs["model_inputs_llm"]['input_ids'][0]),
            #       len(inputs["model_inputs_vlm"]['input_ids'][0]),
            #       len(ids_list))
            # print(ids_list[tokens[0][0]:tokens[0][1]])
            # input_llm = tokenizer_llm.batch_decode(ids_list[tokens[0][0]:tokens[0][1]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            # print(input_llm)
        # print("C"+5)
        inputs['labels'] = torch.tensor(labels_list, dtype=torch.int64)
    else:
        inputs["labels"] = examples["labels"]
    
    inputs["scene_id"] = scene_ids
    inputs["task_id"] = task_ids
    # print(time.time()-curr)
    return inputs

    # input_ids = []
    # attention_mask = []
    # labels = []
    # pixel_values_videos = []
    # video_grid_thw = []
    # for sample in batch:
    #     input_ids.append(torch.tensor(sample["input_ids"]))
    #     attention_mask.append(torch.tensor(sample["attention_mask"]))
    #     labels.append(torch.tensor(sample["labels"]))
    #     pixel_values_videos.append(torch.tensor(sample["pixel_values_videos"]))
    #     video_grid_thw.append(torch.tensor(sample["video_grid_thw"]))

    # max_len = max(len(seq) for seq in input_ids)

    # Step 2: Pad sequences to the maximum length with padding token (assuming PAD token is 0)
    padded_sequences = []
    padded_attns = []
    for i, seq in enumerate(input_ids):
        padding_length = max_len - len(seq)
        if(padding_length == 0):
            padded_sequences.append(seq)
            padded_attns.append(attention_mask[i])
            continue
        padded_sequence = torch.cat((torch.tensor([processor.tokenizer.pad_token_id] * padding_length),seq),dim=0)  # Padding with token '0'
        padded_sequences.append(padded_sequence)

        padded_attn = torch.cat((torch.tensor([0] * padding_length),attention_mask[i]),dim=0)
        padded_attns.append(padded_attn)

    batch_dict = {
        "input_ids" : torch.stack(padded_sequences),
        "attention_mask" : torch.stack(padded_attns),
        "labels" : torch.stack(labels),
        "pixel_values_videos" : torch.stack(pixel_values_videos),
        "video_grid_thw" : torch.stack(video_grid_thw),
    }
    return batch_dict
