
from transformers import AutoModel, AutoTokenizer
from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
import torch
import av
import numpy as np
from transformers import BitsAndBytesConfig
import os
import json
from decord import VideoReader, cpu
from easydict import EasyDict
import torch.nn.functional as F

import utils
# from llava.mm_utils import (get_model_name_from_path, process_images,
#                             tokenizer_image_token)
from utils import read_video_pyav_aurora

def Qwen2_VL_7B_Instruct(model,examples,mode,processor,tokenizer_llm,llm_prompt_for_vision):


    if(mode == "train"  or mode == "val_loss"):
        if(llm_prompt_for_vision):
            messages = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": 400 * 711,
                                "fps": 1.0,
                            },
                            {"type": "text", "text": prompt_vlm},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": label}
                        ]
                    }
                ] for video_path, prompt_vlm, label in zip(examples["video_path"],examples["prompt_llm"],examples["labels"])
            ]
        else:
            messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 400 * 711,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": prompt_vlm},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label}
                    ]
                }
            ] for video_path, prompt_vlm, label in zip(examples["video_path"],examples["prompt_vlm"],examples["labels"])
        ]
        
        add_generation_prompt = False
        if(mode == "val_loss"):
            processor.padding_side="left"
            if(tokenizer_llm is not None):
                tokenizer_llm.padding_side="left"
        
    else:
        if(llm_prompt_for_vision):
            messages = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": 400 * 711,
                                "fps": 1.0,
                            },
                            {"type": "text", "text": prompt_vlm},
                        ],
                    }
                ] for video_path, prompt_vlm in zip(examples["video_path"],examples["prompt_llm"])
            ]
        else:
            messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 400 * 711,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": prompt_vlm},
                    ],
                }
            ] for video_path, prompt_vlm in zip(examples["video_path"],examples["prompt_vlm"])
        ]
        
        processor.padding_side="left"
        if(tokenizer_llm is not None):
            tokenizer_llm.padding_side="left"
        add_generation_prompt = True


    # curr = time.time()
    # print(curr-start)
    
    texts = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=add_generation_prompt) for example in messages
    ]  # Prepare texts for processing

    # print(time.time()-curr)
    
    batch_size = len(texts)
    # for scene in scene_ids:
    # video_path = "../AutoSeg-SAM2/processed_lingoqa/{}/video.mp4".format(scene_ids[0])
    # scene_path = "../lingo/LingoQA/evaluation/images/val/" + scene_ids[0] + "/"
    # image_names = os.listdir(scene_path)
    # image_paths = [scene_path + image_name for image_name in image_names]
    # image_paths = natsort.natsorted(image_paths)
    # video = utils.create_video_from_frames(image_paths, video_path, fps=1)
    # messages = [
    # {
    #     "role": "user",
    #     "content": [
    #         {
    #             "type": "video",
    #             "video": video_path,
    #             "max_pixels": 400 * 711,
    #             "fps": 1.0,
    #         }
    #         ],
    #     }
    # ]
    # image_inputs = process_vision_info(messages)[1]
    # torch.save(image_inputs[0].to(torch.uint8), "../AutoSeg-SAM2/processed_lingoqa/{}/video.pth".format(scene_ids[0]))
    # os.remove("../AutoSeg-SAM2/processed_lingoqa/{}/video.mp4".format(scene_ids[0]))

    # print(examples["video_path"])
    if("/new/" in examples["video_path"][0]):
        replaced_str = "../new/AutoSeg-SAM2/processed_lingoqa"
    else:
        replaced_str = "../AutoSeg-SAM2/processed_lingoqa"
    image_inputs = [torch.load(examples["video_path"][i].replace(replaced_str,"../new/LLaVA-NeXT/processed_lingoqa").replace(".mp4",".pth"),weights_only=False).to(torch.float32) for i in range(batch_size)]
    new_height = int(image_inputs[0].size(2) / 1.5)
    new_width = int(image_inputs[0].size(3) / 1.5)
    # print(image_inputs[0].shape)
    # for i in range(len(image_inputs)):
    #     # image_inputs[i] = F.avg_pool2d(image_inputs[i], kernel_size=2, stride=2)


    #     # Resize the tensor
    #     image_inputs[i] = F.interpolate(image_inputs[i], size=(new_height, new_width), mode='bilinear', align_corners=False)
    #     image_inputs[i] = torch.round(image_inputs[i])

        # print(image_inputs[i])
    # print(time.time()-curr)
    # print(image_inputs[0].shape)
    # Tokenize the texts and process the images
    inputs = processor(
        text=texts, videos=image_inputs, padding=True, return_tensors="pt"#
    )  # Encode texts and images into tensors
    
    # print(time.time()-curr)
    inputs["pixel_values_videos"] = [inputs["pixel_values_videos"][i * int(len(inputs["pixel_values_videos"]) / batch_size):(i + 1) * int(len(inputs["pixel_values_videos"]) / batch_size)] for i in range(batch_size)]
    inputs["pixel_values_videos"] = torch.stack(inputs["pixel_values_videos"])

    # print(inputs["input_ids"])  
    # print(inputs["attention_mask"])    

    # vlm_data = {
    #     "input_ids":inputs.pop("input_ids"),
    #     "attention_mask":inputs.pop("attention_mask"),
    #     "pixel_values_videos":inputs.pop("pixel_values_videos"),
    #     "video_grid_thw":inputs.pop("video_grid_thw"),
    # }
    return inputs, add_generation_prompt


def LLaVA_Video_7B_Qwen2(model,examples,mode,processor,tokenizer_llm,llm_prompt_for_vision,max_frames_num=5):

    processor.padding_side="left"
    tokenizer_llm.padding_side="left"
    add_generation_prompt = True


    videos = [utils.load_video(video_path.replace("/video","").replace("../AutoSeg-SAM2/processed_lingoqa/","../new/LLaVA-NeXT/processed_lingoqa"), max_frames_num, 1, force_sample=True) for video_path in examples["video_path"]]
    # image_inputs = [torch.load(examples["video_path"][i].replace("../AutoSeg-SAM2/processed_lingoqa","../new/LLaVA-NeXT/processed_lingoqa").replace(".mp4",".pth"),weights_only=False).to(torch.float32) for i in range(batch_size)]
    
    videos_preprocessed = [processor.preprocess(video[0], return_tensors="pt")["pixel_values"].cuda().half() for video in videos]
    # videos_preprocessed = [videos_preprocessed]
    
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instrucitons = [f"The video lasts for {video[2]:.2f} seconds, and {video_preprocessed[0].shape[0]} frames are uniformly sampled from it. \
                        These frames are located at {video[1]}.Please answer the following questions related to this video." for video,video_preprocessed in zip(videos,videos_preprocessed)]
    
    if(llm_prompt_for_vision):
        questions = [DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n" + prompt_vlm_ for prompt_vlm_,time_instruciton in zip(examples["prompt_llm"],time_instrucitons)]
    else:
        questions = [DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n" + prompt_vlm_ for prompt_vlm_,time_instruciton in zip(examples["prompt_vlm"],time_instrucitons)]

    vlm_input_ids = []
    attention_masks = []
    max_length = -1
    for question in questions:
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        vlm_input_id = tokenizer_image_token(prompt_question, tokenizer_llm, IMAGE_TOKEN_INDEX, return_tensors="pt").to("cuda:0")
        vlm_input_ids.append(vlm_input_id)
        attention_masks.append(torch.ones(vlm_input_id.shape[0]).to(torch.bfloat16))
        
        if(vlm_input_ids[-1].shape[0] > max_length):
            max_length = vlm_input_ids[-1].shape[0]
    for i in range(len(vlm_input_ids)):
        tensor = vlm_input_ids[i]
        # Calculate the padding amount on the left side
        pad_left = max_length - tensor.size(0)
        
        if pad_left > 0:
            # Pad tensor on the left side (before the existing data)
            vlm_input_ids[i] = F.pad(tensor, (pad_left,0), "constant", tokenizer_llm.pad_token_id)
            attention_masks[i] = F.pad(attention_masks[i], (pad_left,0), "constant", 0)
            
    vlm_input_ids = torch.stack(vlm_input_ids)#.to(torch.bfloat16)
    attention_masks = torch.stack(attention_masks)
    videos_preprocessed = torch.stack(videos_preprocessed).to(torch.bfloat16)
    vlm_data = {"input_ids":vlm_input_ids,
                # "do_sample":False,
                "attention_mask":attention_masks,
        #    "images":[vid.to(torch.bfloat16) for vid in video],
        "images":videos_preprocessed,#[vid for vid in video],
        "modalities":["video"]*vlm_input_ids.shape[0]}
        
    return vlm_data, True

    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    video_path = "XXXX"
    # max_frames_num = 64
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    print(text_outputs)

def longVA(frames,prompt,args):
    model, processor, tokenizer = args
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{}}<|im_end|>\n<|im_start|>assistant\n".format(prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    gen_kwargs = {"do_sample": False, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}

    video_tensor = processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def method_1(video,frame_time,video_time,prompt,args):

    model, processor, tokenizer = args
    

    video = processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
    video = [video]
    
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
    # print(input_ids,torch.min(input_ids),torch.argmin(input_ids),torch.max(input_ids),torch.argmax(input_ids))
    # print("C"+5)
    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs

def method_2(video,prompt,args):

    model, processor = args
    
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


    inputs = processor(text=[prompt], videos=[video], padding=True, return_tensors="pt").to(model.device, torch.float16)
    generate_kwargs = {"max_new_tokens": 1000, "do_sample": False, "top_p": 0.9}

    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)

    return generated_text

def method_3(video,prompt,args):

    model, processor = args

    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs_video = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=1000, do_sample=False)
    
    return processor.decode(output[0][2:], skip_special_tokens=True) 

class composer():
    models = {"composer-hd":"internlm/internlm-xcomposer2-4khd-7b",
              "composer-hd-336":"internlm/internlm-xcomposer2-4khd-7b",
              "composer-vl":"internlm/internlm-xcomposer2-vl-7b",
              "composer-2.5":'internlm/internlm-xcomposer2d5-7b'}
    def __init__(self,model_name):
        self.model_name = model_name
        self.model_path = self.models[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)#.cuda()
        # Set `torch_dtype=torch.floatb16` to load model in bfloat16, otherwise it will be loaded as float32 and might cause OOM Error.
        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        self.model.tokenizer = self.tokenizer
        self.model = self.model.eval()
    
    def run(self,image_path, prompt):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                #first:55
                #second:1
                if(self.model_path == "internlm/internlm-xcomposer2d5-7b"):
                    # prompt = "Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; Image4 <ImageHere>; Image5 <ImageHere>;  for each of the images asnwer the following task: " + prompt
                    

                    response, his = self.model.chat(self.tokenizer, query=prompt, image=[image_path], history=[], do_sample=False, num_beams=5,
                        use_cache=True,
                        # output_scores=True,
                        # return_dict_in_generate=True
                        )
                elif(self.model_path == "internlm/internlm-xcomposer2-4khd-7b"):
                    if(self.model_name == "composer-hd"):
                        response, his = self.model.chat(self.tokenizer, query="<ImageHere>" + prompt, image=image_path, hd_num=55, history=[], do_sample=False, num_beams=1,
                            use_cache=True,
                            # output_scores=True,
                            # return_dict_in_generate=True
                            )
                        print(his)
                else:
                    response, his = self.model.chat(self.tokenizer, query="<ImageHere>" + prompt, image=image_path, history=[], do_sample=False, num_beams=1,
                        use_cache=True,
                        # output_scores=True,
                        # return_dict_in_generate=True
                        )
            return response
        
def method_4(model_name):
    model = composer(model_name)
    prompt = 'Here are some frames of a video. In each frame at the top left corner, the timestamp for the frame is provided. Describe the content in the video, referencing the timestamp per frame.'
    output = model.run("/home/ge32buc/new/LLaVA-NeXT/output_video_with_frame_timestamp.mp4",prompt)

def qwen(video_path, prompt, args):
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    model, processor = args

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 400 * 711,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The image is dark."}],
        },
    ]

    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages
    ]  # Prepare texts for processing
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"

    )

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = inputs["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    inputs["labels"] = labels  # Add labels to the batch

    # print(messages)

    # Preparation for inference
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    inputs = inputs.to("cuda")
  
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1280,do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    # print("c"+5)
    return output_text

def video_chat(video, prompt, resolution, num_frame, args):

    model = args


    new_pos_emb = utils.get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    model.vision_encoder.encoder.pos_embed = new_pos_emb

        
    # The model expects inputs of shape: T x C x H x W
    T_, C, H, W = video.shape
    video = video.reshape(1, T_, C, H, W).to("cuda:0")

    img_list = []
    with torch.no_grad():
        image_emb, _, _ = model.encode_img(video, "Watch the video and answer the question.")
    #     image_emb, _, _ = model.encode_img(video, "")

    img_list.append(image_emb[0])

    chat = EasyDict({
        "system": "",
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
    # chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg} [/INST]"])
    utils.ask(prompt, chat)

    llm_message = utils.answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=512, print_res=True)[0]
    # print(llm_message)
    return llm_message

def THUDM(video, prompt, strategy, args):

    model, tokenizer = args

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.float16)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": 0.1,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def aurora(video_path, prompt, args):
    # Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos, up to 32 frames)
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav_aurora(container, indices)

    model, processor = args
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, # we still use image type for video input
                {"type": "text", "text": prompt},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda:0", torch.float16)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=1024, token_kept_ratio=0.2)
    return processor.decode(output[0], skip_special_tokens=True)


def llama(prompt,args):

    pipeline = args




    messages = [
        # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=2560,
    )[0]["generated_text"][-1]
    
    return outputs

def preprocess(model_name):
    print(model_name)
    if("OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf" in model_name):
        from transformers import AutoModel
        model=AutoModel.from_pretrained("OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",trust_remote_code=True).to("cuda:0")
        return model
    if("Athene" in model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",quantization_config=quantization_config).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer


        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        return pipeline
    if("meta-llama/Meta" in model_name):
        import transformers
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            # quantization_config=quantization_config
        )
        return pipeline
    if("share" in model_name):
        
        model_name_modified = os.path.expanduser(model_name)
        model_type = get_model_name_from_path(model_name_modified)
        
        tokenizer, model, processor, context_len = load_pretrained_model(
            model_name, None, model_type, device_map='auto')
        model = model.eval()
        return model, processor, tokenizer
    if("Qwen/" in model_name and "VL" in model_name or "/mnt/mergekit/merged" in model_name):
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

        # default: Load the model on the available device(s)
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8", torch_dtype="auto", device_map="balanced"
        # )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="balanced",
        ).eval()

        if(model_name == "/mnt/mergekit/merged"):
            model_name = "Qwen/Qwen2-VL-7B-Instruct"
        # default processer
        processor = AutoProcessor.from_pretrained(model_name)
        return model,processor

    elif("llava-hf" in model_name):
        if("one" in model_name):
            from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor, LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
            if("72b" in model_name):
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )

                model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name, device_map='auto', quantization_config=quantization_config)
            else:
                model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name, device_map='auto')

            processor = LlavaOnevisionProcessor.from_pretrained(model_name)
            processor.tokenizer.padding_side = "left"
            
            # model.to(0)
            model.eval()
            return model, processor
        elif("NeXT" in model_name):
            from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
        
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
                device_map = "balanced"
           )#.to(0)
           
            # device = "cuda"

            processor = LlavaNextVideoProcessor.from_pretrained(model_name)
            model.eval()
            return model, processor
            
    elif("lmms-lab" in model_name and "onevision" in model_name):
        
        model_type = "llava_qwen"
        device_map = "auto"
        #
        tokenizer, model, processor, max_length = load_pretrained_model(model_name, None, model_type, torch_dtype="bfloat16",  device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    
        
        model.eval()
        return model, processor, tokenizer
    elif("lmms-lab" in model_name):
        model_type = "llava_qwen"
        # from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
        if("72B" in model_name):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            tokenizer, model, processor, max_length = load_pretrained_model(model_name, None, model_type, torch_dtype="bfloat16", device_map="auto",quantization_config=quantization_config)  # Add any other thing you want to pass in llava_model_args

        else:
            overwrite_config = {}
            if("lmms-lab/llava-next-interleave-qwen-7b" in model_name):
                overwrite_config = {"attention_bias": False,"mm_spatial_pool_mode": "average"}

            tokenizer, model, processor, max_length = load_pretrained_model(model_name, None, model_type, torch_dtype="bfloat16", device_map="balanced",
                                                                                overwrite_config=overwrite_config)  
            model.vlm_processor = processor
            model.tokenizer_llm = tokenizer
        
        model.eval()
        # print("C"+5)


        # processor = LlavaNextVideoProcessor.from_pretrained(model_name)
        return model, processor, tokenizer
    elif("THUDM" in model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                
                
        ).to("cuda").eval()
        return model, tokenizer
    elif("AuroraCap" in model_name):
        from transformers import AuroraForConditionalGeneration, AuroraProcessor
        
        processor = AuroraProcessor.from_pretrained(model_name)
        model = AuroraForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        model.to("cuda:0")

        return model, processor
    
    elif("Qwen/" in model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,            
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="balanced",
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    elif(model_name == "nvidia/Llama-3.1-Nemotron-70B-Instruct"):
        from transformers import AutoModel

        model = AutoModel.from_pretrained("/home/ge32buc/new/Llama-3.1-Nemotron-70B-Instruct")
        return model
    elif("Nemo" in model_name):
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",quantization_config=quantization_config).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

#"lmms-lab/LLaVA-Video-7B-Qwen2","lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only",
# models = ["lmms-lab/llava-onevision-qwen2-7b-ov-chat"]

def qwen_text(prompt,args):
    model, tokenizer = args
    
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)    
    
    generated_ids = model.generate(
        **model_inputs,
        do_sample=False,
        max_new_tokens=12000
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def nemo(prompt,args):
    model, tokenizer = args

    messages = [{"role": "user", "content": prompt}]

    tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=4096, pad_token_id = tokenizer.eos_token_id)
    generated_tokens =response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return generated_text
    
# nemo()


def share(vid,prompt,args):
    model, processor, tokenizer = args
    conv_mode = "llava_llama_3"
    img_grid = vid
    
    pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames."

    conv = conv_templates[conv_mode].copy()
    if pre_query_prompt is not None:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + pre_query_prompt + prompt
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if not isinstance(img_grid, (list, tuple)):
        img_grid = [img_grid]
    image_size = img_grid[0].size
    image_tensor = process_images(img_grid, processor, model.config)[0]
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(
        device=model.device, non_blocking=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(
                dtype=torch.float16, device=model.device, non_blocking=True),
            image_sizes=[image_size],
            do_sample=False,
            temperature=1,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=500,
            pad_token_id=pad_token_id,
            use_cache=True)
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        return outputs

def aria(prompt, args):
    model, tokenizer = args

    messages = [{"role": "user", "content": prompt}]

    tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=4096, pad_token_id = tokenizer.eos_token_id)
    generated_tokens =response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return generated_text

    outputs = pipeline_(
        messages,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][-1]
