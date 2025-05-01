# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import os
import sys
import pickle
import json
# os.chdir("../ShareGPT4Video")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
# if("new" not in os.getcwd()):
#     os.chdir("LLaVA-NeXT")
from tqdm import tqdm
from load_nuscenes import NuScenesData
from PIL import Image
import requests
import torch
import sys
import warnings
import numpy as np
from transformers import pipeline, AutoProcessor
import requests
warnings.filterwarnings("ignore")
import datetime
import prepare_models
import utils
import pandas as pd
import cv2
import models.internvideo.utils as internvideouitls
import debugpy
from evaluate import evaluation

# Start debugpy listener
debugpy.listen(('0.0.0.0', 5678))  # Expose the debugger on port 5678
print("Waiting for debugger to attach...")
debugpy.wait_for_client() 

# # import some common detectron2 utilities

# sys.path.append("Mask2Former/")
# # from Mask2Former import mask2former_video

# from Mask2Former import mask2former#, mask2former_video
# from Mask2Former.mask2former_video.video_maskformer_model import VideoMaskFormer
# print(MetadataCatalog.list())
# print("C"+5)

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from PIL import Image
# import open_clip

# from sklearn.metrics.pairwise import cosine_similarity


# model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
# model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
# model.to(device)

def extract_clip_features(image):
    if(isinstance(image,str)):
        image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    else:
        image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

# Function to load saved features
def load_features(feature_file):
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    return features

# Function to compare the new image with saved features
def compare_with_saved_features(new_image, feature_file):
    # Extract features of the new image
    new_image_features = extract_clip_features(new_image)
    
    # Load saved features
    saved_features = load_features(feature_file)
    
    # Compute cosine similarity between the new image features and all saved image features
    similarities = {}
    for image_path, saved_features_vector in saved_features.items():
        similarity = cosine_similarity(new_image_features, saved_features_vector)
        similarities[image_path] = similarity[0][0]  # Extract scalar similarity
    
    # Sort similarities (highest to lowest)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_similarities



def scene_description_from_frames(data,scene_token):
    questions = []
    for key_frame in data[scene_token]["key_frames"]:
        frame_data = data[scene_token]["key_frames"][key_frame]
        qa = frame_data["QA"]
        tasks = ["perception","prediction","planning","behaviour"]
        for task in tasks:
            if(task not in qa):
                continue
            qa_pairs = qa[task]
            for qa_pair in qa_pairs:
                q = qa_pair["Q"]
                a = qa_pair["A"]
                if("<" in q or "<" in a):
                    continue
                if(q not in questions):
                    questions.append(q)
    return questions

def get_distance_in_meters(output_depth, pixel_coordinates, estimated_distance):
    # pixel_coordinates = [[950,700],[733,1358]]
    relative_inverse_depth_1, relative_inverse_depth_2 = output_depth[pixel_coordinates[0]], output_depth[pixel_coordinates[1]]
    inverse_depth_1,inverse_depth_2 = 1/estimated_distance[0],1/estimated_distance[1]
    
    shift = (inverse_depth_1 - (relative_inverse_depth_1/relative_inverse_depth_2) * inverse_depth_2)/(1-(relative_inverse_depth_1/relative_inverse_depth_2))
    scale = (inverse_depth_1 - shift) / relative_inverse_depth_1
    depth = 1/(scale * output_depth + shift)
    return depth


def load_mask2former():
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    # from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config
    from Mask2Former.mask2former import add_maskformer2_config  
    import torch
    import torchvision

    cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_panoptic_val")

  # This should be your custom Mask2Former import
    

    # Step 1: Register the custom Mask2Former architecture (VideoMaskFormer)
    # You need to register your custom model (this should point to your Mask2Former class)
    # Register the architecture in the META_ARCH registry
    # META_ARCH_REGISTRY.register(VideoMaskFormer)
    # import Mask2Former project
  

    # processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-instance")
    # model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-instance")
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # traffic sign and traffic light per frame
    cfg.merge_from_file("Mask2Former/configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_064788.pkl'

    # videobase
    # cfg.merge_from_file("Mask2Former/configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml")
    # cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_swin_large_IN21k_384_bs16_8ep/model_final_4da256.pkl"
    # cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    # cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    predictor = DefaultPredictor(cfg)
    return predictor, cityscapes_metadata
    return processor, model



import torch
import torchvision
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


def get_sign_category():
    tasks = pd.read_parquet(r"datasets/lingoqa/val.parquet").to_numpy()
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    
    for task in tasks:
        id_, scene, images, question, answer = task
        distance_maps = []

        images = ["datasets/lingoqa/" + image for image in images]
        # video = utils.create_video_from_frames(images,scene + ".mp4",fps=0.5)
        # import torch
        object_list = None
        with open("../AutoSeg-SAM2/processed_lingoqa/{}/objects.pkl".format(scene), 'rb') as file:
            object_list = pickle.load(file)
        prompt = ""
        image_0 = Image.open(images[0])
        for i, objects_per_frame in enumerate(object_list):
            image = Image.open(images[i])
            prompt += "frame nr: {}\n".format(i)
            for object_ in objects_per_frame:
                bbox,obj_id,category_id,depth = object_
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                # mean_y = (bbox[3] - bbox[1]) + bbox[1]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                # y = ((mean_y - image_0.height/2) * 1/depth) / 1030
                prompt += "a {}, with id: {}, horizontal position: {}, distance: {}\n".format(categories[category_id],obj_id,x,1/depth)
                if(category_id == 7):
                    cropped_image = image.crop(bbox)
                    similarities = compare_with_saved_features(cropped_image,"sign_features_uk.pkl")


def prepare_lingoqa_data():
    # dataset_path = "datasets/lingoqa/"
    dataset_path = "datasets/lingoqa/"
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    


    # Load a .parquet file
    tasks = pd.read_parquet(r"datasets/lingoqa/val.parquet")
    # tasks = pd.read_parquet(r"../lingo/LingoQA/action/train.parquet")
    
    # print(len(tasks))
    # print(tasks.head())
    # print("C"+5)
    tasks = tasks.to_numpy()
    # predictor, cityscapes_metadata = load_mask2former()
    moveable_objects_ids = [11,12,13,14,15,16,17,18]
    signs_lights = []

    signs_lights_ids = [6,7]
    text_prompt_relative = """The following text describes the content of a traffic scene. You are given a sequence of frames.
    The scene is observed from the perspective of the ego-vehicle. For each object, the relative horizontal position and distance to the ego-vehicle
    are given. If the same object is observed in multiple frames that is indicated by it having the same id, -1 indicates that this object was not tracked across frames.
    Use this information to reason across the entire scene not per frame. Do not repeat the numbers for position and rotation. 
    Do not give additional information. Only answer the question directly and concisely.\n"""
    task_data = []

    for task in tasks:

        id_, scene, images, question, answer = task
        # if(question != "What are you currently doing and why?" or scene != "a9f0e311b0c6f46a9cc7cb923234e60a"):
        #     continue

        distance_maps = []

        images = [dataset_path + image for image in images]
        if(os.path.exists("videos/" + scene + ".mp4") == False):
            video = utils.create_video_from_frames(images,"videos/" + scene + ".mp4",fps=1)
        # import torch
        object_list = None
        with open("../new/AutoSeg-SAM2/processed_lingoqa/{}/objects.pkl".format(scene), 'rb') as file:
            object_list = pickle.load(file)
        try:
            with open("../new/AutoSeg-SAM2/processed_lingoqa/{}/objects_traffic_signs.pkl".format(scene), 'rb') as file:
                sign_list = pickle.load(file)
        except:
            sign_list = []
        try:
            with open("../new/AutoSeg-SAM2/processed_lingoqa/{}/objects_traffic_lights.pkl".format(scene), 'rb') as file:
                light_list = pickle.load(file)
        except:
            light_list = []
        
        total_signs = [n for fr_lis in sign_list for n in fr_lis]
        total_lights = [n for fr_lis in light_list for n in sign_list]
        
        prompt = ""
        image_0 = Image.open(images[0])
        for i, objects_per_frame in enumerate(object_list):
            prompt += "frame nr: {}\n".format(i)
            for object_ in objects_per_frame:
                bbox,obj_id,category_id,depth = object_
                if(category_id > 5):
                    continue
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                prompt += "{}, id: {}, horizontal position: {}, distance: {}\n".format(categories[category_id],obj_id,x,1/depth)

            for object_ in sign_list[i]:
                bbox,obj_id,category_id,depth,header,descr = object_
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                prompt += "traffic sign, description :{}, horizontal position: {}, distance: {}\n".format(header + " " + descr,x,1/depth)

            for object_ in light_list[i]:
                
                bbox,state,category_id,depth = object_
                if(category_id != 6):
                    continue
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                prompt += "traffic light, state {}, horizontal position: {}, distance: {}\n".format(state,x,1/depth)
        # if(len(total_lights) != 0 and len(total_signs) != 0):
        #     print(prompt)
        #     print(sign_list,light_list)
        #     print("C"+5)
        task_data.append([id_, scene, images, question, answer, prompt.replace("\u00a0","")])

    return text_prompt_relative, task_data

        
def run(model_components, output_file_json, config, dataset_name,mode="quick",trans_dict=None):
    if(dataset_name == "nuscenes"):
        dataset_type = 'v1.0-trainval'
        dataset = NuScenesData(dataset_type)#mini,trainval
        scenes = dataset.nusc.scene
        with open("gt/v1_1_train_nus.json", 'r') as file:
            data = json.load(file)
        gt = data[scene_token]["scene_description"]
        # questions = scene_description_from_frames(data,scene_token)

        scene_info_prompt = utils.construct_dynamic_nuscenes_prompt(scene_data)
        
    elif(dataset_name == "lingoqa"):
        text_prompt, task_data = prepare_lingoqa_data()

        with open(output_file_json, mode='w', encoding='utf-8') as f:
            json.dump([], f)
        # print(task_data)
        for i, task in enumerate(tqdm(task_data[:200])):
            if(i % 2 == 0):
                continue
            if(mode == "quick"):
                
                if(i == 51):
                    model_components.to("cpu")

                    trans_dict,similarity, bleu_mean, meteor_mean, rouge_mean, cos_mean, lingo_mean,accuracy,nr_samples = evaluation.analyse_qa_similarity(trans_dict,model,None,"",output_file_json,False)
                    print(lingo_mean)
                    if(lingo_mean < 0.1):
                        return
                    model_components.to("cuda")
                if(i == 101):
                    model_components.to("cpu")
                    trans_dict,similarity, bleu_mean, meteor_mean, rouge_mean, cos_mean, lingo_mean,accuracy,nr_samples = evaluation.analyse_qa_similarity(trans_dict,model,None,"",output_file_json,False)
                    print(lingo_mean)
                    if(lingo_mean < 0.15):
                        return
                    model_components.to("cuda")
                if(i == 201):
                    model_components.to("cpu")
                    trans_dict,similarity, bleu_mean, meteor_mean, rouge_mean, cos_mean, lingo_mean,accuracy,nr_samples = evaluation.analyse_qa_similarity(trans_dict,model,None,"",output_file_json,False)
                    print(lingo_mean)
                    if(lingo_mean < 0.2):
                        return
                    model_components.to("cuda")
                if(i == 501):
                    model_components.to("cpu")
                    trans_dict,similarity, bleu_mean, meteor_mean, rouge_mean, cos_mean, lingo_mean,accuracy,nr_samples = evaluation.analyse_qa_similarity(trans_dict,model,None,"",output_file_json,False)
                    print(lingo_mean)
                    if(lingo_mean < 0.3):
                        return
                    model_components.to("cuda")
            current_time = datetime.datetime.now()
            # print(current_time)

            # Define the time limit (6 AM)
            # time_limit = current_time.replace(hour=6, minute=0, second=0, microsecond=0)
            # # Check if current time is later than 6 AM
            # if current_time > time_limit:
            #     print("Current time is later than 6 AM. Exiting the program.")
            #     sys.exit()  # This will stop the program
            id_, scene, images, question, answer, scene_info_prompt = task
            

            # print(question,answer)
            # print(text_prompt)
            # print(scene_info_prompt)
            video_path = "videos/" + scene + ".mp4"
            video = utils.create_video_from_frames(images, video_path, fps=1)
            if("combination" in model_name):
                merge_feature_layers,feature_weights = config[3:5]
                llm_prompt = text_prompt + question + scene_info_prompt#video_prompt.replace("video","scene")
                vlm_prompt = question#.replace("?"," in this video?") 
                # merge_feature_layers = [-1]
                # feature_weights = [0.5,0.5]
                result = model_components.forward(video_path,llm_prompt,vlm_prompt,merge_feature_layers,feature_weights)
                # print(result)
                data = {
                    "model_name": "combination_" + vlm_name + "_" + llm_name,
                    "question": question,
                    "scene_token": scene,
                    "task_token": id_,
                    "llm_prompt": llm_prompt,
                    "vlm_prompt": vlm_prompt,
                    "result": result,
                    "merge_feature_layers":merge_feature_layers,
                    "feature_weights":feature_weights,
                    "merge_head_weights":model_components.merge_head_weights,
                    "merge_layer_weights_layers":model_components.merge_layer_weights_layers,
                    "merge_layer_weights_weights":model_components.merge_layer_weights_weights,
                    "mode":mode,
                    "get_all_vlm_features_first":model_components.get_all_vlm_features_first
                }
                # print(data)
                # print(result)
                with open(output_file_json, 'w') as json_file:
                    datas.append(data)
                    json.dump(datas, json_file, indent=4)
                continue
            elif(model_name in video_names):
                prompt = text_prompt + question + scene_info_prompt
                prompt = question
                

                # result = model_components.forward(text_prompt + construct_dynamic_prompt(scene_data) + video_prompt_,video_prompt)
                if("InternVideo" in model_name):
                    video_tensor = internvideouitls.load_video(video_path, num_segments=8, return_msg=False, resolution=448, hd_num=12)
                    result = prepare_models.internvideo(video_tensor, prompt, model_components)
                if("Tarsier" in model_name):
                    result = prepare_models.tarsier(video_path,prompt,model_components)

                if("InternVL" in model_name):
                    pixel_values, num_patches_list = utils.load_video_intern(video_path, num_segments=8, max_num=12)
                    result = prepare_models.internvl2(pixel_values, num_patches_list,prompt,model_components)
                if("share" in model_name):
                    vid, msg = utils.load_video_share(video_path, num_segments=16, return_msg=True)
                    result = prepare_models.share(vid,prompt,model_components)
                if("longVA" in model_name):
                    frames = utils.load_video_longVQ(video_path,16)
                    result = prepare_models.longVA(frames,prompt,model_components)
                if("lmms-lab" in model_name):
                    video,frame_time,video_time = utils.load_video(video_path, 16, 1, force_sample=True)
                    result = prepare_models.method_1(video,frame_time,video_time,prompt,model_components)
                elif("llava-hf" in model_name):
                    video = utils.read_video_pyav(video_path)
                    if("onevision" in model_name):
                        result = prepare_models.method_3(video,prompt,model_components)
                    else:
                        result = prepare_models.method_2(video,prompt,model_components)

                elif("Qwen/" in model_name or "mergekit" in model_name):
                    result = prepare_models.qwen(video_path,prompt,model_components)
                elif("THUDM" in model_name):
                            
                    strategy = 'chat'
                    video_data = open(video_path, 'rb').read()
                    video = utils.load_video_THUDM(video_data, strategy=strategy)
                    
                    result = prepare_models.THUDM(video,prompt,model_components)
                elif("AURORA" in model_name):
                    result = prepare_models.aurora(video_path,prompt,model_components)
                # print("vlm")


            else:
                prompt = text_prompt + question + scene_info_prompt#video_prompt.replace("video","scene")
                # elif(i == 1):
                #     prompt = text_prompt + construct_dynamic_prompt(scene_data) + question#video_prompt.replace("video","scene")
                if("meta-llama/Meta" in model_name):
                    result = prepare_models.llama(prompt,model_components)
                elif("Athene" in model_name):
                    result = prepare_models.aria(prompt,model_components)
                elif("Qwen" in model_name):
                    result = prepare_models.qwen_text(prompt,model_components)
                elif("Nemo" in model_name):
                    result = prepare_models.nemo(prompt,model_components)
                # print("llm")
            # else:
            #     prompt = text_prompt + construct_dynamic_prompt(scene_data) + video_prompt
            #     result = qwen_text(prompt,model_components)
            # print(result)

            # print(result)
            data = {
                "model_name": model_name,
                "question": question,
                "scene_token": scene,
                "task_token": str(id_),
                "prompt": prompt,
                "result": result,
            }
            # os.remove(video_path)
            # print(data)
            with open(output_file_json, 'w') as json_file:
                datas.append(data)
                json.dump(datas, json_file, indent=4)
            # break


    # for i in tqdm(range(0,len(scenes))):
    #     if(dataset_name == "nuscenes"):
    #         scene_data, scene_token = dataset.get_nuscenes_data(i)
    #     elif(dataset_name == "lingoqa"):
    #         text_prompt_relative, task_data = 
    #     data = {}
        
    #     if(scene_token not in data):
    #         continue
    #     scene_description = ""


    #     questions = utils.get_questions()
        
    #     video = utils.create_video_from_frames(scene_data, scene_token + ".mp4", fps=4)
    #     # print("C"+5)
    #     video_path = scene_token + ".mp4"
    #     for question in questions:
            


            
            


models_70b = ["Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8","lmms-lab/LLaVA-Video-72B-Qwen2","Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8","nvidia/Llama-3.1-Nemotron-70B-Instruct-HF","lmms-lab/llava-onevision-qwen2-72b-ov-chat",
              "meta-llama/Meta-Llama-3.1-70B-Instruct"]

llavaonevision_models = ["llava-hf/llava-onevision-qwen2-7b-ov-chat-hf","lmms-lab/llava-onevision-qwen2-7b-ov-chat","lmms-lab/llava-onevision-qwen2-7b-si","lmms-lab/llava-onevision-qwen2-7b-ov"]
llavanext_models = ["llava-hf/LLaVA-NeXT-Video-34B-hf","lmms-lab/LLaVA-NeXT-Video-32B-Qwen","lmms-lab/llava-next-interleave-qwen-7b-dpo","lmms-lab/llava-next-interleave-qwen-7b"]
llava_video_models = ["lmms-lab/LongVA-7B-DPO","lmms-lab/LLaVA-Video-7B-Qwen2","lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only"]
aurora_models = ["wchai/AuroraCap-7B-image"]
cogvlm_models = ["THUDM/cogvlm2-llama3-caption"]
sharegpt_models = ["Lin-Chen/sharegpt4video-8b"]
qwen_models = ["Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2-VL-7B-Instruct-AWQ","Qwen/Qwen2.5-7B-Instruct"]
intern_models = ['OpenGVLab/InternVL2-8B',"OpenGVLab/InternVideo2_Chat_8B_InternLM2_5",'OpenGVLab/InternVideo2_chat_8B_HD']
tarsier_models = ["omni-research/Tarsier-7b"]

model_list = llavaonevision_models + llavanext_models + aurora_models+ cogvlm_models + sharegpt_models + qwen_models + llava_video_models + intern_models + tarsier_models
#"/mnt/mergekit/merged","Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2-VL-7B-Instruct-AWQ","Lin-Chen/sharegpt4video-8b",
#"llava-hf/LLaVA-NeXT-Video-34B-hf","lmms-lab/LLaVA-NeXT-Video-32B-Qwen",
video_names = []
#,"lmms-lab/LongVA-7B-DPO",
#,"Qwen/Qwen2-VL-2B-Instruct","Qwen/Qwen2-VL-2B-Instruct-AWQ"


###todo short prompt
# "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
# long prompt
# "Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2-VL-7B-Instruct-AWQ",
video_names = [
    "lmms-lab/LLaVA-Video-7B-Qwen2","lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only",
    "lmms-lab/llava-onevision-qwen2-7b-si","lmms-lab/llava-onevision-qwen2-7b-ov-chat",
    "lmms-lab/llava-onevision-qwen2-7b-ov","lmms-lab/llava-next-interleave-qwen-7b",
    "lmms-lab/llava-next-interleave-qwen-7b-dpo","llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
    "Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2-VL-7B-Instruct-AWQ",'OpenGVLab/InternVL2-8B',"omni-research/Tarsier-7b",
    "OpenGVLab/InternVideo2_Chat_8B_InternLM2_5",'OpenGVLab/InternVideo2_chat_8B_HD']

#,"lmms-lab/LLaVA-Video-72B-Qwen2",
#"Qwen/Qwen2-VL-2B-Instruct","Qwen/Qwen2-VL-2B-Instruct-AWQ",
models = ["combination"]

video_prompt = "Describe the video in detail. Focus on the ego-vehicle and its actions."

# video_prompt_ = "you are given a video. what is the status of the traffic light?"
# text_prompt = """The following text describes the content of a traffic scene. You are given a sequence of frames with the corresponding time in microseconds,          
#             the position and size of objects is given in meters in x,y,z coordinates and the rotation in roll, pitch, yaw. Use this information to reason across the entire scene not per frame, do not repeat the numbers for position and rotation."""
# models = ["combination"]

text_prompt = """The following text describes the content of a traffic scene. You are given a sequence of frames with the corresponding time in microseconds.
            The global position and size of objects is given in meters in x, y, z coordinates and their rotation in roll, pitch, yaw. 
            Use this information to reason across the entire scene not per frame. Do not repeat the numbers for position and rotation. 
            Do not give additional information. Only answer the question directly and concisely.\n"""
            # text_prompt = """The following text describes the content of a traffic scene. You are given a sequence of frames with the corresponding time in microseconds,
#             the position and size of objects is given in meters in x,y,z coordinates w.r.t a global coordinate system. Use this information to reason, do not repeat the numbers for the position.\n
#             """

text_prompt_relative = """The following text describes the content of a traffic scene. You are given a sequence of frames.
            The global position and size of objects is given in meters in x, y, z coordinates and their rotation in roll, pitch, yaw. 
            Use this information to reason across the entire scene not per frame. Do not repeat the numbers for position and rotation. 
            Do not give additional information. Only answer the question directly and concisely.\n"""

llm_name = "Qwen/Qwen2.5-7B-Instruct"#"Qwen/Qwen2-7B-Instruct"#"Qwen/Qwen2-7B-Instruct-AWQ"#
vlm_name = "Qwen/Qwen2-VL-7B-Instruct"#"lmms-lab/llava-onevision-qwen2-7b-ov"#

models = ["combination"]#,"Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2-VL-7B-Instruct"]

# models = ["combination"]
# merge_feature_layers = [-1]
# feature_weights = [0.5,0.5]
#["combination",True,[],[-1],[0.1,0.9]],
# models = [["combination",[],[-1],[0.1,0.9],[0.1,0.9]],["combination",[],[-1],[0.3,0.7],[0.3,0.7]],["combination",[],[-1],[0.7,0.3],[0.7,0.3]],["combination",[],[-1],[0.5,0.5],[0.5,0.5]],
#           ["combination",[],[-1],[0.9,0.1],[0.9,0.1]],["combination",[],[-1],[1.,1.],[1.,1.]],
#           ["combination",[],[-1],[0.1,0.9],None],["combination",[],[-1],[0.3,0.7],None],["combination",[],[-1],[0.7,0.3],None],["combination",[],[-1],[0.5,0.5],None],
#           ["combination",[],[-1],[0.9,0.1],None],["combination",[],[-1],[1.,1.],None]]
models = [["combination",[i for i in range(50)],[0.1,0.9],[-1],[0.1,0.9],[0.1,0.9]],["combination",[i for i in range(50)],[0.3,0.7],[-1],[0.3,0.7],[0.3,0.7]],["combination",[i for i in range(50)],[0.7,0.3],[-1],[0.7,0.3],[0.7,0.3]],["combination",[i for i in range(50)],[0.5,0.5],[-1],[0.5,0.5],[0.5,0.5]],
          ["combination",[i for i in range(50)],[0.9,0.1],[-1],[0.9,0.1],[0.9,0.1]],["combination",[i for i in range(50)],[1.,1.],[-1],[1.,1.],[1.,1.]],
          ["combination",[i for i in range(50)],[0.1,0.9],[-1],[0.1,0.9],None],["combination",[i for i in range(50)],[0.3,0.7],[-1],[0.3,0.7],None],["combination",[i for i in range(50)],[0.7,0.3],[-1],[0.7,0.3],None],["combination",[i for i in range(50)],[0.5,0.5],[-1],[0.5,0.5],None],
          ["combination",[i for i in range(50)],[0.9,0.1],[-1],[0.9,0.1],None],["combination",[i for i in range(50)],[1.0,1.0],[-1],[1.0,1.0],None]]
        #   ["combination",[],[-1],[0.1,0.9],None],["combination",[],[-1],[0.3,0.7],None],["combination",[],[-1],[0.7,0.3],None],["combination",[],[-1],[0.5,0.5],None],
        #   ["combination",[],[-1],[0.9,0.1],None],["combination",[],[-1],[1.,1.],None]]

models = ["Qwen/Qwen2.5-7B-Instruct"]
models = ["combination"]
# models = ["Qwen/Qwen2-VL-7B-Instruct"]
layers_weights_to_merge = [[]]
#
weights = [[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1],[1.,1.]]
weights = [[0.9,0.1]]

layers_features_to_merge = [[-1],[i for i in range(10,28)] + [-1],[i for i in range(20,28)] + [-1],[26,27,-1],[27,-1]]
layers_features_to_merge = [[25,26,27,-1]]
## 
configs = []
for model in models:
    for layer_weights in layers_weights_to_merge:
        for weight in weights:
            print(model,layer_weights,weight[0])
            for layer_features in layers_features_to_merge:
                configs.append([model,layer_weights,weight,layer_features,weight,weight])
print(len(configs))
# configs = [[name] for name in video_names + qwen_models]
#
configs = [[name] for name in ["lmms-lab/LLaVA-Video-7B-Qwen2","OpenGVLab/InternVideo2_Chat_8B_InternLM2_5",'OpenGVLab/InternVideo2_chat_8B_HD',"omni-research/Tarsier-7b",'OpenGVLab/InternVL2-8B',
                               "Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2-VL-7B-Instruct"]]
if __name__ == "__main__":
    trans_dict = {}
    for run_nr, config in enumerate(configs):
        # if(run_nr != 2 and run_nr != 3 and run_nr != 5 and run_nr != 9 and run_nr != 28):
        #     continue
        print(run_nr,config)
        # continue
        if(len(config) == 1):
            model_name = config[0]
        else:
            model_name, merge_layer_weights_layers,merge_layer_weights_weights, merge_feature_layers,feature_weights,merge_head_weights = config
        datas = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_file = f'out/results_{timestamp}.txt'
        output_file_json = f'out/results_{timestamp}.json'
        # merge_head_weights = True
        # merge_layer_weights = []

        model_components = None
        if("combination" in model_name):
            from combine_models_old import Combination
            # if(25 in merge_feature_layers and 10 not in merge_feature_layers):
            #     continue
            model_components = Combination(llm_name,vlm_name,merge_head_weights=merge_head_weights,merge_layer_weights_layers=[],
                                           merge_layer_weights_weights=merge_layer_weights_weights,get_all_vlm_features_first=True,
                                           merge_feature_layers=merge_feature_layers,feature_weights=feature_weights)
        elif(model_name in model_list):
            model_components = prepare_models.preprocess(model_name)


        print(model_name)

        prediction_dict = {}
        with torch.no_grad():
            # try:
            run(model_components, output_file_json, config, "lingoqa","quick",trans_dict)
            # except Exception as e:
            #     # Print the exception message
            #     print(f"An error occurred: {e}")
            #     time.sleep(5)
        # break
        # run("nuscenes")