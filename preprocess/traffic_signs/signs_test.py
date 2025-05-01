
import os

gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

import torch
from PIL import Image,ImageOps,ImageFilter
from transformers import CLIPVisionModel
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel


import pickle
import cv2

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

import debugpy
device = "cuda" if torch.cuda.is_available() else "cpu"
# debugpy.listen(('0.0.0.0', 5678))  # Expose the debugger on port 5678
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client() 

# Function to extract CLIP features from an image
def extract_clip_features(image,model,preprocess):
    # print(image)
    if(isinstance(image,str)):
        image = preprocess(text=[""],images=Image.open(image), return_tensors="pt", padding=True)#.unsqueeze(0)
    else:
        image = preprocess(text=[""],images=image, return_tensors="pt", padding=True)#.unsqueeze(0)
    # print(image)
    image = image["pixel_values"].to(device)
    # print(image.shape)
    with torch.no_grad():
        image_features = model(image)["pooler_output"]
        # print(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def apply_tilt(image):
    width, height = image.size
    new_width = int(width * 0.8)  # 50% of the original width
    new_height = int(height * 0.8)  # 50% of the original height

    # Resize the image to the new dimensions
    zoomed_out_image = image.resize((new_width, new_height))
    padding_left = (width - new_width) // 2
    padding_top = (height - new_height) // 2
    padding_right = width - new_width - padding_left
    padding_bottom = height - new_height - padding_top

    # Pad the image with a background color (e.g., white or black)
    padded_image = ImageOps.expand(zoomed_out_image, 
                                border=(padding_left, padding_top, padding_right, padding_bottom),
                                fill='black')  # You can change the fill color to 'black' or 'transparent'

    original_points = np.array([
        [0, 0],  # Top-left corner
        [width - 1, 0],  # Top-right corner
        [0, height - 1],  # Bottom-left corner
        [width - 1, height - 1]  # Bottom-right corner
    ], dtype='float32')
    new_points = np.array([[[0, -350],  # New top-left
        [width, -350],  # New top-right
        [0, height + 350],  # Bottom-left corner
        [width, height + 350]  # New bottom-right
    ]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(original_points, new_points)

    # Apply the perspective transformation
    transformed_image = padded_image.transform(
        (width, height),  # Output size
        Image.AFFINE,  # Specify the type of transformation
        matrix.flatten(),  # Flatten the matrix to pass as a parameter
        resample=Image.BICUBIC  # Use cubic resampling for smoother results
    )
    return transformed_image
def apply_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=5))

# Function to save features to a file
def save_features(image_paths, feature_file):
    features = {}
    transforms = ["none",apply_tilt,apply_blur]
    for img_path in image_paths:
        for i, transform in enumerate(transforms):
            image = Image.open(img_path)
            if(transform != "none"):
                image = transform(image)
                if(i == 1):
                    for ii in range(2):
                        if(ii == 1):
                            image = apply_blur(image)
                        features[img_path + "/" + str(transform) + "/" +str(ii)] = extract_clip_features(image)
                else:
                    features[img_path + "/" + str(transform)] = extract_clip_features(image)
            
    
    # Save features as a dictionary (or list) to a file (pickle or JSON)
    with open(feature_file, 'wb') as f:
        pickle.dump(features, f)


# Example usage:
# base_dir_signs = "traffic_signs/uk/"
# signs = os.listdir(base_dir_signs)
# image_paths = [base_dir_signs + image_path for image_path in signs if ".png" in image_path]  # List of image paths
# save_features(image_paths, "sign_features_uk.pkl")

# Load the CLIP model

# Function to load saved features
def load_features(feature_file):
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    return features

# Function to compare the new image with saved features
def compare_with_saved_features(new_image, feature_file,model,preprocess):
    # Extract features of the new image
    new_image_features = extract_clip_features(new_image,model,preprocess)
    
    # Load saved features
    saved_features = load_features(feature_file)
    
    # Compute cosine similarity between the new image features and all saved image features
    similarities = {}
    # similarity = cosine_similarity(new_image_features, saved_features_vector)
    
    for image_path, saved_features_vector in saved_features.items():
        similarity = cosine_similarity(new_image_features, saved_features_vector)
        similarities[image_path] = similarity[0][0]  # Extract scalar similarity
    
    # Sort similarities (highest to lowest)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_similarities
from tqdm import tqdm


def get_sign_category(start_idx,end_idx,preprocess,model,save_imgs=False):
    # lingo_path = "../LLaVA-NeXT/datasets/lingoqa/"
    # tasks = pd.read_parquet(lingo_path + "val.parquet").to_numpy()
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    scene_path = "../AutoSeg-SAM2/processed_lingoqa/"
    

    scenes = os.listdir(scene_path)
    info_dict = None
    with open("uk/info_dict.pkl","rb") as f:
        info_dict = pickle.load(f)
    dataset_paths = ["../lingo/LingoQA/evaluation/images/val/","../lingo/LingoQA/action/images/images/train/","../lingo/LingoQA/scenery/images/images/train/"]

    for scene in tqdm(scenes[int(start_idx):int(end_idx)]):
        try:
            with open("../AutoSeg-SAM2/processed_lingoqa/{}/objects_traffic_signs.pkl".format(scene), 'rb') as file:
                file = pickle.load(file)
            continue
        except:
            pass
        dataset_idx = 10000
        for i in range(len(dataset_paths)):
            try:
                frame_path = dataset_paths[i] + scene + "/{}.jpg".format(0)
                image = Image.open(frame_path)
                dataset_idx = i
                break
            except:
                pass


        # id_, scene, images, question, answer = task
        # if(scene in scenes):
        #     continue
        # print(scene)
        distance_maps = []
        sim_threshold = 0.8
        prev_sim_threshold = sim_threshold + 0.3

        images = [dataset_paths[dataset_idx] + scene + "/{}.jpg".format(i) for i in range(5)]
        # video = utils.create_video_from_frames(images,scene + ".mp4",fps=0.5)
        # import torch
        object_list = None
        # print(scene)
        with open("../AutoSeg-SAM2/processed_lingoqa/{}/objects.pkl".format(scene), 'rb') as file:
            object_list = pickle.load(file)
        prompt = ""
        image_0 = Image.open(images[0])
        traffic_signs = []
        for i, objects_per_frame in enumerate(object_list):
            image = Image.open(images[i])
            prompt += "frame nr: {}\n".format(i)
            traffic_signs_per_frame = []
            for ii, object_ in enumerate(objects_per_frame):
                bbox,obj_id,category_id,depth = object_
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                # mean_y = (bbox[3] - bbox[1]) + bbox[1]
                # print(image_0.width,depth)
                # x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                # y = ((mean_y - image_0.height/2) * 1/depth) / 1030
                # prompt += "a {}, with id: {}, horizontal position: {}, distance: {}\n".format(categories[category_id],obj_id,x,1/depth)
                if(category_id == 7):
                    cropped_image = image.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
                    out = model
                    similarities = compare_with_saved_features(cropped_image,"sign_features_uk.pkl",model,preprocess)
                    if(similarities[0][1] > sim_threshold and similarities[0][1] < prev_sim_threshold):
                        # if("670V" not in similarities[0][0]):
                        #     continue
                        if(save_imgs):
                            img_nr = 0
                            path = "{}_{}.png".format(i,img_nr)
                            try:
                                while True:
                                    path = "{}_{}.png".format(i,img_nr)
                                    image = Image.open(path)
                                    img_nr += 1
                                print(similarities[0])
                                continue
                            except:
                                cropped_image.save(path)
                        
                        
                        
                        sign_name = similarities[0][0].split(".png")[-2].split("/")[-1].replace("_transform_0","").replace("_transform_1","")
                        # print(sign_name)
                        try:
                            sign_info = info_dict[sign_name]
                        except:
                            continue
                        # print(i,img_nr,similarities[0][0],similarities[0][0].split(".png")[-2].split("/")[-1],sign_info)
                        # print(i,obj_id,img_nr,similarities[0][0].split(".png")[-2].split("/")[-1],similarities[0][1])
                        # print("C"+5)
                        traffic_signs_per_frame.append([bbox,obj_id,category_id,depth,sign_info["header"],sign_info["description"]])
                    else:
                        similarities[0]
            traffic_signs.append(traffic_signs_per_frame)
        scenes.append(scene)
        # continue
        with open("../AutoSeg-SAM2/processed_lingoqa/{}/objects_traffic_signs.pkl".format(scene), 'wb') as f:
            pickle.dump(traffic_signs, f)
        # if(task_nr >= 300):
        #     print("c"+5)
# get_sign_category()

def to_dict():
    with open("uk/info.pkl","rb") as f:
        info = pickle.load(f)

    sign_dict = {}
    for info_ in info:
        image_path, image_link, header, desc = info_

        key_ = image_link.split("-")[-1].replace(".svg.png","")
        sign_dict[key_] = {"image_path":image_path,
                        "image_link":image_link,
                        "header":header,
                        "description":desc}

    with open("uk/info_dict.pkl", 'wb') as f:
        pickle.dump(sign_dict, f)

def go(thread_name,start_idx,end_idx):
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPVisionModel.from_pretrained('tanganke/clip-vit-large-patch14_gtsrb')
    # Load the CLIP model
    

    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    model.to(device)
    get_sign_category(start_idx,end_idx,preprocess,model)
    # detect_traffic_lights(start_idx,end_idx,False)
import threading
with torch.no_grad():
    thread_list = []
    total = 30000
    threads = 12
    for i in range(threads):
        thread_list.append(threading.Thread(target=go, args=("Thread-{}".format(i),i * (total / threads) / 2 + gpu * (total / 2),(i+1) * (total / threads) / 2 + gpu * (total / 2))))
        thread_list[i].start()
        time.sleep(5)
# Example usage:
# new_image_path = "path_to_new_image.jpg"
# similarities = compare_with_saved_features(new_image_path, "clip_features.pkl")
# print("Similarities:", similarities)

