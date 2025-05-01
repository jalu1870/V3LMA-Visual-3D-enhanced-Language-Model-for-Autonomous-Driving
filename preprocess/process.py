import numpy as np
import os
from PIL import Image
import json
import cv2
from hydra import compose, initialize
import time
import debugpy
import shutil
import gc
import os
import cv2
import torch
import numpy as np
# import supervision as sv
from PIL import Image
import sys
print(os.getcwd())
gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

import json
import copy
import time



import pickle
from tqdm import tqdm
import torch
import natsort
from sklearn.metrics.pairwise import cosine_similarity

from qwen_vl_utils import process_vision_info
import pandas as pd
from datasets import load_dataset

from transformers import CLIPVisionModel
from PIL import Image
import numpy as np
from transformers import CLIPProcessor

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

def load_features(feature_file):
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    return features

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


def get_sign_category(start_idx,end_idx,dataset_path,preprocess,model,save_imgs=False):
    # lingo_path = "../LLaVA-NeXT/datasets/lingoqa/"
    # tasks = pd.read_parquet(lingo_path + "val.parquet").to_numpy()
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    scene_path = "processed_lingoqa/"
    

    scenes = os.listdir(scene_path)
    info_dict = None
    with open("traffic_signs/uk/info_dict.pkl","rb") as f:
        info_dict = pickle.load(f)
    dataset_paths = [dataset_path]
    
    for scene in tqdm(scenes[int(start_idx):int(end_idx)]):
        try:
            with open("processed_lingoqa/{}/objects_traffic_signs.pkl".format(scene), 'rb') as file:
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

        distance_maps = []
        sim_threshold = 0.8
        prev_sim_threshold = sim_threshold + 0.3

        images = [dataset_paths[dataset_idx] + scene + "/{}.jpg".format(i) for i in range(5)]
        
        object_list = None
        
        with open("processed_lingoqa/{}/objects.pkl".format(scene), 'rb') as file:
            object_list = pickle.load(file)
        prompt = ""
        image_0 = Image.open(images[0])
        traffic_signs = []
        for i, objects_per_frame in enumerate(object_list):
            image = Image.open(images[i])
            print(images[i])
            prompt += "frame nr: {}\n".format(i)
            traffic_signs_per_frame = []
            for ii, object_ in enumerate(objects_per_frame):
                bbox,obj_id,category_id,depth = object_
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                if(category_id == 7):
                    cropped_image = image.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
                    out = model
                    similarities = compare_with_saved_features(cropped_image,"traffic_signs/sign_features_uk.pkl",model,preprocess)
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
                        
                        try:
                            sign_info = info_dict[sign_name]
                        except:
                            continue
                        traffic_signs_per_frame.append([bbox,obj_id,category_id,depth,sign_info["header"],sign_info["description"]])
                    else:
                        similarities[0]
            traffic_signs.append(traffic_signs_per_frame)
        scenes.append(scene)
        
        with open("processed_lingoqa/{}/objects_traffic_signs.pkl".format(scene), 'wb') as f:
            pickle.dump(traffic_signs, f)

def create_dataset(parquet_path,out_path):
    
    text_prompt_relative = """The following text describes the content of a traffic scene. You are given a sequence of frames.
    The scene is observed from the perspective of the ego-vehicle. For each object, the relative horizontal position and distance to the ego-vehicle
    are given. If the same object is observed in multiple frames that is indicated by it having the same id, -1 indicates that this object was not tracked across frames.
    Use this information to reason across the entire scene not per frame. Do not repeat the numbers for position and rotation. 
    Do not give additional information. Only answer the question directly and concisely.\n"""


    scene_id = []
    task_id = []
    video_paths = []
    vlm_prompts = []
    llm_prompts = []
    llm_prompts = []
    labels = []
    merge_feature_layers = []
    feature_weights = []
    eval_dataset = load_dataset('parquet', data_files=parquet_path, split='train')
    
    for sample in tqdm(eval_dataset):
        
        # question_id, segment_id, images, question,answer,__index_level_0__ = sample
        scene = sample["segment_id"]
        question = sample["question"]
        answer = sample["answer"]

        load_base_path = "preprocess/processed_lingoqa/"
        base_path = "processed_lingoqa/"

        try:
            with open(base_path + "{}/prompt_yolo_11.pkl".format(scene), 'rb') as file:
                prompt = pickle.load(file)
            print(base_path + "{}/prompt_yolo_11.pkl".format(scene))
        except:
            continue

        video_path = load_base_path + "{}/video.mp4".format(scene)

        llm_prompt = text_prompt_relative + question + "\n" + prompt
        vlm_prompt = question
        label = answer
        merge_feature_layers_ = [-1]
        feature_weights_ = [0.1,0.9]

        scene_id.append(scene)
        task_id.append(sample["question_id"])
        video_paths.append(video_path)
        llm_prompts.append(llm_prompt)
        vlm_prompts.append(vlm_prompt)
        labels.append(label)
        merge_feature_layers.append(merge_feature_layers_)
        feature_weights.append(feature_weights_)
        

    # Sample data: Create a simple DataFrame
    data = {
        "scene_id":scene_id,
        "task_id":task_id,
        'video_path': video_paths,
        'prompt_llm': llm_prompts,
        'prompt_vlm': vlm_prompts,
        'labels': labels,
        'merge_feature_layers': merge_feature_layers,
        'feature_weights': feature_weights,
    }

    df = pd.DataFrame(data)

    # Specify the output Parquet file name
    if(".parquet" in out_path):
        output_file = out_path
    else:
        output_file = out_path + ".parquet"

    # Save the DataFrame as a Parquet file using pyarrow
    df.to_parquet(output_file, engine='pyarrow')

def create_video_from_frames(frames, output_video_path, fps=12, invert_rb = True):
    """
    Creates a video from a list of frames.

    Args:
        frames (list): List of frames (numpy arrays) to be included in the video.
        output_video_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.
    """
    if(isinstance(frames[0],str)):
        frames = [[Image.open(frame)] for frame in frames]
    
    frames = [frame[0] for frame in frames]

    # Get the width and height from the first frame
    width, height = frames[0].size

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in frames:
        arr = np.array(frame)[:,:,::-1]#.permute(2,1,0)
        video_writer.write(arr)

    # Release the video writer
    video_writer.release()

def prepare_lingoqa_data(start_idx,end_idx,dataset_path,create_video,preprocess_videos):
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    
    object_path = "processed_lingoqa/"
    scenes = os.listdir(object_path)

    dataset_paths = [dataset_path]
    
    for scene in tqdm(scenes[start_idx:end_idx]):
        # try:
        #     with open("processed_lingoqa/{}/prompt.pkl".format(scene), 'rb') as file:
        #         file = pickle.load(file)
        #     continue
        # except:
        #     pass
        dataset_idx = 10000
        for i in range(len(dataset_paths)):
            try:
                frame_path = dataset_paths[i] + scene + "/{}.jpg".format(0)
                image = Image.open(frame_path)
                dataset_idx = i
                break
            except:
                pass
        if(dataset_idx == 10000):
            print(scene)

        images = [dataset_paths[dataset_idx] + scene + "/{}.jpg".format(i) for i in range(5)]

        object_list = None
        try:
            with open(object_path + "{}/objects.pkl".format(scene), 'rb') as file:
                object_list = pickle.load(file)
        except:
            print(scene," not available, skipping")
            continue
        try:
            with open(object_path + "{}/objects_traffic_signs.pkl".format(scene), 'rb') as file:
                sign_list = pickle.load(file)
        except:
            print(scene," not available, skipping")

            continue
            sign_list = []
        try:
            with open(object_path + "{}/objects_traffic_lights.pkl".format(scene), 'rb') as file:
                light_list = pickle.load(file)
        except:
            print(scene," not available, skipping")
            continue
            light_list = []
        
        # total_signs = [n for fr_lis in sign_list for n in fr_lis]
        # total_lights = [n for fr_lis in light_list for n in sign_list]

        video_path = "processed_lingoqa/{}/video.mp4".format(scene)
        if(create_video):
            scene_path = dataset_paths[dataset_idx] + scene + "/"
            image_names = os.listdir(scene_path)
            image_paths = [scene_path + image_name for image_name in image_names]
            image_paths = natsort.natsorted(image_paths)
            video = create_video_from_frames(image_paths, video_path, fps=1)
            
        if(preprocess_videos):
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 400 * 711,
                        "fps": 1.0,
                    }
                    ],
                }
            ]
            image_inputs = process_vision_info(messages)[1]
            torch.save(image_inputs[0].to(torch.uint8), "processed_lingoqa/{}/video.pth".format(scene))

        # # image_inputs = torch.load("../AutoSeg-SAM2/processed_lingoqa/{}/video.pth".format(scene))
        # os.remove("../AutoSeg-SAM2/processed_lingoqa/{}/video.mp4".format(scene))
        
        # continue
        prompt = ""
        image_0 = Image.open(images[0])
        # print(len(object_list),object_list[0])
        # print("C"+5)
        for i, objects_per_frame in enumerate(object_list):
            prompt += "frame nr: {}\n".format(i)
            for object_ in objects_per_frame:
                bbox,obj_id,category_id,depth = object_
                if(category_id > 5):
                    continue
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                prompt += "{}, id: {}, horizontal position: {}, distance: {}\n".format(categories[category_id],obj_id,round(x,3),round(1/depth,3))

            for object_ in sign_list[i]:
                bbox,obj_id,category_id,depth,header,descr = object_
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                prompt += "traffic sign, description :{}, horizontal position: {}, distance: {}\n".format(header + " " + descr,round(x,3),round(1/depth,3))

            for object_ in light_list[i]:
                
                bbox,state,category_id,depth = object_
                if(category_id != 6):
                    continue
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                if("green" in state):
                    state = "green"
                if("yellow" in state):
                    state = "yellow"
                if("red" in state):
                    state = "red"
                prompt += "traffic light, state {}, horizontal position: {}, distance: {}\n".format(state,round(x,3),round(1/depth,3))

        with open(object_path + "{}/prompt_yolo_11.pkl".format(scene), 'wb') as f:
            pickle.dump(prompt, f)

    return

def run_depth_estimation(midas_model,transform,image):

    input_batch = transform(image).to("cuda")
    with torch.no_grad():
        prediction = midas_model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

def load_midas():
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas_model = torch.hub.load("intel-isl/MiDaS", model_type).to("cuda")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    return midas_model, transform

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]



def process_data_new(start_idx,dataset_path,video_predictor, image_predictor, processor, grounding_model, text):
    from grounded_sam2_tracking_demo_with_continuous_id import run_segmentation_tracking
    scenes = os.listdir(dataset_path)
    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
    # 'output_dir' is the directory to save the annotated frames
    
    # 'output_video_path' is the path to save the final video

    for scene in tqdm(scenes):#[start_idx:start_idx+1300]
        print(scene)
        output_dir = "processed_lingoqa/" + scene
        video_dir = dataset_path + scene
        output_video_path = "processed_lingoqa/" + scene
        try:
            f = np.load("processed_lingoqa/" + scene + "/segment_array.npy")
            continue
        except:
            pass
        run_segmentation_tracking(video_predictor, image_predictor, processor, grounding_model, text, video_dir, output_dir, output_video_path)
        # np.save("processed_lingoqa/" + scene + "/segment_array.npy", segments.cpu().numpy().astype(np.int16))


def load_model():
    from PIL import Image
    import requests

    from transformers import AutoProcessor, AutoModel
    from transformers import CLIPModel, CLIPProcessor, CLIPConfig
    # from transformers import AutoModel, AutoConfig, AutoTokenizer
    # import sys
    # sys.path.append("LLM2CLIP/llm2clip/")
    # from eva_clip import create_model_and_transforms
    # from llm2vec import LLM2Vec
    model_id = "zer0int/CLIP-GmP-ViT-L-14"
    # model_id = "openai/clip-vit-large-patch14"
    # config = CLIPConfig.from_pretrained(model_id)

    model = CLIPModel.from_pretrained(model_id).cuda()
    processor = CLIPProcessor.from_pretrained(model_id)
    
    # model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    # processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    return model, processor
    
def load_mask2former():
    import torch
    import torchvision
    
    # # import some common detectron2 utilities
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    
    from detectron2.data import MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config
    import sys
    sys.path.append("../LLaVA-NeXT/")
    from Mask2Former.mask2former import add_maskformer2_config  

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
    cfg.merge_from_file("../LLaVA-NeXT/Mask2Former/configs/cityscapes/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/panoptic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_064788.pkl'

    # videobase
    # cfg.merge_from_file("Mask2Former/configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml")
    # cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2021/video_maskformer2_swin_large_IN21k_384_bs16_8ep/model_final_4da256.pkl"
    # cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    # cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    predictor = DefaultPredictor(cfg)
    return predictor, cityscapes_metadata

def compute_iou(box1, box2):
    # Extract coordinates and dimensions from the boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Convert to (x1, y1, x2, y2) format
    # x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    # x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # Compute the coordinates of the intersection
    x1_intersection = max(x1_1, x1_2)
    y1_intersection = max(y1_1, y1_2)
    x2_intersection = min(x2_1, x2_2)
    y2_intersection = min(y2_1, y2_2)

    # Calculate the area of the intersection
    intersection_width = max(0, x2_intersection - x1_intersection)
    intersection_height = max(0, y2_intersection - y1_intersection)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of both bounding boxes
    area_box_1 = (x2_1-x1_1) * (y2_1-y1_1)
    area_box_2 = (x2_2-x1_2) * (y2_2-y1_2)

    # Calculate the union area
    union_area = area_box_1 + area_box_2 - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def run_with_custom_source(frame_path):
    # Initialize Hydra without running the main function (to avoid decorators)
    with initialize(config_path="../../detection/ultralytics/ultralytics_/yolo/configs/", job_name="my_job"):
        # Compose the config and override the `source` value
        cfg = compose(config_name="default.yaml", overrides=[f"source={frame_path}"])

    # Now you can call your `predict` function with the new config
    light_det = ultra.yolo.v8.detect.predict(cfg)

def classify_traffic_sign(image):
    # Step 1: Convert the image to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Step 4: Approximate the contour to a polygon and get the number of vertices
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Step 5: Classify based on shape
        if len(approx) == 8:  # Stop sign is typically an octagon
            return 'Stop Sign'
        elif len(approx) == 3:  # Yield sign is typically a triangle
            return 'Yield Sign'
        elif len(approx) == 4:  # Speed limit sign could be square/rectangular
            return 'Speed Limit Sign'
    
    # Step 6: If no shape matched, return unknown
    return 'Unknown Sign'

def traffic_sign_state_recognition(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize image for faster processing (optional)
    image = cv2.resize(image, (300, 300))

    # Step 7: Use color filtering for simple color-based classification (e.g., for speed limit signs)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # Step 8: Combine the contour-based and color-based classification
    result = classify_traffic_sign(image)
    if result == 'Unknown Sign' and np.sum(red_mask) > 1000:
        result = 'Red Sign Detected'  # Simple rule-based state (could be stop sign)

    return result

def detect_contours(image):
    # Step 1: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Apply edge detection (Canny or Thresholding)
    # Using Canny edge detector here
    edges = cv2.Canny(blurred, 50, 150)

    # Alternatively, you could use thresholding instead of Canny:
    # _, edges = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours in the edge-detected image
    contours, hierarchy = cv2.findContours(
        edges, 
        cv2.RETR_EXTERNAL,  # Retrieve only external contours
        cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal, vertical, and diagonal segments to save memory
    )
    print(contours)

    # Step 5: Draw contours on the original image
    result_image = image.copy()  # Make a copy of the original image to draw on
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # Draw all contours in green

    # Return the image with contours drawn
    return result_image
def detect_circles(image):
    # Step 1: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Step 3: Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT,  # Method used for detecting circles
        dp=1,                # Inverse ratio of accumulator resolution to image resolution
        minDist=5,          # Minimum distance between detected centers of circles
        param1=45,           # Higher threshold for the internal Canny edge detector
        param2=15,           # Threshold for center detection
        minRadius=3,        # Minimum radius to consider
        maxRadius=100        # Maximum radius to consider
    )
    # return circles

    # Step 4: If circles are detected, draw them on the image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Round the circle coordinates
        for (x, y, r) in circles:
            # Draw the outer circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

    # Return the image with circles drawn
    print(circles)
    return image
def classify_traffic_light_color(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the color ranges for red, yellow, and green in HSV
    # Red (covers two ranges in hue)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Green
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Create masks for each color
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Combine the two red masks
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Calculate the areas of each color
    red_area = np.sum(red_mask)
    yellow_area = np.sum(yellow_mask)
    green_area = np.sum(green_mask)

    # Determine the color with the maximum area
    max_area = max(red_area, yellow_area, green_area)
    
    if max_area == red_area:
        return 'Red'
    elif max_area == yellow_area:
        return 'Yellow'
    elif max_area == green_area:
        return 'Green'
    else:
        return 'Unknown'

from ultralytics import YOLO
from detectron2.utils.visualizer import Visualizer, ColorMode
def detect_objects(start_idx,end_idx,dataset_path,write_to_video=False):
    

    # load yolo esimation
    model, processor = load_model()
    # det_model = torch.hub.load('../../detection/yolov5/yolov5/', 'custom', path='yolov5x6.pt', source='local') 
    det_model = YOLO("yolo11x.pt")
    
    dataset_paths = [dataset_path]
    
    # load depth esimation
    midas_model, midas_transform = load_midas()
    midas_model.eval()

    scene_path = "processed_lingoqa/"
    

    scenes = os.listdir(scene_path)

    category_map = {0:0,1:1,2:2,3:3,5:4,7:5,9:6}
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    input_text = ["a " + category for category in categories]
    
    # iterate over scenes
    for scene in tqdm(scenes[int(start_idx):int(end_idx)]):
        print(scene)
        try:
            with open("processed_lingoqa/" + scene + "/objects.pkl", 'rb') as file:
                file = pickle.load(file)
            continue
        except:
            pass
        masks = np.load(scene_path + scene + "/segment_array.npy")
        unique_ = []
        for mask in masks:
            unique_ = unique_ + list(np.unique(mask))
        ids_1 = np.unique(unique_) 
        object_categories = np.full([np.max(ids_1)+1],0)

        # print(lingo_path + scene + "/{}.jpg".format(0))
        dataset_idx = 10000
        for i in range(len(dataset_paths)):
            try:
                frame_path = dataset_paths[i] + scene + "/{}.jpg".format(0)
                image = Image.open(frame_path)
                dataset_idx = i
                break
            except:
                pass

        if(write_to_video):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the XVID codec
            frame_0 = cv2.imread(dataset_paths[dataset_idx] + scene + "/{}.jpg".format(0))
            out = cv2.VideoWriter('videos/{}.mp4'.format(scene), fourcc, 1, (frame_0.shape[1], frame_0.shape[0])) 

        object_list = []
        
        # iterate over frames per scene
        for frame_nr, frame_masks in enumerate(masks):
            object_list_per_frame = []
            
            
            frame_path = dataset_paths[dataset_idx] + scene + "/{}.jpg".format(frame_nr)
            image = Image.open(frame_path)
            # print("C"+5)
            
            # depth esimation
            image_array = np.array(image)
            frame_masks = cv2.resize(frame_masks, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            ids_ = np.unique(frame_masks) 
            output_distances = run_depth_estimation(midas_model, midas_transform,image_array[:,:,::-1]).cpu().numpy()
         
         
            if(write_to_video):
                img = cv2.imread(frame_path)
            with torch.amp.autocast('cuda'):
                results = det_model(frame_path,verbose=False)
                
            detection_boxes = []
            # process detections
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                    conf = box.conf[0]      # Confidence score
                    class_id = int(box.cls[0])         # Class ID
                    
                    if(class_id in category_map):
                        class_id = category_map[class_id]
                    else:
                        continue

                    detection_boxes.append([int(x1), int(y1), int(x2), int(y2), conf, int(class_id)])
                    
            # iterate over object masks
            for id_ in ids_:
                if(id_ == 0):
                    continue
                id_mask = frame_masks == id_
                id_mask = np.tile(id_mask, (3, 1,1)).transpose(1,2,0)
                masked = image_array * id_mask

                # crop image
                mask_points = np.argwhere(id_mask)
                bbox = [np.min(mask_points[:,1]),np.min(mask_points[:,0]),np.max(mask_points[:,1]),np.max(mask_points[:,0])]

                # print(masked.shape,bbox)
                cropped = masked[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                if(cropped.shape[1] <= 3):
                    if(bbox[2]+4 < image_array.shape[0]):
                        bbox[2] = bbox[2]+4
                    else:
                        bbox[0] = bbox[0]-4
                if(cropped.shape[0] <= 3):
                    if(bbox[3]+4 < image_array.shape[1]):
                        bbox[3] = bbox[3]+4
                    else:
                        bbox[1] = bbox[1]-4

                cropped = masked[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                
                if(cropped.shape[0] <= 0 or cropped.shape[1] <= 0 or cropped.shape[0] > cropped.shape[1] * 100 or cropped.shape[1] > cropped.shape[0] * 100):
                    continue

                image = Image.fromarray(cropped,mode="RGB")

                # classifc objects in crops using clip
                inputs = processor(text=input_text, images=image, padding=True, return_tensors="pt")
                inputs = {key: value.cuda() for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                logits_per_image = outputs.logits_per_image
                text_probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()

                category_id = np.argmax(text_probs)
                

                # map yolo detections to masks
                max_iou = -1
                max_id_ = -1
                for box_id, box in enumerate(detection_boxes):
                    iou = compute_iou(box[:4],bbox)
                    if(iou > max_iou):
                        max_iou = iou
                        max_id = box_id
                if(max_iou > 0.35):
                    category_id = detection_boxes[max_id][-1]
                    # print(category_id,max_iou,bbox,detection_boxes[max_id])

                object_category_name = categories[category_id]
                object_categories[id_] = category_id
                
                
                crop_rel = cropped.shape[1]/cropped.shape[0]
                if(np.sum(id_mask) > 500000 and (category_id == 1 or category_id == 6 or category_id == 7) or crop_rel > 0.7 and category_id == 0):
                    continue
                
                # skip some classes
                if(category_id < 8):
                    # print()
                    # print(id_,bbox,object_category_name,category_id,np.sum(id_mask))
                    # if(category_id == 2 and bbox[3] > 1100 and bbox[2] > 1600):
                    #     continue
                    if(category_id == 6 or category_id == 2 and bbox[0] < 100 and bbox[1] > 900 and bbox[2] > 1000 and bbox[3] > 1200):
                        continue
                    object_list_per_frame.append([bbox,id_,category_id])

            iou_threshold = 0.35

            # add yolo detections that were not tracked by masks
            for box_id, new_box in enumerate(detection_boxes):
                new_category_id = new_box[-1]
                if(category_id == 6):
                    continue
                
                if(new_category_id == 2 or new_category_id == 4 or new_category_id == 5):
                    max_iou = iou_threshold
                    max_id = -1
                    for old_box in object_list_per_frame:
                        old_category_id = old_box[2]
                        if(new_category_id == old_category_id or new_category_id == 2 or new_category_id == 4 or new_category_id == 5):
                            # print(old_box)
                            iou = compute_iou(new_box[:4],old_box[0])
                            # print(new_box[:4],old_box[0],iou,new_category_id)
                            if(iou > max_iou):
                                max_iou = iou
                                max_id = box_id
                                break
                    if(max_id == -1):
                        # print(new_category_id,max_iou)
                        object_list_per_frame.append([new_box[:4],-1,new_category_id])
            
            # depth estimation per object
            for i, box in enumerate(object_list_per_frame):
                bbox,id_,category_id = box

                object_category_name = categories[category_id]
                if(write_to_video):
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(img, "{},{}".format(id_,object_category_name), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                z = output_distances[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                median_depth = np.median(z)
                object_list_per_frame[i].append(median_depth)
                
            if(write_to_video):
                out.write(img)
            object_list.append(object_list_per_frame)
            
        if(write_to_video):
            out.write(img)
            out.release()


        # Saving the object to a .pkl file
        with open("processed_lingoqa/" + scene + "/objects.pkl", 'wb') as file:
            pickle.dump(object_list, file)
            


def detect_traffic_lights(start_idx,end_idx,dataset_path,write_to_video=False,load_list = True):

    sys.path.append("traffic-light-detection/")
    from inference.inference_images import load_model as load_traffic
    model_path = "traffic-light-detection/model_weights/traffic_lights_yolov8x.pt"
    model_type = "v8"
    model_traffic = load_traffic(model_type, model_path)

    midas_model, midas_transform = load_midas()
    midas_model.eval()


    scene_path = "processed_lingoqa/"
    

    scenes = os.listdir(scene_path)
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    dataset_paths = [dataset_path]
    
        

    for scene in tqdm(scenes[int(start_idx):int(end_idx)]):
        try:
            with open("processed_lingoqa/" + scene + "/objects_traffic_lights.pkl", 'rb') as file:
                file = pickle.load(file)
            continue
        except:
            pass
        masks = np.load(scene_path + scene + "/segment_array.npy")
        ids_1 = np.unique(masks[0]) 
        dataset_idx = 10000
        for i in range(len(dataset_paths)):
            try:
                frame_path = dataset_paths[i] + scene + "/{}.jpg".format(0)
                image = Image.open(frame_path)
                dataset_idx = i
                break
            except:
                pass
        

        if(write_to_video):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the XVID codec
            frame_0 = cv2.imread(dataset_paths[dataset_idx] + scene + "/{}.jpg".format(0))
            out = cv2.VideoWriter('videos/{}.mp4'.format(scene), fourcc, 1, (frame_0.shape[1], frame_0.shape[0])) 
    
        object_list = []
        if(load_list):
            with open("processed_lingoqa/" + scene + "/objects.pkl", 'rb') as file:
                object_list = pickle.load( file)
        for frame_nr, frame_masks in enumerate(masks):
            object_list_per_frame = []
            frame_path = dataset_paths[dataset_idx] + scene + "/{}.jpg".format(frame_nr)
            
            
            # print("C"+5)
            image = Image.open(frame_path)
            image_array = np.array(image)
            
            output_distances = run_depth_estimation(midas_model, midas_transform,image_array[:,:,::-1]).cpu().numpy()

            img = cv2.imread(frame_path)

            img_ = cv2.imread(frame_path)
            img_ = Image.fromarray(img_[:,:,::-1])
            img_.save("{}.png".format(frame_nr))
            result = model_traffic.predict(img, verbose=False, imgsz=(1920, 1216), conf=0.35, iou=0.6)
                    
            to_draw = []
            for box in json.loads(result[0].to_json()):
                box_dims = box['box']
                result_args = {
                    "left": box_dims['x1'],
                    "top": box_dims['y1'],
                    "right": box_dims['x2'],
                    'bottom': box_dims['y2'],
                    'label': box['name']
                }
                object_list_per_frame.append([[int(box_dims['x1']), int(box_dims['y1']), 
                                              int(box_dims['x2']), int(box_dims['y2'])], box['name'], 6])
                to_draw.append(result_args)
            if(write_to_video):
                drawBoundingBoxes(img, "imgs/" + scene + "/{}.png".format(frame_nr), to_draw)
            
            for i, box in enumerate(object_list_per_frame):
                bbox,id_,category_id = box

                object_category_name = categories[category_id]
                if(write_to_video):
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(img, "{},{}".format(id_,object_category_name), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                z = output_distances[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                median_depth = np.median(z)
                object_list_per_frame[i].append(median_depth)
                
            if(write_to_video):
                out.write(img)
            if(len(object_list) > frame_nr):
                object_list[frame_nr] = object_list[frame_nr] + object_list_per_frame
            else:
                object_list.append(object_list_per_frame)
            
        if(write_to_video):
            out.write(img)
            out.release()
            
        with open("processed_lingoqa/" + scene + "/objects_traffic_lights.pkl", 'wb') as file:
            pickle.dump(object_list, file)


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)


# VERY important: text queries need to be lowercased + end with a dot
text = "truck.motorbike.car.pedestrian.bicycle.traffic light.traffic sign."


# dataset_path = "../lingo/LingoQA/evaluation/images/val/"

import argparse
parser = argparse.ArgumentParser(description="inference arguments are model_name, val_data_path, llm_prompt_for_vision.")

# Add arguments
parser.add_argument('--dataset_path', type=str, help='path to the directory containing the scene files')
parser.add_argument('--dataset_parquet_path', type=str, help='path to the parquet file of the dataset')
parser.add_argument('--output_path', type=str, help='path to output the dataset file to, a .parquet file')
args = parser.parse_args()

def go(thread_name,start_idx,end_idx):
    sys.path.append("Grounded-SAM-2/")
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
    
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    dataset_path = args.dataset_path
    print(dataset_path)
    process_data_new(start_idx,dataset_path,video_predictor, image_predictor, processor, grounding_model, text)
    detect_objects(start_idx,end_idx,dataset_path,False)
    preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    sign_model = CLIPVisionModel.from_pretrained('tanganke/clip-vit-large-patch14_gtsrb')
    
    sign_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    sign_model.to(device)
    detect_traffic_lights(start_idx,end_idx,dataset_path,False)
    get_sign_category(start_idx,end_idx,dataset_path,preprocessor,sign_model)

    import pickle



    # print(data)  # Display the contents of the file
    create_video = True
    preprocess_videos = True
    prepare_lingoqa_data(start_idx,end_idx,dataset_path,create_video,preprocess_videos)
    out_path = args.output_path
    data_parquet_path = args.dataset_parquet_path
    create_dataset(data_parquet_path,out_path)

    

go("-",0,1000000)
# with torch.no_grad():
#     thread_list = []
#     total = 30000
#     threads = 8
#     for i in range(threads):
#         thread_list.append(threading.Thread(target=go, args=("Thread-{}".format(i),i * (total / threads) / 2 + gpu * (total / 2),(i+1) * (total / threads) / 2 + gpu * (total / 2))))
#         thread_list[i].start()
#         time.sleep(5)
#     # go(15000)
    
    
    # detect_traffic_lights(dataset_path,False)
# get_classes()