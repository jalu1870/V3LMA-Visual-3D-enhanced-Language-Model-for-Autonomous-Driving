import math
import cv2
import os

import numpy as np
from PIL import Image    
from decord import cpu, VideoReader, bridge
import io
import av
import torch
import natsort

from torchvision import transforms
from transformers import StoppingCriteria, StoppingCriteriaList
import pickle
import pandas as pd
from tqdm import tqdm
import shutil

def prepare_dataset(examples):
    # Get the texts and images, and apply the chat template
    
    # print(messages)

    # Preparation for inference
  
    # Inference: Generation of the output

    if("labels" in examples):
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
        processor = processor
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
            ] for video_path, prompt_vlm, label in zip(examples["video_path"],examples["prompt_vlm"])
        ]
        processor = val_processor

    # curr = time.time()
    # print(curr-start)
    texts = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in messages
    ]  # Prepare texts for processing
    # curr1_ = time.time()
    # print(curr1_-curr)
    # dataset_paths = ["../lingo/LingoQA/evaluation/images/val/","../lingo/LingoQA/action/images/images/train/","../lingo/LingoQA/scenery/images/images/train/"]

    
    # for video_path in examples["video_path"]:
    #     scene = video_path.split("/")[-2]
    #     dataset_idx = 10000
    #     for i in range(len(dataset_paths)):
    #         try:
    #             frame_path = dataset_paths[i] + scene + "/{}.jpg".format(0)
    #             image = Image.open(frame_path)
    #             dataset_idx = i
    #             break
    #         except:
    #             pass
    #     scene_path = dataset_paths[dataset_idx] + scene + "/"
    #     image_names = os.listdir(scene_path)
    #     image_paths = [scene_path + image_name for image_name in image_names]
    #     image_paths = natsort.natsorted(image_paths)
    #     video = utils.create_video_from_frames(image_paths, video_path, fps=1)
    # image_inputs = [process_vision_info(example)[1][0] for example in messages]  # Process the images to extract inputs
    # image_inputs = image_inputs[0]#[0].to(torch.uint8)
    # torch.save(image_inputs.to(torch.uint8), examples["video_path"][0].replace(".mp4",".pth"))
    batch_size = len(texts)
    image_inputs = [torch.load(examples["video_path"][i].replace(".mp4",".pth"),weights_only=False).to(torch.float32) for i in range(batch_size)]

    # curr1 = time.time()
    # print(curr1-curr1_)
    # Tokenize the texts and process the images
    inputs = processor(
        text=texts, videos=image_inputs, padding=True, return_tensors="pt"#
    )  # Encode texts and images into tensors
    
    # curr2 = time.time()
    # print(curr2-curr1)
    inputs["pixel_values_videos"] = [inputs["pixel_values_videos"][i * int(len(inputs["pixel_values_videos"]) / batch_size):(i + 1) * int(len(inputs["pixel_values_videos"]) / batch_size)] for i in range(batch_size)]
    inputs["pixel_values_videos"] = torch.stack(inputs["pixel_values_videos"])

    # inputs['labels'] = processor.tokenizer(examples['labels'], return_tensors="pt", padding=True).input_ids
    if("labels" in examples):
        input_ids_lists = inputs['input_ids'].tolist()
        assert len(messages) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            labels_list.append(label_ids)


        # print("ii")
        # ending_ids = [[151644, 77091,    198]]
        # ending_att = [[1,1,1]]
        # inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.tensor(ending_ids)), dim=1)
        # inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.tensor(ending_att)), dim=1)

        # curr3 = time.time()
        # print(curr3-curr2)
        inputs['labels'] = torch.tensor(labels_list, dtype=torch.int64)
    return inputs


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



def prepare_lingoqa_data_old():
    dataset_path = "datasets/lingoqa/"
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    


    # Load a .parquet file
    tasks = pd.read_parquet(r"datasets/lingoqa/val.parquet")
    
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
        distance_maps = []

        images = ["datasets/lingoqa/" + image for image in images]
        # video = utils.create_video_from_frames(images,scene + ".mp4",fps=0.5)
        # import torch
        object_list = None
        with open("../AutoSeg-SAM2/processed_lingoqa/{}/objects.pkl".format(scene), 'rb') as file:
            object_list = pickle.load(file)
        try:
            with open("../AutoSeg-SAM2/processed_lingoqa/{}/objects_traffic_signs.pkl".format(scene), 'rb') as file:
                sign_list = pickle.load(file)
        except:
            sign_list = []
        try:
            with open("../AutoSeg-SAM2/processed_lingoqa/{}/objects_traffic_lights.pkl".format(scene), 'rb') as file:
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
                prompt += "a {}, with id: {}, at horizontal position: {} and distance: {}\n".format(categories[category_id],obj_id,x,1/depth)

            for object_ in sign_list[i]:
                bbox,obj_id,category_id,depth,header,descr = object_
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                prompt += "a traffic sign with description :{}, at horizontal position: {} and distance: {}\n".format(header + " " + descr,x,1/depth)

            for object_ in light_list[i]:
                bbox,state,category_id,depth = object_
                mean_x = (bbox[2] - bbox[0]) + bbox[0]
                x = ((mean_x - image_0.width/2) * 1/depth) / 1030
                prompt += "a traffic light with state {}, at horizontal position: {} and distance: {}\n".format(state,x,1/depth)
        # if(len(total_lights) != 0 and len(total_signs) != 0):
        #     print(prompt)
        #     print(sign_list,light_list)
        #     print("C"+5)
        task_data.append([id_, scene, images, question, answer, prompt])

    return text_prompt_relative, task_data


from qwen_vl_utils import process_vision_info
def prepare_lingoqa_data(start_idx,end_idx,dataset_path,create_vid_from_frames,preprocess_video):
    categories = ["person","bicycle","car","motorcycle","bus","truck","traffic light","traffic sign","sky","wall","building","road","street","backpack","bag","other"]
    
    scene_path = "/processed_lingoqa/"
    scenes = os.listdir(scene_path)

    # Load a .parquet file
    # tasks = pd.read_parquet(parquet_path)
    dataset_paths = [dataset_path]

    
    # print(len(tasks))
    # print(tasks.head())
    # print("C"+5)
    # predictor, cityscapes_metadata = load_mask2former()
    moveable_objects_ids = [11,12,13,14,15,16,17,18]
    signs_lights = []

    signs_lights_ids = [6,7]

    task_data = []

    for scene in tqdm(scenes[start_idx:end_idx]):
        # try:
        #     with open("../AutoSeg-SAM2/processed_lingoqa/{}/prompt.pkl".format(scene), 'rb') as file:
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

        #_no_yolo
        object_list = None
        try:
            with open("processed_lingoqa/{}/objects.pkl".format(scene), 'rb') as file:
                object_list = pickle.load(file)
        except:
            print(scene," not available, skipping")
            continue
        try:
            with open("processed_lingoqa/{}/objects_traffic_signs.pkl".format(scene), 'rb') as file:
                sign_list = pickle.load(file)
        except:
            continue
            sign_list = []
        try:
            with open("processed_lingoqa/{}/objects_traffic_lights.pkl".format(scene), 'rb') as file:
                light_list = pickle.load(file)
        except:
            continue
            light_list = []
        
        # total_signs = [n for fr_lis in sign_list for n in fr_lis]
        # total_lights = [n for fr_lis in light_list for n in sign_list]
        video_path = "processed_lingoqa/{}/video.mp4".format(scene)
        if(create_vid_from_frames):
        
        # scene_path = dataset_paths[dataset_idx] + scene + "/"
            image_names = os.listdir(scene_path)
            image_paths = [scene_path + image_name for image_name in image_names]
            image_paths = natsort.natsorted(image_paths)
            video = create_video_from_frames(image_paths, video_path, fps=1)
        if(preprocess_video):
            # continue 
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

            # image_inputs = torch.load("../AutoSeg-SAM2/processed_lingoqa/{}/video.pth".format(scene))
            # os.remove("../AutoSeg-SAM2/processed_lingoqa/{}/video.mp4".format(scene))
        
        # continue
        prompt = ""
        image_0 = Image.open(images[0])
        print(len(object_list),object_list[0])
        print("C"+5)
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

        with open("../AutoSeg-SAM2/processed_lingoqa/{}/prompt.pkl".format(scene), 'wb') as f:
            pickle.dump(prompt, f)

    return

        

def load_video_vid_chat(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6, padding=False):
    from video_chat import hd_utils
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    print(type(frames))
    # print(frames)
    frames_numpy = frames.asnumpy()

    # Convert NumPy array to PyTorch tensor
    frames = torch.from_numpy(frames_numpy)

    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = hd_utils.HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = hd_utils.HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    # print(frames.shape)
    
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    # print(f"n_position: {n_position}")
    # print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
def ask(text, conv):
    conv.messages.append([conv.roles[0], text])


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    # if print_res:
    #     # print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()

def get_questions():
    questions = [
    "Describe the ego-vehicles actions in this scene and give reasons for its behaviour."
    # "What should the ego vehicle be especially careful of in this scene?",
    # "Are there groups of pedestrians in the scene that should be modeled as a swarm? Where are they located?",
    # "Where is the white van which is followed by the ego vehicle going towards the end of the scene?",
    # # "Describe the traffic rules at the intersection which is approached by the ego-vehicle.",
    # "What are the right-of-way rules at the intersection?",
    # "Is the intended path of the ego-vehicle blocked such that it needs to deviate in its path throughout the scene?",
    # "Are there traffic lights in the scene? What is their state?",
    # "Are there traffic signs in the scene? What is their state?",
    # "What is the traffic state?",
    # "Why did the truck to the left stop?",
    # "Does the ego-vehicle need to stop at the traffic light to follow its path?"
    ]
    return questions


def create_frame_grid(img_array, interval_width=50):
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))

    horizontal_band = np.ones((h, interval_width, c),
                              dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1)
                            * (w + interval_width), c), dtype=img_array.dtype) * 255

    rows = []
    for i in range(grid_size):
        row_frames = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n:
                frame = img_array[idx]
            else:
                frame = np.ones_like(img_array[0]) * 255
            if j > 0:
                row_frames.append(horizontal_band)
            row_frames.append(frame)
        combined_row = np.concatenate(row_frames, axis=1)
        if i > 0:
            rows.append(vertical_band)
        rows.append(combined_row)

    final_grid = np.concatenate(rows, axis=0)
    return final_grid


def resize_image_grid(image, max_length=1920):
    width, height = image.size
    if max(width, height) > max_length:
        if width > height:
            scale = max_length / width
        else:
            scale = max_length / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = image.resize((new_width, new_height), Image.BILINEAR)
    else:
        img_resized = image
    return img_resized


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video_share(video_path, num_segments=8, return_msg=False, num_frames=4):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)
    img_array = vr.get_batch(frame_indices).asnumpy()
    img_grid = create_frame_grid(img_array, 50)
    img_grid = Image.fromarray(img_grid).convert("RGB")
    img_grid = resize_image_grid(img_grid)
    img_grid.save("img.jpg")
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return img_grid, msg
    else:
        return img_grid
        
def clip_to_last_n_digits(number, n):
    # Convert the number to string and get the last n digits
    clipped_str = str(number)[-n:]
    # Convert back to integer (or leave it as a string if preferred)
    return int(clipped_str)

def read_video_pyav_aurora(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def load_video_THUDM(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data

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
    # print(f"Video saved at {output_video_path}")


def store_vlm_results(video_id, prompt,result,output_file):
    """
    Calls the video-language model and stores the video ID, prompt, and result in a specified output file.

    Args:
        video_id (str): The unique identifier for the video.
        video_path (str): The path to the input video.
        prompt (str): The prompt for the VLM.
        output_file (str): The file where the output will be stored.
    """
    


    # Store the video ID, prompt, and result in the output file
    with open(output_file, 'a') as file:
        file.write(f"Video ID: {video_id}\n")
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Result: {result}\n")
        file.write("\n")  # Add a blank line for better readability

    print(f"Results stored in {output_file}")

def load_video_longVQ(video_path,max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def read_video_pyav(video_path):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames/4).astype(int)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def quaternion_to_euler(x, y, z, w):
    # print(x, y, z, w)
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = 0.0
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # 90 degrees
    else:
        pitch = math.degrees(math.asin(sinp))
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw

def rotation_matrix(roll, pitch, yaw):
    # Yaw rotation matrix (R_z)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Pitch rotation matrix (R_y)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Roll rotation matrix (R_x)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation matrix (R = Rz * Ry * Rx)
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

# Function to determine if the object is behind the car
def is_object_behind_car(car_pos, roll, pitch, yaw, object_pos):
    # Rotation matrix from yaw, pitch, and roll
    R = rotation_matrix(roll, pitch, yaw)
    
    # Car's local forward vector (assumes the car is facing along the z-axis in local coordinates)
    forward_local = np.array([0, 0, 1])
    
    # Apply the rotation to get the car's forward vector in world coordinates
    forward_world = np.dot(R, forward_local)
    
    # Calculate the vector from the car to the object
    car_to_object = np.array(object_pos) - np.array(car_pos)
    
    # Compute the dot product to check if the object is behind
    dot_product = np.dot(forward_world, car_to_object)
    
    if dot_product > 0:
        return False  # The object is in front of the car
    elif dot_product < 0:
        return True   # The object is behind the car
    else:
        return None   # The object is to the side of the car
    


def construct_dynamic_nuscenes_prompt(scene_data):
    objects_of_interest = [
        "vehicle",
        "pedestrian",
        "animal",
        "traffic_light"
    ]

    dynamic_prompt = ""
    each_nth_frame = 6

    for i, frame_data in enumerate(scene_data):
        if(i % each_nth_frame != 0):
            continue
        ego_pose = frame_data[3]
        dynamic_prompt += "Frame {} at time {}\n".format(i / each_nth_frame, clip_to_last_n_digits(ego_pose["timestamp"],6))
        
        
        roll, pitch, yaw = quaternion_to_euler(*ego_pose["rotation"])
        dynamic_prompt += "ego_vehicle_position: {}, ego_vehicle_rotation: {}, {}, {}\n".format(ego_pose["translation"], roll, pitch, yaw)
        dynamic_prompt += "vehicles and pedestrians in the frame:\n"
        anns_data = frame_data[4]
        for ann_data in anns_data:
            translation, rotation, size, category = ann_data
            in_ooi = [True for cat in objects_of_interest if cat in category]
            if(True not in in_ooi):
                continue
            new_translation = []
            new_rotation = []
            new_size = []
            if(len(rotation) == 3):
                roll, pitch,yaw = rotation
            else:
                roll, pitch, yaw = quaternion_to_euler(*rotation)
            is_behind_car = is_object_behind_car(ego_pose["translation"], roll, pitch, yaw,translation)
            if(is_behind_car == True):
                continue
            for trans in translation:
                new_translation.append(round(trans,2))
            # for rot in rotation:
            #     new_rotation.append(round(rot,2))
            roll, pitch, yaw = round(roll,2), round(pitch,2), round(yaw,2)
            for siz in size:
                new_size.append(round(siz,2))

            category = category.replace("vehicle.","").replace("human.pedestrian.","")
            category = category.replace("emergency.","emergency vehicle ")
            dynamic_prompt += "{}, position: {}, rotation: {}, {}, {}, size: {}\n".format(category,new_translation,roll,pitch,yaw,new_size)
            # dynamic_prompt += "{}, position: {}\n".format(category.replace("human.pedestrian","").replace("vehicle.car","car").replace("vehicle.bus","bus").replace("vehicle.car","car"),new_translation)

    # print(dynamic_prompt)
    d = dynamic_prompt.count("\n")
    return dynamic_prompt.replace("[","").replace("]","")

# def prep_lingoqa_old

#         # from sam2.build_sam import build_sam2_video_predictor

#         # checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
#         # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#         # predictor = build_sam2_video_predictor(model_cfg, checkpoint)
#         # # print(predictor)

#         # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#         #     state = predictor.init_state(scene + ".mp4")

#         #     # add new prompts and instantly get the output on the same frame
#         #     frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, "instance-wise segmentation")

#         #     # propagate the prompts to get masklets throughout the video
#         #     for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#         #         print(masks)
#         # print("C"+5)
#         # load Mask2Former trained on YouTubeVIS 2021 instance segmentation
#         processor = AutoImageProcessor.from_pretrained("shivi/video-mask2former-swin-tiny-youtubevis-2021-instance")
#         model = Mask2FormerForUniversalSegmentation.from_pretrained("shivi/video-mask2former-swin-tiny-youtubevis-2021-instance")

#         # file_path = hf_hub_download(repo_id="shivi/video-demo", filename="cars.mp4", repo_type="dataset")
#         video = torchvision.io.read_video(scene + ".mp4")[0]
#         video_frames = [processor(images=frame, return_tensors="pt").pixel_values for frame in video]
#         print(video_frames)
#         video_input = torch.cat(video_frames)

#         with torch.no_grad():
#             outputs = model(**video_input)

#         # model predicts class_queries_logits of shape `(batch_size, num_queries, num_classes)`
#         # and masks_queries_logits of shape `(num_queries, batch_size, height, width)`
#         class_queries_logits = outputs.class_queries_logits
#         masks_queries_logits = outputs.masks_queries_logits

#         # you can pass them to processor for postprocessing
#         result = processor.post_process_video_instance_segmentation(outputs, target_sizes=[tuple(video.shape[1:3])])[0]
#         # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
#         predicted_video_instance_map = result["segmentation"]
#         print(predicted_video_instance_map)
#         print("C"+5)
#         for image_name in images:
#             print(dataset_path + image_name)
#             image_path = dataset_path + image_name
#             image = cv2.imread(image_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             output_distances = run_depth_estimation(midas_model, midas_transform,image).cpu().numpy()
#             distance_maps.append(output_distances)
            
#             # distance_in_meters = get_distance_in_meters(output_depth,pixel_coordinates,estimated_distance)
#             # cv2.imwrite("{}.png".format(image_name.split("/")[-1]), output_distances)

#             output_seg = predictor(image)
#             # v = Visualizer(image[:, :, ::-1], cityscapes_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
#             # instance_result = v.draw_instance_predictions(output_seg["instances"].to("cpu")).get_image()
#             # panoptic_result = v.draw_panoptic_seg(output_seg["panoptic_seg"][0].to("cpu"), output_seg["panoptic_seg"][1]).get_image()

#             # print(outputs)
#             panoptic, objects = output_seg["panoptic_seg"]
            
#             image_height = image.shape[0]
#             image_width = image.shape[1]
#             for output in objects:
#                 id_ = output["id"]
#                 cat_id = output["category_id"]
#                 if(cat_id not in signs_lights_ids):
#                     continue
#                 panoptic_spec = panoptic == id_ 
#                 positions = np.argwhere(panoptic_spec.cpu())
#                 points = positions.transpose(0,1).cpu().numpy()
#                 x, y = points[:,0] , points[:,1]
#                 # print(x, y)
#                 object_center = np.mean(x), np.mean(y)
#                 z = output_distances[x,y]
#                 mean_depth = np.mean(z)

#                 x = ((points[:,1] - image_width/2) * z) / 1030
#                 y = ((points[:,0] - image_height/2) * z) / 1030
#                 print(output)
                
#                 print(cat_id,object_center, mean_depth, np.mean(x),np.mean(y))

#                 # print("C"+5)
#             # print(outputs)
#             cv2.imwrite("instance_seg_{}.png".format(image_name.split("/")[-1]), instance_result)
#             cv2.imwrite("panoptic_seg_{}.png".format(image_name.split("/")[-1]), panoptic_result)
            
#             print("C"+5)
        
