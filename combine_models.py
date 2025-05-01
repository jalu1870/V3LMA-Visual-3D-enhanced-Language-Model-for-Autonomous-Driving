import torch

        
from transformers.generation import GenerationMixin
from torch import nn
import utils

import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast, _prepare_4d_causal_attention_mask_with_cache_position
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
    CausalLMOutputWithPast,
)
import time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle
import copy

# results_20241127_091700: mean of weights in llm head, features combined
# results_20241127_125156: sum of weights in llm head, features combined
# results_20241127_170327: mean of weights in llm, features combined
# results_20241127_170327: mean of weights in vlm, sum of weights in llm head, features combined

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    HQQQuantizedCache,
    HybridCache,
    MambaCache,
    OffloadedStaticCache,
    QuantoQuantizedCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import (
    is_accelerate_available,
    is_torchdynamo_compiling,
    logging,
)
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)
from peft import prepare_model_for_kbit_training

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
    "offloaded_static": OffloadedStaticCache,
    "sliding_window": SlidingWindowCache,
    "hybrid": HybridCache,
    "mamba": MambaCache,
}
QUANT_BACKEND_CLASSES_MAPPING = {"quanto": QuantoQuantizedCache, "HQQ": HQQQuantizedCache}

from combine import models
from combine.models.qwen2 import CombineQwen2ForCausalLM, CombineQwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

class Learnable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.,1.]))
    def forward(self,feature_vlm,feature_llm):
        x = self.weight[0] * feature_vlm + self.weight[1] * feature_llm
        return x



class Combination(torch.nn.Module, GenerationMixin):
    def __init__(self, llm_name, vlm_name,merge_head_weights,merge_layer_weights_layers,merge_layer_weights_weights,
                 get_all_vlm_features_first,merge_feature_layers,feature_weights,mode,apply_on_entire_state,
                 sum_weight_feature,local_rank,use_same_index,get_all_llm_features_first,sample=False,learn_feature_weights=False):
        super().__init__()
        self.mode = mode
        
        from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
        
        #from combine import models
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        # )

        # bitsandbytes config
        USE_NESTED_QUANT = True  # use_nested_quant
        BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # bnb_4bit_compute_dtype
        load_in_8bit = True
        compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=compute_dtype,
            # bnb_4bit_use_double_quant=USE_NESTED_QUANT,
        )
        self.apply_on_entire_state = apply_on_entire_state
        self.sum_weight_feature = sum_weight_feature
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.devices = []  # Always include CPU
        if torch.cuda.is_available():
            self.devices.extend([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.sample = sample
        self.get_all_vlm_features_first = get_all_vlm_features_first
        self.get_all_llm_features_first = get_all_llm_features_first
        self.llm_name = llm_name
        self.vlm_name = vlm_name
        self.device_llm = "auto"
        self.device_vlm = "auto"
        self.merge_feature_layers = merge_feature_layers
        self.feature_weights = feature_weights
        self.use_same_index = use_same_index
        if(len(self.devices) == 2):
            self.device_llm = self.devices[0]
            self.device_vlm = self.devices[1]
        else:
            self.device_llm = self.devices[0]
            self.device_vlm = self.devices[0]
            # try:
        # device_llm = "cpu"
        # device_vlm = "cpu"
            # except:
        # models.CombineQwen2ForCausalLM
        self.llm = models.CombineQwen2ForCausalLM.from_pretrained(
            llm_name,            
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # load_in_4bit=load_in_8bit,
            # device_map=f"cuda:{local_rank}",
            # quantization_config=quantization_config
        )#.eval()#.to(self.device_llm)#.eval()#.to(device_1)
        # self.llm = prepare_model_for_kbit_training(self.llm)
        self.config = self.llm.config
        self.test = False
        self.tokenizer_llm = AutoTokenizer.from_pretrained(llm_name)
        # self.tokenizer_llm.padding_side  = 'left'
        if("lmms-lab" in self.vlm_name and self.test == False):
            # tokenizer, model, processor, max_length = load_pretrained_model(vlm_name, None, "llava_qwen", torch_dtype="bfloat16",device_map="auto",
            #                                                                         overwrite_config={})

            tokenizer, self.vlm, processor, max_length = models.load_pretrained_model(vlm_name, None, "llava_qwen", torch_dtype="bfloat16")#,device_map=self.device_vlm,
                                                                                #    quantization_config=quantization_config)
            # tokenizer, model, processor, max_length = load_pretrained_model(vlm_name, None, "llava_qwen", torch_dtype="bfloat16")  
            # model = model.to(torch.bfloat16)
            self.vlm_processor = processor
            self.tokenizer_vlm = tokenizer
            self.vlm.model.embed_tokens = self.vlm.model.embed_tokens.to(self.device_vlm)#.to(torch.bloat16)
            # self.vlm = model.eval()

        else:
            # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
            self.vlm = CombineQwen2VLForConditionalGeneration.from_pretrained(
                vlm_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                # load_in_4bit=load_in_8bit,
                # device_map=f"cuda:{local_rank}",
                # quantization_config=quantization_config
            )#.eval()#.to(self.device_vlm)#.eval()
            self.vlm_processor = AutoProcessor.from_pretrained(vlm_name)
            # self.vlm_processor.padding_side  = 'left'
            # self.vlm = prepare_model_for_kbit_training(self.vlm)
            self.vlm.model.embed_tokens = self.vlm.model.embed_tokens.to(self.device_vlm)

        # if(len(self.devices) == 2):
        #     if("vision_tower" in vars(self.vlm.model)):
        #         self.vlm.model.vision_tower = self.vlm.model.vision_tower#.to(self.vlm.model.device)
                
        #     elif("visual" in vars(self.vlm.model)):
        #         self.vlm.model.visual = self.vlm.model.visual#.to(self.vlm.model.device)
                


        # if(len(self.devices) == 2):
        #     self.vlm = self.vlm.to(self.devices[1])
        #     self.llm = self.llm.to(self.devices[0])

        # self.weight_llm = nn.Parameter(torch.tensor(0.5))
        # self.weight_vlm = nn.Parameter(torch.tensor(0.5))

        self.merge_head_weights = merge_head_weights
        self.merge_layer_weights_layers = merge_layer_weights_layers
        self.merge_layer_weights_weights = merge_layer_weights_weights
        # print(merge_head_weights,merge_layer_weights_layers,merge_layer_weights_weights)
        # merge_head_weights = None
        # merge_layer_weights_layers = []

        self.combined_head = copy.deepcopy(self.llm.lm_head)
        if(merge_head_weights is not None):
            merged_state_dict = {}
            for key in self.llm.lm_head.state_dict().keys(): 
                if(self.llm.lm_head.state_dict()[key].shape == self.vlm.lm_head.state_dict()[key].shape):
                    # print(self.llm.lm_head.state_dict()[key].shape, self.vlm.lm_head.state_dict()[key].shape)
                    merged_state_dict[key] = (merge_head_weights[1] * self.vlm.lm_head.state_dict()[key].to(self.llm.device) + merge_head_weights[0] * self.llm.lm_head.state_dict()[key].to(self.llm.device))# / 2
                    # merged_state_dict[key] = (merge_head_weights[0] * self.llm.lm_head.state_dict()[key] + merge_head_weights[1] * self.vlm.lm_head.state_dict()[key])# / 2
                else:
                    merged_state_dict[key] = self.llm.lm_head.state_dict()[key]
            # # Create a new model and load the merged weights
            self.combined_head.load_state_dict(merged_state_dict)
        if(len(merge_layer_weights_layers) > 0):
            import warnings

            # warnings.warn("Merging multiple layers weights is not correctly implemented, yet!")
            merged_state_dict = {}
            for i, key in enumerate(self.llm.state_dict().keys()):
                # if():
                #     merged_state_dict[key] = self.vlm.state_dict()[key]
                #     continue
                if(i in merge_layer_weights_layers and key in self.llm.state_dict() and "lm_head" not in key and 
                   self.llm.state_dict()[key].shape == self.vlm.state_dict()[key].shape):
                    # print(self.llm.state_dict()[key].shape, self.vlm.state_dict()[key].shape)
                    merged_state_dict[key] = (merge_layer_weights_weights[0] * self.llm.state_dict()[key] + merge_layer_weights_weights[1] * self.vlm.state_dict()[key])
                else:
                    merged_state_dict[key] = self.llm.state_dict()[key]
            # Create a new model and load the merged weights
            self.llm.load_state_dict(merged_state_dict)
        self.dummy = nn.Linear(in_features=2, out_features=1, bias=False)
        self.learn_feature_weights = learn_feature_weights
        if(learn_feature_weights):
            self.learnt_feature_weight = nn.Parameter(torch.tensor([1.0,1.0]))
        else:
            self.learnt_feature_weight = None

        self.model = {
            "llm": self.llm,
            "vlm": self.vlm,
            "learnt_feature_weight":self.learnt_feature_weight
        }
        # self.model = nn.DataParallel(self.model, device_ids = [ 0, 1]).cuda()

    def run(self,
                video_path=None,
                prompt_llm=None,
                prompt_vlm=None,
                merge_feature_layers=None,
                feature_weights=None,
                labels=None,
                llm_data=None,
                vlm_data=None,
                labels_ids=None,
                combined_head=None,
                input_ids=None,
                attention_mask=None,
                **kwargs):
    # def forward(self,llm_data,vlm_data,labels_ids,merge_feature_layers,feature_weights,combined_head):

        if(video_path is not None):
            
            # llm
            messages_llm = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt_llm}
            ]


            text_llm = self.tokenizer_llm.apply_chat_template(
                messages_llm,
                tokenize=False,
                add_generation_prompt=True
            )
            llm_data = self.tokenizer_llm([text_llm], return_tensors="pt").to(self.device_llm)
            


            #vlm
            if("lmms-lab" in self.vlm_name and self.test == False):
                video,frame_time,video_time = utils.load_video(video_path, 16, 1, force_sample=True)
                video = self.vlm_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
                video = [video]
                
                conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + prompt_vlm
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                vlm_input_ids = tokenizer_image_token(prompt_question, self.tokenizer_vlm, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.vlm.device)

                # print(input_ids)
                # print(video[0].shape)

                

                # cont = self.vlm.generate(
                #     input_ids,
                #     images=video,
                #     modalities= ["video"],
                #     do_sample=False,
                #     temperature=0,
                #     max_new_tokens=4096,
                # )

                vlm_data = {"input_ids":vlm_input_ids,
                            "do_sample":False,
                    #    "attention_mask":torch.ones(input_ids.shape).to(torch.bfloat16),
                    #    "images":[vid.to(torch.bfloat16) for vid in video],
                    "images":video,#[vid for vid in video],
                    "modalities":["video"]}
            else:
                messages_vlm = [
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
                ]
                text_vlm = self.vlm_processor.apply_chat_template(
                    messages_vlm, tokenize=False, add_generation_prompt=True#,chat_template="qwen_1_5"
                )
                image_inputs, video_inputs = process_vision_info(messages_vlm)
                
                vlm_data = self.vlm_processor(
                    text=[text_vlm],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device_vlm)
                # print(vlm_data.keys())
                # print(vlm_data["input_ids"])
                # print(vlm_data["input_ids"].shape)
                # print("C"+5)
                # print(len(video_inputs))
                # print(vlm_data["attention_mask"].shape)
                # print(vlm_data["video_grid_thw"])
                # print("C"+5)
        
        # if("lmms-lab/llava-onevision-qwen2-7b-ov" in self.vlm_name):
        #     vlm_data["images"] = [vlm_data["images"][0].to(self.vlm.model.device)]
        #     vlm_data["input_ids"] = vlm_data["input_ids"].to(self.vlm.model.device)

            
        # llm_data = llm_data.to(self.llm.model.device)
        max_new_tokens = 75
        # vlm_data["max_new_tokens"] = llm_data["max_new_tokens"]
        return_dict = True
        output_hidden_states = True

        # with torch.no_grad():
            # generated_ids = self.vlm.generate(**vlm_data)
        generated_ids = self.generate_(kwargs_llm=llm_data,
                                          kwargs_vlm=vlm_data,
                                          return_dict=return_dict,
                                          output_hidden_states=output_hidden_states,
                                          max_new_tokens=max_new_tokens,
                                          merge_feature_layers=self.merge_feature_layers,
                                          feature_weights=self.feature_weights,
                                          combined_head=self.combined_head,
                                          labels=labels_ids)
            # generated_ids = self.generate(kwargs_llm=llm_data,
            #                               kwargs_vlm=vlm_data,
            #                               return_dict=return_dict,
            #                               output_hidden_states=output_hidden_states,
            #                               merge_feature_layers=merge_feature_layers,
            #                               feature_weights=feature_weights,
            #                               combined_head=combined_head,
            #                               labels=labels_ids)
        
        # print("C"+5)
        # print(generated_ids)
        # print("C"+5)
        # generated_ids_llm = self.llm.generate(
        #     **model_inputs_llm,
        #     max_new_tokens=400
        # )
        # generated_ids = [
        #     output_ids[len(input_ids_vlm):] for input_ids_vlm, output_ids in zip(vlm_input_ids, generated_ids)
        # ]
        # print(generated_ids)
        # generated_ids = [
        # output_ids[len(input_ids):] for input_ids, output_ids in zip(vlm_data.input_ids, generated_ids)
        # ]
        
        response_llm = ["".join(self.tokenizer_llm.batch_decode(generated_id, skip_special_tokens=True)) for generated_id in generated_ids]
        # for r in response_llm:
        #     print(r)
        # print(response_llm)
        # print(len(generated_ids[0]),len(vlm_input_ids))
        # response_vlm = self.tokenizer_vlm.batch_decode(generated_ids, skip_special_tokens=True)#[0]
        # output_text = self.vlm_processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # print(response_llm)
        # return response_vlm
        return response_llm[0]


        # Inference: Generation of the output
        generated_ids_vlm = self.vlm.generate(**inputs_vlm, max_new_tokens=1280)


        generated_ids_trimmed_vlm = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_vlm.input_ids, generated_ids_vlm)
        ]
        output_text_vlm = self.vlm_processor.batch_decode(
            generated_ids_trimmed_vlm, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
    def forward(
        self,
        model_inputs_llm,
        model_inputs_vlm,
        return_dict,
        labels = None,
        output_hidden_states=None,
        merge_feature_layers=None,
        feature_weights=None,
        combined_head=None,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        scene_id=None,
        task_id=None,
        video_path=None,
        prompt_llm=None,
        prompt_vlm=None,
        
        # input_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # images: Optional[torch.FloatTensor] = None,
        # image_sizes: Optional[List[List[int]]] = None,
        # return_dict: Optional[bool] = None,
        # modalities: Optional[List[str]] = ["image"],
        # dpo_forward: Optional[bool] = False,
        # cache_position=None,

    ):
        # start = time.time()
        input_ids_llm, attention_mask_llm, position_ids_llm, past_key_values_llm, inputs_embeds_llm, use_cache, output_attentions_llm, output_hidden_states_llm, pixel_values_llm, pixel_values_videos_llm, output_attentions_llm = [None] * 11
        input_ids_vlm, attention_mask_vlm, position_ids_vlm, past_key_values_vlm, inputs_embeds_vlm, output_attentions_vlm, output_hidden_states_vlm, pixel_values_vlm, pixel_values_videos_vlm, output_attentions_vlm , images, modalities= [None] * 12
        pixel_values_vlm, image_grid_thw_vlm = [None] * 2
        # print(input_ids_llm,attention_mask_llm)
        
        if("input_ids" in model_inputs_llm):
            input_ids_llm = model_inputs_llm["input_ids"]
        if("labels" in model_inputs_llm):
            labels = model_inputs_llm["labels"]
        if("past_key_values" in model_inputs_llm):
            past_key_values_llm = model_inputs_llm["past_key_values"]
        if("attention_mask" in model_inputs_llm):
            attention_mask_llm = model_inputs_llm["attention_mask"]
        if("inputs_embeds" in model_inputs_llm):
            inputs_embeds_llm = model_inputs_llm["inputs_embeds"]
        if("cache_position" in model_inputs_llm):
            cache_position_llm = model_inputs_llm["cache_position"]
        else:
            cache_position_llm = None
        if("position_ids" in model_inputs_llm):
            position_ids_llm = model_inputs_llm["position_ids"]
        if("use_cache" in model_inputs_llm):
            use_cache = model_inputs_llm["use_cache"]
        if("num_logits_to_keep" in model_inputs_llm):
            num_logits_to_keep_llm = model_inputs_llm["num_logits_to_keep"]
        if("pixel_values_videos" in model_inputs_llm):
            pixel_values_videos_llm = model_inputs_llm["pixel_values_videos"]
        if("video_grid_thw" in model_inputs_llm):
            video_grid_thw_llm = model_inputs_llm["video_grid_thw"]
        if("rope_deltas" in model_inputs_llm):
            rope_deltas_llm = model_inputs_llm["rope_deltas"]

        if("input_ids" in model_inputs_vlm):
            input_ids_vlm = model_inputs_vlm["input_ids"]
        if("past_key_values" in model_inputs_vlm):
            past_key_values_vlm = model_inputs_vlm["past_key_values"]
        if("attention_mask" in model_inputs_vlm):
            attention_mask_vlm = model_inputs_vlm["attention_mask"]
        if("inputs_embeds" in model_inputs_vlm):
            inputs_embeds_vlm = model_inputs_vlm["inputs_embeds"]
        if("cache_position" in model_inputs_vlm):
            cache_position_vlm = model_inputs_vlm["cache_position"]
        else:
            cache_position_vlm = None
        if("position_ids" in model_inputs_vlm):
            position_ids_vlm = model_inputs_vlm["position_ids"]
        if("num_logits_to_keep" in model_inputs_vlm):
            num_logits_to_keep_vlm = model_inputs_vlm["num_logits_to_keep"]
        if("pixel_values_videos" in model_inputs_vlm):
            pixel_values_videos_vlm = model_inputs_vlm["pixel_values_videos"]
        if("video_grid_thw" in model_inputs_vlm):
            video_grid_thw_vlm = model_inputs_vlm["video_grid_thw"]
        if("rope_deltas" in model_inputs_vlm):
            rope_deltas_vlm = model_inputs_vlm["rope_deltas"]
        else:
            rope_deltas_vlm = None
        if("images" in model_inputs_vlm):
            images = model_inputs_vlm["images"]
        if("modalities" in model_inputs_vlm):
            modalities = model_inputs_vlm["modalities"]
        # print("sel",time.time()-start)
        # print(model_inputs_vlm)
        # print(input_ids_llm.shape)
        # print(input_ids_vlm.shape)

        # output_attentions_llm = output_attentions_llm if output_attentions_vlm is not None else self.llm.config.output_attentions
        # output_attentions_vlm = output_attentions_vlm if output_attentions_vlm is not None else self.vlm.config.output_attentions
        # output_attentions_llm = output_attentions if output_attentions is not None else self.llm.config.output_attentions
        # output_hidden_states_llm = (
        #     output_hidden_states_llm if output_hidden_states_llm is not None else self.llm.config.output_hidden_states
        # )
        # output_hidden_states_vlm = (
        #     output_hidden_states_vlm if output_hidden_states_vlm is not None else self.vlm.config.output_hidden_states
        # )
        return_dict = return_dict if return_dict is not None else self.vlm.config.use_return_dict

        # print(input_ids_vlm)
        # print(input_ids_vlm,torch.min(input_ids_vlm),torch.argmin(input_ids_vlm),torch.max(input_ids_vlm),torch.argmax(input_ids_vlm))
        # print(self.vlm.model.embed_tokens)
        # print("C"+5)
        # print(input_ids_vlm, position_ids_vlm, attention_mask_vlm, past_key_values_vlm, labels_vlm, images, modalities, None)
        # print("C"+5)
        # print(inputs_embeds_vlm)
        
        # print(input_ids_llm,attention_mask_llm)
        # print("aft")
        output_hidden_states_vlm = True
        if inputs_embeds_vlm is None:
            # print(self.vlm_name)
            if("lmms-lab" not in self.vlm_name):
                inputs_embeds_vlm = self.vlm.model.embed_tokens(input_ids_vlm)#.to(self.vlm.model.embed_tokens.weight.device))
                
                if pixel_values_vlm is not None:
                    pixel_values_vlm = pixel_values_vlm.type(self.vlm.visual.get_dtype())
                    image_embeds_vlm = self.vlm.visual(pixel_values_vlm, grid_thw=image_grid_thw_vlm)#.to(inputs_embeds_vlm.device)
                    image_mask_vlm = input_ids_vlm == self.vlm.config.image_token_id
                    if self.vlm.training:
                        inputs_embeds_vlm = inputs_embeds_vlm.clone()
                    inputs_embeds_vlm[image_mask_vlm] = image_embeds_vlm
                if pixel_values_videos_vlm is not None:
                    pixel_values_videos_vlm = pixel_values_videos_vlm.type(self.vlm.visual.get_dtype())
                    video_embeds_vlm = self.vlm.visual(pixel_values_videos_vlm, grid_thw=video_grid_thw_vlm)#.to(inputs_embeds_vlm.device)
                    video_mask_vlm = input_ids_vlm == self.vlm.config.video_token_id
                    inputs_embeds_vlm = inputs_embeds_vlm.clone()
                    inputs_embeds_vlm[video_mask_vlm] = video_embeds_vlm
                    # inputs_embeds_vlm[video_mask_vlm] = video_embeds_vlm
                if attention_mask_vlm is not None:
                    attention_mask_vlm = attention_mask_vlm#.to(inputs_embeds_vlm.device)
                input_ids_vlm = None
        # print(attention_mask_llm)
        # print(input_ids_llm)
        # print("feat",time.time()-start)
        # if cache_position_vlm is None:
        #     past_seen_tokens = past_key_values_vlm.get_seq_length() if past_key_values_vlm is not None else 0
        #     cache_position_vlm = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds_vlm.shape[1], device=inputs_embeds_vlm.device
        #     )
        # if cache_position_llm is None:
        #     past_seen_tokens = past_key_values_vlm.get_seq_length() if past_key_values_llm is not None else 0
        #     cache_position_llm = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds_llm.shape[1], device=inputs_embeds_llm.device
        #     )
        # print("start")
        # print(torch.sum(inputs_embeds_vlm))
        # print(torch.sum(attention_mask_vlm))
        # print(torch.sum(position_ids_vlm))
        # try:
        #     print(sum([torch.sum(key) for key in past_key_values_vlm.key_cache]))
        #     print(sum([torch.sum(value) for value in past_key_values_vlm.value_cache]))
        # except:
        #     pass
        # past_key_values_vlm=torch.load("past_key_values_vlm.pth",weights_only=False)
        # if(torch.sum(attention_mask_vlm) == 756):
        #     position_ids_vlm=torch.load("position_ids_vlm.pth",weights_only=False)
        #     attention_mask_vlm=torch.load("attention_mask_vlm.pth",weights_only=False)
        #     past_key_values_vlm=torch.load("past_key_values_vlm.pth",weights_only=False)
        #     inputs_embeds_vlm=torch.load("inputs_embeds_vlm.pth",weights_only=False)
        #     output_attentions_vlm=torch.load("output_attentions_vlm.pth",weights_only=False)
        #     output_hidden_states_vlm=torch.load("output_hidden_states.pth",weights_only=False)
        #     return_dict=torch.load("return_dict.pth",weights_only=False)
        #     use_cache=torch.load("use_cache.pth",weights_only=False)
        # print("loaded")
        # print(torch.sum(inputs_embeds_vlm))
        # print(torch.sum(attention_mask_vlm))
        # print(torch.sum(position_ids_vlm))
        # try:
        #     print(sum([torch.sum(key) for key in past_key_values_vlm.key_cache]))
        #     print(sum([torch.sum(value) for value in past_key_values_vlm.value_cache]))
        # except:
        #     pass
        # if torch.cuda.is_available():
        #     print(f"CUDA is available. Using GPU: 0, Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2} MB, Current GPU memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2} MB")
        #     print(f"CUDA is available. Using GPU: 1, Current GPU memory allocated: {torch.cuda.memory_allocated(1) / 1024 ** 2} MB, Current GPU memory cached: {torch.cuda.memory_reserved(1) / 1024 ** 2} MB")
        # else:
        #     print("CUDA is not available. Using CPU.")
        # print(input_ids_vlm)  
        # print(inputs_embeds_vlm) 
        (
            all_hidden_states_vlm, 
            all_self_attns_vlm, 
            next_decoder_cache_vlm, 
            output_hidden_states_vlm, 
            hidden_states_vlm, 
            attention_mask_vlm, 
            position_ids_vlm,
            past_key_values_vlm,
            output_attentions_vlm,
            use_cache,return_dict_vlm,
            use_legacy_cache_vlm,
            cache_position_vlm
        ) = self.vlm.prepare_layerwise_feature_extraction(
            input_ids=input_ids_vlm,#.to(self.device_vlm), 
            attention_mask=attention_mask_vlm, 
            position_ids=position_ids_vlm, 
            past_key_values=past_key_values_vlm, 
            inputs_embeds=inputs_embeds_vlm, 
            use_cache=use_cache,
            output_attentions=output_attentions_vlm, 
            output_hidden_states=output_hidden_states_vlm,
            return_dict=return_dict, 
            images=images, 
            modalities=modalities,
            pixel_values_vlm=pixel_values_vlm, 
            pixel_values_videos_vlm=pixel_values_videos_vlm, 
            labels=labels, 
            cache_position_vlm=cache_position_vlm
        )
        # print(input_ids_llm)  
        # print(inputs_embeds_llm) 
        # print("start",time.time()-start)
        output_hidden_states_llm = True
        (
            all_hidden_states_llm, 
            all_self_attns_llm, 
            next_decoder_cache_llm, 
            output_hidden_states_llm, 
            hidden_states_llm, 
            attention_mask_llm, 
            position_ids_llm,
            past_key_values_llm,
            output_attentions_llm,
            use_cache,return_dict_llm,
            use_legacy_cache_llm,
            cache_position_llm
        ) = self.llm.prepare_layerwise_feature_extraction(
            input_ids_llm,#.to(self.device_llm), 
            attention_mask_llm, 
            position_ids_llm, 
            past_key_values_llm, 
            inputs_embeds_llm, 
            use_cache,
            output_attentions_llm, 
            output_hidden_states_llm,
            return_dict, 
            images, 
            modalities,
            pixel_values_llm, 
            pixel_values_videos_llm, 
            labels, 
        )
        # print("start",time.time()-start)
        hidden_states_combined = None
        # print(torch.sum(hidden_states_vlm))
        # print(torch.sum(attention_mask_vlm))
        # print(torch.sum(position_ids_vlm))
        # try:
        #     print(sum([torch.sum(key) for key in past_key_values_vlm.key_cache]))
        #     print(sum([torch.sum(value) for value in past_key_values_vlm.value_cache]))
        # except:
        #     pass
        # print(torch.sum(cache_position_vlm))
        # print(hidden_states_llm.shape)  
        # print(hidden_states_vlm.shape) 
            

        if(self.get_all_vlm_features_first):
            if output_hidden_states_vlm:
                all_hidden_states_vlm += (hidden_states_vlm,)
            for layer_idx, layer in enumerate(self.vlm.model.layers):
            
                # if(hidden_states_combined is not None):
                #     hidden_states_llm = hidden_states_combined#.to(torch.bfloat16)
                # print(hidden_states_vlm.shape)
                # print(attention_mask_vlm.shape)
                (
                    hidden_states_vlm, use_cache, all_hidden_states_vlm,next_decoder_cache_vlm
                ) = self.vlm.get_layerwise_features(
                    all_hidden_states_vlm, all_self_attns_vlm, 
                    output_hidden_states_vlm,hidden_states_vlm,#.to(self.vlm.model.device), 
                    attention_mask_vlm, position_ids_vlm,
                    past_key_values_vlm,output_attentions_vlm,
                    use_cache,cache_position_vlm,layer_idx
                ) 
                # print(torch.sum(hidden_states_vlm))
            # print(hidden_states_llm.shape)
            # print(attention_mask_llm.shape)
            # print("end_vlm",time.time()-start)
            
            
            if output_hidden_states_llm:
                all_hidden_states_llm += (hidden_states_llm,)
            for layer_idx, layer in enumerate(self.llm.model.layers):
                (
                    hidden_states_llm,use_cache, all_hidden_states_llm, next_decoder_cache_llm
                )=self.llm.get_layerwise_features(
                    all_hidden_states_llm, all_self_attns_llm, 
                    output_hidden_states_llm,hidden_states_llm,#.to(self.llm.model.device), 
                    attention_mask_llm, position_ids_llm,
                    past_key_values_llm,output_attentions_llm,
                    use_cache,cache_position_llm,layer_idx
                )  
                # print("begin_llm",time.time()-start)
                # print(layer_idx,merge_feature_layers)
                
                if(self.use_same_index):
                    used_index = layer_idx+1
                else:
                    used_index = layer_idx
                if(layer_idx in self.merge_feature_layers):
                    if(self.apply_on_entire_state):
                        if(self.sum_weight_feature):
                            hidden_states_llm = all_hidden_states_vlm[used_index][:,-1:,:] + \
                                hidden_states_llm
                        else:
                            # print(hidden_states_llm.requires_grad)  # Should be True
                            # print(all_hidden_states_vlm[layer_idx].requires_grad)  # Should be True
                            # self.learnt_feature_weight
                            if(self.learnt_feature_weight is not None):
                                hidden_states_llm = self.learnt_feature_weight[1] * all_hidden_states_vlm[used_index][:,-1:,:] + \
                                    self.learnt_feature_weight[0] * hidden_states_llm
                            else:
                                hidden_states_llm = self.feature_weights[1] * all_hidden_states_vlm[used_index][:,-1:,:] + \
                                    self.feature_weights[0] * hidden_states_llm
                                # print(hidden_states_llm)
                                # print("C"+5)

                                
                            # print(hidden_states_llm.requires_grad)  # Should be True
                                
                    else:
                        if(self.sum_weight_feature):
                            # print(len(all_hidden_states_vlm))
                            # print(hidden_states_llm[:,-1,:].shape,all_hidden_states_vlm[used_index][:,-1,:].shape)
                            hidden_states_llm[:,-1,:] = all_hidden_states_vlm[used_index][:,-1,:] + \
                                hidden_states_llm[:,-1,:]
                        else:
                            # print(len(all_hidden_states_vlm))
                            # print(hidden_states_llm[:,-1,:].shape,all_hidden_states_vlm[used_index][:,-1,:].shape)
                            hidden_states_llm[:,-1,:] = self.feature_weights[1] * all_hidden_states_vlm[used_index][:,-1,:] + \
                                self.feature_weights[0] * hidden_states_llm[:,-1,:]
                    # hidden_states_llm = all_hidden_states_vlm[layer_idx][:,-1,:] + hidden_states_llm
            
                    # print("mid_llm",time.time()-start)
                    
                # print("end_llm",time.time()-start)
                # if(layer_idx in merge_feature_layers):

                #     hidden_states_vlm[:,-1,:] = feature_weights[0] * hidden_states_vlm[:,-1,:].to(self.vlm.model.device) + \
                #             feature_weights[1] * hidden_states_llm[:,-1,:].to(self.vlm.model.device)
                #     hidden_states_llm[:,-1,:] = feature_weights[0] * hidden_states_vlm[:,-1,:].to(self.llm.model.device) + \
                #             feature_weights[1] * hidden_states_llm[:,-1,:].to(self.llm.model.device)
        elif(self.get_all_llm_features_first):
            if output_hidden_states_llm:
                all_hidden_states_llm += (hidden_states_llm,)
            for layer_idx, layer in enumerate(self.llm.model.layers):
            
                # if(hidden_states_combined is not None):
                #     hidden_states_llm = hidden_states_combined#.to(torch.bfloat16)
                # print(hidden_states_vlm.shape)
                # print(attention_mask_vlm.shape)
                (
                    hidden_states_llm, use_cache, all_hidden_states_llm,next_decoder_cache_llm
                ) = self.llm.get_layerwise_features(
                    all_hidden_states_llm, all_self_attns_llm, 
                    output_hidden_states_llm,hidden_states_llm,#.to(self.vlm.model.device), 
                    attention_mask_llm, position_ids_llm,
                    past_key_values_llm,output_attentions_llm,
                    use_cache,cache_position_llm,layer_idx
                ) 
                # print(torch.sum(hidden_states_vlm))
            # print(hidden_states_llm.shape)
            # print(attention_mask_llm.shape)
            # print("end_vlm",time.time()-start)
            
            
            if output_hidden_states_vlm:
                all_hidden_states_vlm += (hidden_states_vlm,)
            for layer_idx, layer in enumerate(self.vlm.model.layers):
                (
                    hidden_states_vlm,use_cache, all_hidden_states_vlm, next_decoder_cache_vlm
                )=self.vlm.get_layerwise_features(
                    all_hidden_states_vlm, all_self_attns_vlm, 
                    output_hidden_states_vlm,hidden_states_vlm,#.to(self.llm.model.device), 
                    attention_mask_vlm, position_ids_vlm,
                    past_key_values_vlm,output_attentions_vlm,
                    use_cache,cache_position_vlm,layer_idx
                )  
                # print("begin_llm",time.time()-start)
                # print(layer_idx,merge_feature_layers)
                
                if(self.use_same_index):
                    used_index = layer_idx+1
                else:
                    used_index = layer_idx
                if(layer_idx in self.merge_feature_layers):
                    if(self.apply_on_entire_state):
                        if(self.sum_weight_feature):
                            hidden_states_vlm = all_hidden_states_llm[used_index][:,-1:,:] + \
                                hidden_states_vlm
                        else:
                            # print(hidden_states_llm.requires_grad)  # Should be True
                            # print(all_hidden_states_vlm[layer_idx].requires_grad)  # Should be True
                            # self.learnt_feature_weight
                            if(self.learnt_feature_weight is not None):
                                hidden_states_vlm = self.learnt_feature_weight[0] * all_hidden_states_llm[used_index][:,-1:,:] + \
                                    self.learnt_feature_weight[1] * hidden_states_vlm
                            else:
                                hidden_states_vlm = self.feature_weights[0] * all_hidden_states_llm[used_index][:,-1:,:] + \
                                    self.feature_weights[1] * hidden_states_vlm

                                
                            # print(hidden_states_llm.requires_grad)  # Should be True
                                
                    else:
                        if(self.sum_weight_feature):
                            # print(len(all_hidden_states_vlm))
                            # print(hidden_states_llm[:,-1,:].shape,all_hidden_states_vlm[used_index][:,-1,:].shape)
                            hidden_states_vlm[:,-1,:] = all_hidden_states_llm[used_index][:,-1,:] + \
                                hidden_states_vlm[:,-1,:]
                        else:
                            # print(len(all_hidden_states_vlm))
                            # print(hidden_states_llm[:,-1,:].shape,all_hidden_states_vlm[used_index][:,-1,:].shape)
                            hidden_states_vlm[:,-1,:] = self.feature_weights[0] * all_hidden_states_llm[used_index][:,-1,:] + \
                                self.feature_weights[1] * hidden_states_vlm[:,-1,:]
        else:
            # print(len(self.llm.model.layers),len(self.vlm.model.layers))
            for layer_idx, layer in enumerate(self.llm.model.layers):
                
                # if(hidden_states_combined is not None):
                #     hidden_states_llm = hidden_states_combined#.to(torch.bfloat16)
                (
                    hidden_states_vlm, use_cache, all_hidden_states_vlm,next_decoder_cache_vlm
                ) = self.vlm.get_layerwise_features(
                    all_hidden_states_vlm, all_self_attns_vlm, 
                    output_hidden_states_vlm,hidden_states_vlm,#.to(self.vlm.model.device), 
                    attention_mask_vlm, position_ids_vlm,
                    past_key_values_vlm,output_attentions_vlm,
                    use_cache,cache_position_vlm,layer_idx
                )    
                (
                    hidden_states_llm,use_cache, all_hidden_states_llm, next_decoder_cache_llm
                )=self.llm.get_layerwise_features(
                    all_hidden_states_llm, all_self_attns_llm, 
                    output_hidden_states_llm,hidden_states_llm,#.to(self.llm.model.device), 
                    attention_mask_llm, position_ids_llm,
                    past_key_values_llm,output_attentions_llm,
                    use_cache,cache_position_llm,layer_idx
                )

                if(layer_idx in self.merge_feature_layers):
                    prev_hidden_states_llm = hidden_states_llm.clone()
                    if(self.apply_on_entire_state):
                        if(self.sum_weight_feature):
                            hidden_states_llm = all_hidden_states_vlm[layer_idx][:,-1:,:] + \
                                hidden_states_llm
                            hidden_states_vlm = all_hidden_states_vlm[layer_idx] + \
                                prev_hidden_states_llm[:,-1:,:]
                        else:
                            hidden_states_llm = self.feature_weights[1] * all_hidden_states_vlm[layer_idx][:,-1:,:] + \
                                self.feature_weights[0] * hidden_states_llm
                            hidden_states_vlm = self.feature_weights[1] * all_hidden_states_vlm[layer_idx] + \
                                self.feature_weights[0] * prev_hidden_states_llm[:,-1:,:]
                    else:
                        if(self.sum_weight_feature):
                            hidden_states_llm[:,-1,:] = hidden_states_vlm[:,-1,:] + \
                                    hidden_states_llm[:,-1,:]
                            hidden_states_vlm[:,-1,:] = hidden_states_vlm[:,-1,:] + \
                                    prev_hidden_states_llm[:,-1,:] # + \
                        else:
                            hidden_states_llm[:,-1,:] = self.feature_weights[1] * hidden_states_vlm[:,-1,:].to(self.llm.model.device) + \
                                    self.feature_weights[0] * hidden_states_llm[:,-1,:].to(self.llm.model.device)#
                            hidden_states_vlm[:,-1,:] = self.feature_weights[1] * hidden_states_vlm[:,-1,:].to(self.vlm.model.device) + \
                                    self.feature_weights[0] * prev_hidden_states_llm[:,-1,:].to(self.vlm.model.device) # + \
                    # hidden_states_vlm[:,-1,:] = feature_weights[0] * hidden_states_vlm[:,-1,:] + \
                    #         feature_weights[1] * hidden_states_llm[:,-1,:]#.to(self.vlm.model.device) #.to(self.vlm.model.device) + \
                    # hidden_states_llm[:,-1,:] = feature_weights[0] * hidden_states_vlm[:,-1,:] + \
                            # feature_weights[1] * hidden_states_llm[:,-1,:]#.to(self.llm.model.device)#.to(self.llm.model.device)
                    # if(keep_vlm == True):
                    #     hidden_states_combined = feature_weights[0] * hidden_states_vlm + \
                    #         feature_weights[1] * hidden_states_llm[:,-1,:]
                    # else:
                    #     hidden_states_combined = feature_weights[0] * hidden_states_vlm[:,-1,:] + \
                    #         feature_weights[1] * hidden_states_llm
                        
                # print(hidden_states_vlm.shape,hidden_states_llm.shape)
                # hidden_states_combined = hidden_states_vlm # [:,-1,:] + hidden_states_llm

        # print("C"+5)
        # summ_v = [torch.sum(h_state) for h_state in all_hidden_states_vlm]
        # summ_l = [torch.sum(h_state) for h_state in all_hidden_states_llm]
        hidden_states_llm = self.llm.model.norm(hidden_states_llm)#.to(self.llm.model.device))
        hidden_states_vlm = self.vlm.model.norm(hidden_states_vlm)#.to(self.vlm.model.device))#.to(torch.bfloat16)
        
        # print("norm",time.time()-start)
        if output_hidden_states_vlm:
            all_hidden_states_vlm += (hidden_states_vlm,)
        if output_hidden_states_llm:
            all_hidden_states_llm += (hidden_states_llm,)
        
        # print("hid",time.time()-start)
        # print("vlm_states")
        # for state in all_hidden_states_vlm:
        #     print(torch.sum(state))
        # print(torch.sum(hidden_states_vlm))
        # print("vlm_states_end")
        # print("llm_states")
        # for state in all_hidden_states_llm:
        #     print(torch.sum(state))
        # # print(torch.sum(hidden_states_llm))
        # print("llm_states_end")
        # summ_l = torch.sum(hidden_states_llm)
        # summ_v = torch.sum(hidden_states_vlm)
        # print(summ_l)
        # print(summ_v)
        # summ_v = torch.sum(hidden_states_llm)
        # print(summ_v)

        if(-1 in self.merge_feature_layers):
            # if(self.apply_on_entire_state):
            #     # #.to(torch.bfloat16)
            #     # hidden_states_llm = self.weight_vlm * hidden_states_vlm[:,-1:,:] + self.weight_llm * hidden_states_llm#.to(torch.bfloat16)
            #     if(self.learnt_feature_weight is not None):
            #         hidden_states_llm = self.learnt_feature_weight[1] * hidden_states_vlm[:,-1:,:] + \
            #                         self.learnt_feature_weight[0] * hidden_states_llm
            #     else:
            #         hidden_states_llm = self.feature_weights[1] * hidden_states_vlm[:,-1:,:] + self.feature_weights[0] * hidden_states_llm
            # else:
            hidden_states_llm[:,-1,:] = self.feature_weights[1] * hidden_states_vlm[:,-1,:] + self.feature_weights[0] * hidden_states_llm[:,-1,:]#.to(torch.bfloat16)
        # hidden_states[:,-1,:] = (feature_weights[0] * self.llm.model.norm(hidden_states_llm)[:,-1,:] + feature_weights[1] * hidden_states_final_vlm[-1][:,-1,:].bfloat16())#/sum(feature_weights)
        # print("end",time.time()-start)
        # print(torch.sum(hidden_states_llm))
        # hidden_states_final_vlm = outputs_vlm["hidden_states"]
        # logits_vlm = self.vlm.lm_head(hidden_states_final_vlm[-1][:, -1:, :])
        # logits_vlm = logits_vlm.float()

        # outputs_llm = self.forward__(
        #     input_ids_llm=input_ids_llm,
        #     position_ids_llm=position_ids_llm,
        #     attention_mask_llm=attention_mask_llm,
        #     past_key_values_llm=past_key_values_llm,
        #     inputs_embeds_llm=inputs_embeds_llm,
        #     output_attentions_llm=output_attentions_llm,
        #     cache_position_llm=cache_position_llm,
        #     input_ids_vlm=input_ids_vlm,
        #     position_ids_vlm=position_ids_vlm,
        #     attention_mask_vlm=attention_mask_vlm,
        #     past_key_values_vlm=past_key_values_vlm,
        #     inputs_embeds_vlm=inputs_embeds_vlm,
        #     output_attentions_vlm=output_attentions_vlm,
        #     hidden_states_final_vlm=hidden_states_final_vlm,
        #     return_dict=return_dict,
        #     use_cache=use_cache,
        #     output_hidden_states=output_hidden_states,
        #     merge_feature_layers=merge_feature_layers,
        #     feature_weights=feature_weights
        # )

        # hidden_states = outputs_llm[0]
        # logits_llm = self.llm.lm_head(hidden_states)
        # logits_llm = logits_llm.float()

        # loss = None

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # print("logits")
        
        if(self.mode == "train"):
            outputs = self.llm.return_token_scores(all_hidden_states_llm, all_self_attns_llm, next_decoder_cache_llm, 
                                        hidden_states_llm,labels,use_cache,return_dict,use_legacy_cache_llm,custom_head=self.combined_head)
            loss = outputs.loss
            # print(loss)
            # print(loss)
            return outputs
            # return[
            # self.llm.return_token_scores(all_hidden_states_llm, all_self_attns_llm, next_decoder_cache_llm, 
            #                             hidden_states_llm,labels,use_cache,return_dict,use_legacy_cache_llm,custom_head=self.combined_head),
            #                             None,
            #                             None
            # ]
        else:
            return[
            self.llm.return_token_scores(all_hidden_states_llm, all_self_attns_llm, next_decoder_cache_llm, 
                                        hidden_states_llm,labels,use_cache,return_dict,use_legacy_cache_llm,custom_head=self.combined_head),
                                        None,
                                     self.vlm.return_token_scores(all_hidden_states_vlm, all_self_attns_vlm, next_decoder_cache_vlm, 
                                     hidden_states_vlm,labels,use_cache,return_dict,use_legacy_cache_vlm,rope_deltas=rope_deltas_vlm)
            ]
        # self.llm.return_token_scores(all_hidden_states_llm, all_self_attns_llm, next_decoder_cache_llm, 
        #                              hidden_states_llm,labels,use_cache,return_dict,use_legacy_cache_llm),
        # self.vlm.return_token_scores(all_hidden_states_vlm, all_self_attns_vlm, next_decoder_cache_vlm, 
        #                              hidden_states_vlm,labels,use_cache,return_dict,use_legacy_cache_vlm,rope_deltas=rope_deltas_vlm)
        # ]
        
        vlm_output = None
        if(self.vlm_name ==  "lmms-lab/llava-onevision-qwen2-7b-ov"):
            vlm_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits_vlm,
            past_key_values=outputs_vlm.past_key_values,
            hidden_states=outputs_vlm.hidden_states,
            attentions=outputs_vlm.attentions,
        )
        else:
            vlm_output = Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits_vlm,
            past_key_values=outputs_vlm.past_key_values,
            hidden_states=outputs_vlm.hidden_states,
            attentions=outputs_vlm.attentions,
            rope_deltas=rope_deltas_vlm,
            )
        # print(vlm_output)
        # return [CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits_llm,
        #     past_key_values=outputs_llm.past_key_values,
        #     hidden_states=outputs_llm.hidden_states,
        #     attentions=outputs_llm.attentions,
        # ),
        # vlm_output]
    
    def forward__(
        self,
        input_ids_llm=None,
        position_ids_llm=None,
        attention_mask_llm=None,
        past_key_values_llm=None,
        inputs_embeds_llm=None,
        output_attentions_llm=None,
        cache_position_llm=None,
        input_ids_vlm=None,
        position_ids_vlm=None,
        attention_mask_vlm=None,
        past_key_values_vlm=None,
        inputs_embeds_vlm=None,
        output_attentions_vlm=None,
        cache_position_vlm = None,
        hidden_states_final_vlm=None,
        return_dict = None,
        use_cache=None,
        output_hidden_states=None,
        merge_feature_layers=None,
        feature_weights=None
    ):
        output_attentions_llm = output_attentions_llm if output_attentions_llm is not None else self.llm.config.output_attentions
        output_attentions_vlm = output_attentions_vlm if output_attentions_vlm is not None else self.vlm.config.output_attentions
        # output_hidden_states_llm = (
        #     output_hidden_states_llm if output_hidden_states_llm is not None else self.llm.model.config.output_hidden_states
        # )
        # output_hidden_states_vlm = (
        #     output_hidden_states_vlm if output_hidden_states_vlm is not None else self.vlm.model.config.output_hidden_states
        # )

        use_cache = use_cache if use_cache is not None else self.llm.model.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.llm.model.config.use_return_dict
        # return_dict_vlm = return_dict_vlm if return_dict_vlm is not None else self.vlm.config.use_return_dict

        if (input_ids_llm is None) ^ (inputs_embeds_llm is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if (input_ids_vlm is None) ^ (inputs_embeds_vlm is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if input_ids_vlm is not None and inputs_embeds_vlm is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids_vlm is not None:
            batch_size, seq_length = input_ids_vlm.shape
        elif inputs_embeds_vlm is not None:
            batch_size, seq_length, _ = inputs_embeds_vlm.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")


        if self.llm.model.gradient_checkpointing and self.llm.model.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        use_legacy_cache_llm = False
        if use_cache and not isinstance(past_key_values_llm, Cache) and not self.llm.model.training:
            use_legacy_cache_llm = True
            past_key_values_llm = DynamicCache.from_legacy_cache(past_key_values_llm)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
            )
        elif(use_cache):
            past_key_values_length = past_key_values_vlm.get_usable_length(seq_length)

        if inputs_embeds_llm is None:
            inputs_embeds_llm = self.llm.model.embed_tokens(input_ids_llm)
        if inputs_embeds_vlm is None:
            inputs_embeds_vlm = self.vlm.model.embed_tokens(input_ids_vlm)

        if cache_position_llm is None:
            past_seen_tokens_llm = past_key_values_llm.get_seq_length() if past_key_values_llm is not None else 0
            cache_position_llm = torch.arange(
                past_seen_tokens_llm, past_seen_tokens_llm + inputs_embeds_llm.shape[1], device=inputs_embeds_llm.device
            )
        if cache_position_vlm is None and "lmms-lab" not in self.vlm_name:
            past_seen_tokens_vlm = past_key_values_vlm.get_seq_length() if past_key_values_vlm is not None else 0
            cache_position_vlm = torch.arange(
                past_seen_tokens_vlm, past_seen_tokens_vlm + inputs_embeds_vlm.shape[1], device=inputs_embeds_vlm.device
            )
        if position_ids_llm is None:
            # the hard coded `3` is for temporal, height and width.
            position_ids_llm = cache_position_llm.view(1, 1, -1).expand(3, inputs_embeds_llm.shape[0], -1)
        if position_ids_vlm is None:
            # the hard coded `3` is for temporal, height and width.
            if("lmms-lab" not in self.vlm_name):
                position_ids_vlm = cache_position_vlm.view(1, 1, -1).expand(3, inputs_embeds_vlm.shape[0], -1)
            else:
                device = input_ids_vlm.device if input_ids_vlm is not None else inputs_embeds_vlm.device
                position_ids_vlm = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids_vlm = position_ids_vlm.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids_vlm = position_ids_vlm.view(-1, seq_length).long()
        causal_mask_llm = self.llm.model._update_causal_mask(
            attention_mask_llm, inputs_embeds_llm, cache_position_llm, past_key_values_llm, output_attentions_llm
        )
        causal_mask_vlm = self.vlm.model._update_causal_mask(
            attention_mask_vlm, inputs_embeds_vlm, cache_position_vlm, past_key_values_vlm, output_attentions_vlm
        )

        hidden_states_llm = inputs_embeds_llm# + hidden_states_final_vlm[:,-1,:]
        hidden_states_vlm = inputs_embeds_vlm

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        # all_hidden_states_vlm = () if output_hidden_states else None
        all_self_attns_llm = () if output_attentions_llm else None
        all_self_attns_vlm = () if output_attentions_vlm else None
        next_decoder_cache_llm = None
        next_decoder_cache_vlm = None

        # print(self.llm.model.layers)
        # print(self.vlm.model.layers)
        layer_count = 0
        
        for decoder_layer_llm, decoder_layer_vlm in zip(self.llm.model.layers,self.vlm.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states_llm,)
            # if output_hidden_states_vlm:
            #     all_hidden_states_vlm += (hidden_states_vlm,)

            if self.llm.model.gradient_checkpointing and self.llm.model.training:
                layer_outputs_llm = self.llm.model._gradient_checkpointing_func(
                    decoder_layer_llm.__call__,
                    hidden_states_llm,
                    causal_mask_llm,
                    position_ids_llm,
                    past_key_values_llm,
                    output_attentions_llm,
                    use_cache,
                    cache_position_llm,
                )
            else:
                layer_outputs_llm = decoder_layer_llm(
                    hidden_states_llm,
                    attention_mask=causal_mask_llm,
                    position_ids=position_ids_llm,
                    past_key_value=past_key_values_llm,
                    output_attentions=output_attentions_llm,
                    use_cache=use_cache,
                    cache_position=cache_position_llm,
                )
            # if self.vlm.model.gradient_checkpointing and self.vlm.model.training:
            #     layer_outputs_vlm = self.vlm.model._gradient_checkpointing_func(
            #         decoder_layer_vlm.__call__,
            #         hidden_states_vlm,
            #         causal_mask_vlm,
            #         position_ids_vlm,
            #         past_key_values_vlm,
            #         output_attentions_vlm,
            #         use_cache,
            #         cache_position_vlm,
            #     )
            # else:
            #     layer_outputs_vlm = decoder_layer_vlm(
            #         hidden_states_vlm,
            #         attention_mask=causal_mask_vlm,
            #         position_ids=position_ids_vlm,
            #         past_key_value=past_key_values_vlm,
            #         output_attentions=output_attentions_vlm,
            #         use_cache=use_cache,
            #         cache_position=cache_position_vlm,
            #     )

            # print("hidden_states_llm")

            hidden_states_llm = layer_outputs_llm[0]
            # hidden_states_vlm = layer_outputs_vlm[0]
            
            # hidden_states_llm = hidden_states_llm + hidden_states_final_vlm[layer_count][:,-1,:]
            if(layer_count in merge_feature_layers):
                # print(hidden_states_llm.dtype,hidden_states_final_vlm[layer_count][:,-1,:].dtype)

                hidden_states_llm = hidden_states_llm + hidden_states_final_vlm[layer_count][:,-1,:].bfloat16()
                # print(hidden_states_llm.dtype)
            # if(layer_count == 0):
            #     
            # hidden_states = hidden_states_llm + hidden_states_vlm

            if use_cache:
                next_decoder_cache_llm = layer_outputs_llm[2 if output_attentions_llm else 1]
                # next_decoder_cache_vlm = layer_outputs_vlm[2 if output_attentions_vlm else 1]

            if output_attentions_llm:
                all_self_attns_llm += (layer_outputs_llm[1],)
            # if output_attentions_vlm:
            #     all_self_attns_vlm += (layer_outputs_vlm[1],)
            # print(hidden_states_llm.shape)
            # print(layer_count)
            layer_count += 1
        # print(len(hidden_states_final_vlm),hidden_states_final_vlm[layer_count].shape)
        # print("C"+5)
        if(-1 in merge_feature_layers):
            # print(feature_weights,merge_feature_layers)
            hidden_states = (feature_weights[0] * self.llm.model.norm(hidden_states_llm) + feature_weights[1] * hidden_states_final_vlm[-1][:,-1,:].bfloat16())#/sum(feature_weights)
        else:
            hidden_states = self.llm.model.norm(hidden_states_llm)
        # hidden_states = self.llm.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            next_cache_llm = next_decoder_cache_llm.to_legacy_cache() if use_legacy_cache_llm else next_decoder_cache_llm
            
        next_cache_vlm = next_decoder_cache_vlm if use_cache else None
        # next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache_llm, all_hidden_states, all_self_attns_llm] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache_llm,
            hidden_states=all_hidden_states,
            attentions=all_self_attns_llm,
        )
    
    def prepare_inputs_for_generation_llm(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        kwargs=None,
    ):
        if("input_ids" in kwargs):
            input_ids = kwargs["input_ids"]
        if("past_key_values" in kwargs):
            past_key_values = kwargs["past_key_values"]
        if("attention_mask" in kwargs):
            attention_mask = kwargs["attention_mask"]
        if("inputs_embeds" in kwargs):
            inputs_embeds = kwargs["inputs_embeds"]
        if("cache_position" in kwargs):
            cache_position = kwargs["cache_position"]
        if("position_ids" in kwargs):
            position_ids = kwargs["position_ids"]
        if("use_cache" in kwargs):
            use_cache = kwargs["use_cache"]
        if("num_logits_to_keep" in kwargs):
            num_logits_to_keep = kwargs["num_logits_to_keep"]
        if("pixel_values_videos" in kwargs):
            pixel_values_videos = kwargs["pixel_values_videos"]
        if("video_grid_thw" in kwargs):
            video_grid_thw = kwargs["video_grid_thw"]
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_vlm_llava(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        kwargs = None,
    ):
        if("input_ids" in kwargs):
            input_ids = kwargs["input_ids"]
        if("past_key_values" in kwargs):
            past_key_values = kwargs["past_key_values"]
        if("attention_mask" in kwargs):
            attention_mask = kwargs["attention_mask"]
        if("inputs_embeds" in kwargs):
            inputs_embeds = kwargs["inputs_embeds"]
        if("cache_position" in kwargs):
            cache_position = kwargs["cache_position"]
        if("position_ids" in kwargs):
            position_ids = kwargs["position_ids"]
        if("use_cache" in kwargs):
            use_cache = kwargs["use_cache"]
        if("num_logits_to_keep" in kwargs):
            num_logits_to_keep = kwargs["num_logits_to_keep"]
        if("pixel_values_videos" in kwargs):
            pixel_values_videos = kwargs["pixel_values_videos"]
        if("video_grid_thw" in kwargs):
            video_grid_thw = kwargs["video_grid_thw"]
        if("images" in kwargs):            
            images = kwargs["images"]
        if("modalities" in kwargs):
            modalities = kwargs["modalities"]
        # print(kwargs)
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        if images is not None:
            model_inputs["images"] = images
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
        if modalities is not None:
            model_inputs["modalities"] = modalities
        return model_inputs
    
    def prepare_inputs_for_generation_vlm(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        kwargs=None,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if("input_ids" in kwargs):
            input_ids = kwargs["input_ids"]
        if("past_key_values" in kwargs):
            past_key_values = kwargs["past_key_values"]
        if("attention_mask" in kwargs):
            attention_mask = kwargs["attention_mask"]
        if("inputs_embeds" in kwargs):
            inputs_embeds = kwargs["inputs_embeds"]
        if("cache_position" in kwargs):
            cache_position = kwargs["cache_position"]
        if("position_ids" in kwargs):
            position_ids = kwargs["position_ids"]
        if("use_cache" in kwargs):
            use_cache = kwargs["use_cache"]
        if("num_logits_to_keep" in kwargs):
            num_logits_to_keep = kwargs["num_logits_to_keep"]
        if("pixel_values_videos" in kwargs):
            pixel_values_videos = kwargs["pixel_values_videos"]
        if("video_grid_thw" in kwargs):
            video_grid_thw = kwargs["video_grid_thw"]
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.vlm.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if inputs_embeds is not None:
                batch_size, sequence_length = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.vlm.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        model_kwargs=None,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, model_kwargs

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)


        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
    
    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        kwargs=None,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.llm.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.stop_strings is not None:
            if tokenizer is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))
        if generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
        criteria = self.llm._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def prepare_generation_params(self,model,generation_config=None,inputs=None, synced_gpus = None, logits_processor=None, stopping_criteria = None, assistant_model=None,streamer=None, prefix_allowed_tokens_fn=None, negative_prompt_ids=None, negative_prompt_attention_mask=None, kwargs=None):
        model._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        
        # print(model)
        
        generation_config, kwargs = model._prepare_generation_config(generation_config, **kwargs)

        model._validate_model_kwargs(kwargs.copy())
        model._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled():# and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in kwargs
        kwargs_has_attention_mask = kwargs.get("attention_mask", None) is not None

        # print(inputs)

        # print("here",inputs)
        # 3. Define model inputs
        inputs_tensor, model_input_name, kwargs = model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, kwargs
        )
        batch_size = inputs_tensor.shape[0]
        # print("out",inputs_tensor)

        device = inputs_tensor.device
        model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not model.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )


        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            kwargs["use_cache"] = True
        else:
            kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        if model.config.is_encoder_decoder and "encoder_outputs" not in kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if model.config.is_encoder_decoder:
            input_ids, kwargs = model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = model.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if model._supports_num_logits_to_keep() and "num_logits_to_keep" not in kwargs:
            kwargs["num_logits_to_keep"] = 1

        model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
        
        if("lmms-lab" not in self.vlm_name):
            cache_name = "past_key_values" if "mamba" not in model.__class__.__name__.lower() else "cache_params"
            user_defined_cache = kwargs.get(cache_name)
            model._prepare_cache_for_generation(generation_config, kwargs, assistant_model, batch_size, device)

        # print(kwargs)
        # print("C"+5)
        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and model.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

            
        # print(kwargs)
        # print("C"+5)

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        ######
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, kwargs=kwargs
        )###
        # print(kwargs)
        # print("C"+5)
        return [input_ids, generation_config, kwargs, generation_mode, prepared_stopping_criteria,prepared_logits_processor]

    def generate_(
        self,
        kwargs_llm = None,
        kwargs_vlm = None,
        inputs_llm: Optional[torch.Tensor] = None,        
        generation_config_llm: Optional[GenerationConfig] = None,
        logits_processor_llm: Optional[LogitsProcessorList] = None,
        stopping_criteria_llm: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn_llm: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus_llm: Optional[bool] = None,
        assistant_model_llm: Optional["PreTrainedModel"] = None,
        streamer_llm: Optional["BaseStreamer"] = None,
        negative_prompt_ids_llm: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_llm: Optional[torch.Tensor] = None,
        inputs_vlm: Optional[torch.Tensor] = None,        
        generation_config_vlm: Optional[GenerationConfig] = None,
        logits_processor_vlm: Optional[LogitsProcessorList] = None,
        stopping_criteria_vlm: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn_vlm: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus_vlm: Optional[bool] = None,
        assistant_model_vlm: Optional["PreTrainedModel"] = None,
        streamer_vlm: Optional["BaseStreamer"] = None,
        negative_prompt_ids_vlm: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_vlm: Optional[torch.Tensor] = None,
        return_dict = None,
        output_hidden_states=None,
        merge_feature_layers=None,
        feature_weights=None,
        max_new_tokens=None,
        labels=None,
        combined_head=None
    ):
        # (input_ids_llm, generation_config_llm, kwargs_llm, generation_mode_llm,
        # prepared_logits_processor_llm, prepared_stopping_criteria_llm,prepared_logits_processor_llm) 
        # for i in range(len(kwargs_llm)):
        kwargs_llm["max_new_tokens"] = max_new_tokens
        kwargs_vlm["max_new_tokens"] = max_new_tokens
        generating_params_llm = self.prepare_generation_params(
            self.llm, generation_config_llm, inputs_llm,synced_gpus_llm, logits_processor_llm, stopping_criteria_llm, assistant_model_llm, streamer_llm, prefix_allowed_tokens_fn_llm, negative_prompt_ids_llm,
            negative_prompt_attention_mask_llm, kwargs_llm
        )
        generating_params_llm.append(streamer_llm)
        generating_params_llm.append(synced_gpus_llm)
        
        if("lmms-lab" in self.vlm_name):
            images = kwargs_vlm.pop("images")
            modalities = kwargs_vlm.pop("modalities")
            inputs_vlm = kwargs_vlm.pop("input_ids")
            attention_mask = kwargs_vlm.pop("attention_mask")
            # print(inputs_vlm.device,images[0].device)#,modalities[0].device)
            (inputs_vlm, position_ids, 
            attention_mask, _, inputs_embeds, _
            ) = self.vlm.prepare_inputs_labels_for_multimodal(inputs_vlm, None, attention_mask, None, 
                                                        None, images, modalities)
            generating_params_vlm = self.vlm.prepare_generation_params(
                inputs_vlm, generation_config_vlm, logits_processor_vlm,
                stopping_criteria_vlm, prefix_allowed_tokens_fn_vlm, synced_gpus_vlm, assistant_model_vlm, 
                streamer_vlm, negative_prompt_ids_vlm,
                negative_prompt_attention_mask_vlm,
                position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                **kwargs_vlm
            )
        # kwargs_vlm.update({"position_ids":position_ids,
        #                    "attention_mask":attention_mask,
        #                    "inputs_embeds":inputs_embeds})
        # kwargs_vlm.update({"inputs_embeds":inputs_embeds})
        else:        
            generating_params_vlm = self.prepare_generation_params(
            self.vlm, generation_config_vlm, inputs_vlm,synced_gpus_vlm, logits_processor_vlm, stopping_criteria_vlm, assistant_model_vlm, streamer_vlm, prefix_allowed_tokens_fn_vlm, negative_prompt_ids_vlm,
            negative_prompt_attention_mask_vlm, kwargs_vlm)


        generating_params_vlm.append(streamer_vlm)
        generating_params_vlm.append(synced_gpus_vlm)
        # generating_params_llm["return_dict_in_generate"] = return_dict
        # generating_params_vlm["return_dict_in_generate"] = return_dict
        # generating_params_llm["output_hidden_states"] = return_dict
        
        if generating_params_llm[3] in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids_llm, kwargs_llm = self._expand_inputs_for_generation(
                input_ids=generating_params_llm[0],
                expand_size=generating_params_llm[1].num_return_sequences,
                is_encoder_decoder=self.llm.config.is_encoder_decoder,
                model_kwargs=generating_params_llm[2],
            )
            input_ids_vlm, kwargs_vlm = self._expand_inputs_for_generation(
                input_ids=generating_params_vlm[0],
                expand_size=generating_params_vlm[1].num_return_sequences,
                is_encoder_decoder=self.vlm.config.is_encoder_decoder,
                model_kwargs=generating_params_vlm[2],
            )
            # print(input_ids_vlm, kwargs_vlm)
            # print("C"+5)
            generating_params_llm[0] = input_ids_llm
            generating_params_llm[2] = kwargs_llm
            
            generating_params_vlm[0] = input_ids_vlm
            generating_params_vlm[2] = kwargs_vlm


            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                generating_params_llm,
                generating_params_vlm,
                return_dict=return_dict,
                output_hidden_states=output_hidden_states,
                merge_feature_layers=merge_feature_layers,
                feature_weights=feature_weights,
                combined_head=None
            )
        if generating_params_vlm[1].cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if not callable(getattr(self, "_reset_cache", None)):
                raise ValueError(
                    "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
                    " Make sure this model implements a `_reset_cache` function."
                )
            self._reset_cache()
        # Convert to legacy cache format if requested
        if (
            generating_params_llm[1].return_legacy_cache is not False  # Should check for `True` after v4.47
            and not is_torchdynamo_compiling()
            and hasattr(result, "past_key_values")
            and hasattr(result.past_key_values, "to_legacy_cache")
            and result.past_key_values.to_legacy_cache is not None
        ):
            # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
            should_convert_cache = generation_config_llm.return_legacy_cache
            is_user_defined_cache = user_defined_cache_llm is not None
            is_default_cache_type = (
                type(result.past_key_values) == DynamicCache  # noqa E721
                or (
                    isinstance(result.past_key_values, EncoderDecoderCache)
                    and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
                    and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
                )
            )
            if not is_user_defined_cache and is_default_cache_type:
                logger.warning_once(
                    "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                    "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                    "keep returning the legacy format, please set `return_legacy_cache=True`."
                )
                should_convert_cache = True
            if should_convert_cache:
                result.past_key_values = result.past_key_values.to_legacy_cache()


        return result

    def prepare_sampling_params(self, generation_config, stopping_criteria):
        
        # print(vars(generation_config))
        if("_pad_token_tensor" in vars(generation_config)):
            pad_token_id = generation_config._pad_token_tensor
        else:  
            pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
                
        output_scores = output_scores if output_scores is not None else generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else generation_config.return_dict_in_generate
        )
        
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        return (return_dict_in_generate, output_attentions, output_hidden_states, max_length,
         do_sample,has_eos_stopping_criteria,pad_token_id,scores,raw_logits,decoder_attentions,
         cross_attentions,decoder_hidden_states,output_scores,output_logits)

    def _sample(
        self,
        generating_params_llm,
        generating_params_vlm,
        return_dict=None,
        output_hidden_states=None,
        merge_feature_layers=None,
        feature_weights=None,
        combined_head=None
    ):
        (input_ids_llm, generation_config_llm, kwargs_llm, generation_mode_llm,
        prepared_stopping_criteria_llm,prepared_logits_processor_llm,streamer_llm,synced_gpus_llm) = generating_params_llm
        
        sampling_params_llm = self.prepare_sampling_params(generation_config_llm,prepared_stopping_criteria_llm)
        
        (return_dict_in_generate_llm, output_attentions_llm, output_hidden_states_llm, max_length_llm,
         do_sample,has_eos_stopping_criteria_llm,pad_token_id_llm,scores_llm,raw_logits_llm,decoder_attentions_llm,
         cross_attentions_llm,decoder_hidden_states_llm,output_scores_llm,output_logits_llm) = sampling_params_llm
        

        (input_ids_vlm, generation_config_vlm, kwargs_vlm, generation_mode_vlm,
        prepared_stopping_criteria_vlm,prepared_logits_processor_vlm,streamer_vlm,synced_gpus_vlm) = generating_params_vlm

        sampling_params_vlm = self.prepare_sampling_params(generation_config_vlm,prepared_stopping_criteria_vlm)

        (return_dict_in_generate_vlm, output_attentions_vlm, output_hidden_states_vlm, max_length_vlm,
         do_sample,has_eos_stopping_criteria_vlm,pad_token_id_vlm,scores_vlm,raw_logits_vlm,decoder_attentions_vlm,
         cross_attentions_vlm,decoder_hidden_states_vlm,output_scores_vlm,output_logits_vlm) = sampling_params_vlm
        
        # print(prepared_logits_processor_vlm)
        # prepared_logits_processor_vlm[0].temperature = 0.7
        # prepared_logits_processor_vlm[1].top_k = 20
        # prepared_logits_processor_vlm[2].top_p = 0.8

        input_ids_llm_start = input_ids_llm.clone()
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate_llm and self.llm.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions_llm else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size_llm, cur_len_llm = input_ids_llm.shape
        this_peer_finished_llm = False
        unfinished_sequences = torch.ones(batch_size_llm, dtype=torch.long, device=input_ids_llm.device)
        kwargs_llm = self.llm._get_initial_cache_position(input_ids_llm, kwargs_llm)
        kwargs_vlm = self.vlm._get_initial_cache_position(input_ids_vlm, kwargs_vlm)

        # start = time.time()
        while self._has_unfinished_sequences(
            this_peer_finished_llm, synced_gpus_llm, device=input_ids_llm.device, cur_len=cur_len_llm, max_length=max_length_llm
        ):
            # prepare model inputs
            # model_inputs_llm = self.prepare_inputs_for_generation_llm(input_ids_llm, kwargs=kwargs_llm)
            model_inputs_llm = self.llm.prepare_inputs_for_generation(input_ids_llm, **kwargs_llm)
            model_inputs_llm.update({"output_attentions": output_attentions_llm} if output_attentions_llm else {})
            model_inputs_llm.update({"output_hidden_states": output_hidden_states_llm} if output_hidden_states_llm else {})

            # print(input_ids_vlm,kwargs_vlm)
            model_inputs_vlm = self.vlm.prepare_inputs_for_generation(input_ids_vlm,**kwargs_vlm)
            # prepare model inputs
            # if(self.vlm_name == "Qwen/Qwen2-VL-7B-Instruct"):
            #     model_inputs_vlm = self.prepare_inputs_for_generation_vlm(input_ids_vlm, kwargs=kwargs_vlm)
            # elif(self.vlm_name == "lmms-lab/llava-onevision-qwen2-7b-ov"):
            #     model_inputs_vlm = self.vlm.prepare_inputs_for_generation(input_ids_vlm, kwargs=kwargs_vlm)

            
            model_inputs_vlm.update({"output_attentions": output_attentions_vlm} if output_attentions_vlm else {})
            model_inputs_vlm.update({"output_hidden_states": output_hidden_states_vlm} if output_hidden_states_vlm else {})

            # for elem in model_inputs_llm:
            #     if(elem=="cache_position" or elem=="inputs_embeds" or elem=="num_logits_to_keep"):
            #         continue
            #     if torch.equal(model_inputs_llm[elem], model_inputs_vlm[elem]):
            #         print("Tensors are exactly equal")
            #     else:
            #         print(elem)
            # print("C"+4)
            # forward pass to get next token
            # print(model_inputs_vlm)
            outputs_combined, _, outputs_vlm = self.forward(model_inputs_llm,model_inputs_vlm, return_dict=return_dict, output_hidden_states=output_hidden_states,
                                                     merge_feature_layers=merge_feature_layers,feature_weights=feature_weights,combined_head=combined_head)
            
            # outputs_llm = self.llm(**model_inputs_llm, return_dict=True)
            # outputs_vlm = self.vlm(**model_inputs_vlm, return_dict=True)

            if synced_gpus_llm and this_peer_finished_llm:
                continue  # don't waste resources running the code we don't need

            # print(outputs_llm.logits.shape,outputs_vlm.logits.shape,outputs_combined.logits.shape)

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            # .float() is needed to retain precision for later logits manipulations
            # next_token_logits_llm = outputs_llm.logits[:, -1, :].clone().float()
            # next_token_logits_vlm = outputs_vlm.logits[:, -1, :].clone().float()
            next_token_logits_combined = outputs_combined.logits[:, -1, :].clone().float()
            next_token_logits_combined = next_token_logits_combined
            # next_token_logits_combined = (next_token_logits_llm + next_token_logits_vlm) / 2
            
            ntl = next_token_logits_combined.reshape(-1)
            # for proc in prepared_logits_processor_llm:
            #     print(vars(proc))
            # for proc in prepared_logits_processor_vlm:
            #     print(vars(proc))

            # print(prepared_logits_processor_llm[-1].top_p,prepare
            # pre-process distribution
            # next_token_scores_llm = prepared_logits_processor_llm(input_ids_llm, next_token_logits_llm)
            # finite_mask = torch.isfinite(next_token_scores_llm)
            # # Use the mask to get only the non-infinite entries
            # non_inf_entries = next_token_scores_llm[finite_mask]
            # print(non_inf_entries)

            # next_token_scores_vlm = prepared_logits_processor_vlm(input_ids_vlm, next_token_logits_vlm)
            # print(next_token_logits_llm.min(),next_token_scores_vlm.min())
            # print(next_token_logits_vlm.min(),next_token_scores_vlm.min())
            # finite_mask = torch.isfinite(next_token_scores_vlm)
            # non_inf_entries = next_token_scores_vlm[finite_mask]
            # print(non_inf_entries)

            next_token_logits_combined = next_token_logits_combined#.to(input_ids_vlm.device)


            # next_token_scores_combined = prepared_logits_processor_llm(input_ids_llm, next_token_logits_combined)
            # next_token_scores_combined = prepared_logits_processor_vlm(input_ids_vlm, next_token_logits_combined)
            next_token_scores_combined = prepared_logits_processor_llm(input_ids_llm, next_token_logits_combined)

            
            vid = next_token_scores_combined.reshape(-1)
            do_sample = self.sample
            # token selection
            if do_sample:
                # probs = nn.functional.softmax(next_token_scores_vlm, dim=-1)
                # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                # probs = nn.functional.softmax(next_token_scores_combined, dim=-1)
                # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                #nts combined
                # nts_combined = (next_token_scores_llm + next_token_scores_vlm) /2 # x + -inf = -inf
                # print(nts_combined.min(),nts_combined.max())
                probs_combined = nn.functional.softmax(next_token_scores_combined, dim=-1)
                # print(probs_combined.min())
                # print(probs_combined.max())

                # # average prob
                # probs_llm = nn.functional.softmax(next_token_scores_llm, dim=-1)
                # probs_vlm = nn.functional.softmax(next_token_scores_vlm, dim=-1)
                # probs_combined = (probs_llm + probs_vlm) / 2
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs_combined, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores_combined, dim=-1)
                # print(next_tokens)

            # print(ntl[next_tokens],vid[next_tokens])
            
            if has_eos_stopping_criteria_llm:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id_llm * (1 - unfinished_sequences)

            generated_ids = [
            output_ids[len(input_ids_llm_):] for input_ids_llm_, output_ids in zip(input_ids_llm_start, input_ids_llm)
            ]
            if("lmms-lab" in self.vlm_name):
                output_text = self.tokenizer_vlm.batch_decode(input_ids_llm, skip_special_tokens=True)
            else:
                output_text = self.vlm_processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            # print(output_text)
            response_llm = self.tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print(response_llm)

            # update generated ids, model inputs, and length for next step
            input_ids_llm = torch.cat([input_ids_llm, next_tokens[:, None]], dim=-1)
            input_ids_vlm = torch.cat([input_ids_vlm, next_tokens[:, None]], dim=-1)
            if streamer_llm is not None:
                streamer_llm.put(next_tokens.cpu())
            kwargs_llm = self.llm._update_model_kwargs_for_generation(
                outputs_combined,
                kwargs_llm,
                is_encoder_decoder=self.llm.config.is_encoder_decoder,
            )
            kwargs_vlm = self.vlm._update_model_kwargs_for_generation(
                outputs_vlm,
                kwargs_vlm,
                is_encoder_decoder=self.vlm.config.is_encoder_decoder,
            )
            # print(kwargs_vlm)

            unfinished_sequences = unfinished_sequences & ~prepared_stopping_criteria_llm(input_ids_llm, scores_llm)
            this_peer_finished_llm = unfinished_sequences.max() == 0
            cur_len_llm += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # del outputs_llm
            del outputs_vlm

        if streamer_llm is not None:
            streamer_llm.end()
        # print("finished")
        # return input_ids_vlm
        return input_ids_llm