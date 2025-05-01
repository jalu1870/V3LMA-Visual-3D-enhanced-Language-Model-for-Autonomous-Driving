
import os
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import pandas as pd
from translate import translate
from transformers import pipeline

import nltk
nltk.download('wordnet')
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def bleu(model_output, ground_truth):
    chencherry = SmoothingFunction()
    return sentence_bleu(ground_truth.split(), model_output.split(),smoothing_function=chencherry.method1)

def meteor(model_output, ground_truth):
    return meteor_score([ground_truth.split()], model_output.split())

def rouge(model_output, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ground_truth, model_output)
    return rouge_scores["rouge1"].recall

# Cosine Similarity
def cossimilarity(model_output, ground_truth):
    vectorizer = TfidfVectorizer().fit_transform([ground_truth, model_output])
    return cosine_similarity(vectorizer[0:1], vectorizer[1:2]).flatten()[0]

def load_model_nvidia():
    import torch
    from transformers import AutoTokenizer, AutoModel


    # load model with tokenizer
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True,device_map="auto").eval()
    return model

def load_bert():
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertModel.from_pretrained("bert-large-uncased").to("cuda").eval()
    return model, tokenizer

def load_jinja():
    from transformers import AutoModel

    # Initialize the model
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to("cuda").eval()
    return model, None

import re
import numpy as np

def remove_sequences(text):
    # Regular expression to match any substring starting with '(' and ending with ')'
    return re.sub(r'\(.*?\)', '', text)

def get_nvidia_similarity(model,prediction, gt):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    # Each query needs to be accompanied by an corresponding instruction describing the task.
    task_name_to_instruct = {"example": "Given a ground truth answer, retrieve passages that are semantically similar.",} # identical

    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    # query_prefix = "Retrieve semantically similar text."
    queries = [
        gt
    ]

    # No instruction needed for retrieval passages
    passage_prefix = ""
    passages = [
        prediction
    ]

    # get the embeddings
    max_length = 32768
    query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
    passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)

    # normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
    
    scores = (query_embeddings @ passage_embeddings.T)# * 100
    return scores
def get_bert_similarity(model,prediction, gt):
    model, tokenizer = model

    prediction_input = tokenizer(prediction, return_tensors='pt').to("cuda")
    gt_input = tokenizer(gt, return_tensors='pt').to("cuda")

    if(len(prediction_input["input_ids"][0]) > 512):
        return -1
    prediction_output = model(**prediction_input)["pooler_output"]
    gt_output = model(**gt_input)["pooler_output"]
    
    prediction_output = F.normalize(prediction_output, p=2, dim=1)
    gt_output = F.normalize(gt_output, p=2, dim=1)
    del prediction_input, gt_input
    return prediction_output @ gt_output.T

def load_qa_gt(data,scene_token,displayed,per_frame=False):
    questions = []
    answers = []
    if(per_frame):
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
                        answers.append(a)
                    else:
                        answers[questions.index(q)] += a
                        # print(answers)
            
        for i,answer in enumerate(answers):
            if("no" in answer.lower() and "yes" in answer.lower()):
                answers[i] = answer.lower().replace("no","")
    else:
        answers.append(data[scene_token]["scene_description"])
        questions.append("")
    if(displayed == False):
        # print(answers[0])
        displayed = True
    return questions, answers, displayed

def lingoqa_judge(question,answer,prediction,pipe):
    input = f"[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"

    result = pipe(input)

    # Print the result and score
    score = result[0]['score']
    
    return score

def analyse_qa_similarity(trans_dict,model,data,out_directory,file_name,displayed,save_per_sample_result,per_frame=False,no_q=True,inf_only=True):
    predictions = None
    try:
        with open(out_directory + file_name, 'r') as file:
            predictions = json.load(file)
    except:
        print("Couldnt open: " + file_name)
        return trans_dict,0,0,0,0,0,0,0,0
    similarities, bleu_scores, meteor_scores, rouge_scores, cos_scores, lingo_scores, accuracy = [],[],[],[],[],[],[]
    
    tasks = []
    # translation models
    model_translate = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to("cuda:0")
    tokenizer_translate = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    

    # lingo model
    model_name = 'wayveai/Lingo-Judge'
    pipe = pipeline("text-classification", model=model_name,device="cuda")

    #load dataset
    if(data is None):
        data = load_lingoqa_gt(lingo_qa_parquet_path)
    
    per_sample_dict = {}
    # iterate over all contents of the prediction file
    for scene_nr, scene_prediction in enumerate(predictions):
        if("task_token" not in scene_prediction or scene_prediction["task_token"] in tasks):
            continue
        # print(len(predictions))
        if(scene_prediction["task_token"] == -1):
            continue
        # preprocess input depending on the time of generation of the prediction file
        tasks.append(scene_prediction["task_token"])
        
        model_name = scene_prediction["model_name"]
        scene_token = scene_prediction["scene_token"]
        prompt_name = "prompt"
        if("prompt" not in scene_prediction):
            if("llm_prompt" in scene_prediction):
                prompt_name = "llm_prompt"
            elif("vlm_prompt" in scene_prediction):
                prompt_name = "vlm_prompt"

        question_prediction = scene_prediction[prompt_name].split("\n")[-1]
        
        if("vlm_prompt" in scene_prediction):
            question_prediction = scene_prediction["vlm_prompt"]
        elif("question" in scene_prediction):
            question_prediction = scene_prediction["question"]
            if("\nuser\n" in question_prediction):
                question_prediction = question_prediction.split("\nuser\n")[-1]
            if("\nassistant\n" in question_prediction):
                question_prediction = question_prediction.split("\nassistant\n")[-2]
                print(question_prediction)
        if(question_prediction == ""):
            if("?" in scene_prediction[prompt_name]):
                question_prediction = scene_prediction[prompt_name].split("?")[0].strip() + "?"
                if("." in question_prediction):
                    question_prediction = question_prediction.split(".")[-1].strip()
                if("\n\n" in question_prediction):
                    question_prediction = question_prediction.split("\n\n")[-1].strip()
            else:
                print("no ? or . in output of model: {}, file: {}".format(model_name,file_name))
                continue
          
        answer_prediction = scene_prediction["result"]
        if isinstance(answer_prediction, list):
            answer_prediction = answer_prediction[0]

        if("system\nYou are a helpful assistant.\nuser\n" in answer_prediction):
            answer_prediction = answer_prediction.replace("system\nYou are a helpful assistant.\nuser\n","").replace(question_prediction,"").replace("\nassistant\n","")

        predictions[scene_nr]["eval_answer"] = answer_prediction
        if("translation" in scene_prediction):
            answer_prediction = scene_prediction["translation"]
        else:
            # translate input if not in english
            answer_prediction, trans_dict = translate.check_translate(answer_prediction,trans_dict,model_translate,tokenizer_translate)

            predictions[scene_nr]["translation"] = answer_prediction
            
        
        if("assistant\n" in answer_prediction):
            answer_prediction = answer_prediction.split("assistant\n")[-1]
            
        questions_gt = question_prediction
        
        
        index = 0
        if(no_q == False):
            try:
                index = questions_gt.index(question_prediction)
            except:
                print(model_name,file_name,question_prediction)
                print("c"+5)

        
        
        answer_prediction = remove_sequences(answer_prediction)
        
        try:
            
            questions_gt = data[0][scene_prediction["task_token"]]
            answers_gt = data[1][scene_prediction["task_token"]]
        except:
            print("lingo not loaded correctly: " + scene_prediction["task_token"])
            continue

        max_bleu_score = -1
        max_meteor_score_ = -1
        max_rouge_score_ = -1
        max_cos_score_ = -1
        max_lingo_score = -1

        # for the two answer options get the one with the higher score.
        for answer_gt in answers_gt:
            
            lingo_score = lingoqa_judge(questions_gt,answer_gt, answer_prediction,pipe)
            
            bleu_score_ = bleu(answer_prediction,answer_gt)
            
            meteor_score_ = meteor(answer_prediction,answer_gt)
            
            rouge_score_ = rouge(answer_prediction,answer_gt)
            

            if(max_bleu_score < bleu_score_):
                max_bleu_score = bleu_score_
            if(max_meteor_score_ < meteor_score_):
                max_meteor_score_ = meteor_score_
            if(max_rouge_score_ < rouge_score_):
                max_rouge_score_ = rouge_score_
                
            if(max_lingo_score < lingo_score):
                max_lingo_score = lingo_score
                
        bleu_scores.append(max_bleu_score)
        meteor_scores.append(max_meteor_score_)
        rouge_scores.append(max_rouge_score_)
        lingo_scores.append(max_lingo_score)
        accuracy.append(max_lingo_score > 0.5)

        accuracy.append(lingo_scores[-1] > 0.5)
    if(save_per_sample_result):
        with open(out_directory + file_name.replace(".json","_per_sample_lingo.json"), 'w') as file:
            json.dump(lingo_scores,file,indent=4)

    
    if(inf_only == False):
        with open(out_directory + file_name, 'w') as file:
            json.dump(predictions,file,indent=4)
            
    del pipe
    del model_translate
    del tokenizer_translate
    return trans_dict,np.mean(similarities),np.mean(bleu_scores),np.mean(meteor_scores),np.mean(rouge_scores),np.mean(cos_scores),np.mean(lingo_scores),np.mean(accuracy),len(bleu_scores)

def load_lm_scene_predictions(out_directory,file_name):
    with open(out_directory + file_name, 'r') as file:
        file_data = json.load(file)
    predictions = file_data["predictions"]
    scene_tokens = predictions.keys()
    return scene_tokens, predictions

def load_omnidrive_scene_gt(frame_token):
    omnidirve_path =r"/home/ge32buc/new/DriveLM/challenge/data/train/" + frame_token + ".json"
    with open(omnidirve_path, 'r') as file:
        file_data = json.load(file)
    scene_description += file_data + "\n"
    return scene_description

def load_lingoqa_gt(lingo_qa_parquet_path):
    parquet_path = lingo_qa_parquet_path
    try:
        tasks = pd.read_parquet(parquet_path).to_numpy()
    except:
        tasks = pd.read_parquet(parquet_path.replace("../","")).to_numpy()

    answers = {}
    questions = {}
    for task in tasks:
        id_, scene, images, question, answer = task
        if(id_ in answers):
            answers[id_].append(answer)
        else:
            answers[id_] = [answer]
            questions[id_] = question
        
    return questions, answers


def evaluate(out_directory,lingo_qa_parquet_path,save_per_sample_result):       
    import json
    import os
    from tqdm import tqdm
    
    data = load_lingoqa_gt(lingo_qa_parquet_path)
    
    output_files = os.listdir(out_directory)
    evaluation_results = []
    
    eval_results = {}
    similarity = 0
    count = 0
    model = None
    
    displayed = False
    trans_dict = {}

    
    if(os.path.exists(out_directory + "/eval_result.json")):
        with open(out_directory + "/eval_result.json", 'r') as file:
            eval_results = json.load(file)  # 'indent' for pretty-printing
    
    for file_name in tqdm(output_files):
        
        if(".json" not in file_name or "config" in file_name):
            continue
        
        if("fine_tuned" in file_name):
            continue
        
        print(file_name)
        results = []
        gts = []
        last = file_name[-5:]
        if(file_name[-5:] != ".json" or "eval_result" in file_name):
            continue
        
        trans_dict,similarity, bleu_mean, meteor_mean, rouge_mean, cos_mean, lingo_mean,accuracy,nr_samples = analyse_qa_similarity(trans_dict,model,data,out_directory,file_name,displayed,save_per_sample_result)

        while(file_name in eval_results):
            file_name = file_name + "_"
            
        eval_results[file_name] = {
            "similarity": similarity,
            "lingo_mean": lingo_mean,
            "bleu": bleu_mean,
            "meteor": meteor_mean,
            "rouge": rouge_mean,
            "cos": cos_mean,
            "accuracy": accuracy,
            "nr_samples":nr_samples
        }

        print(file_name + ": "+ str(similarity), lingo_mean, bleu_mean, meteor_mean, rouge_mean, cos_mean,accuracy,nr_samples)
        # return
        print(out_directory + "/eval_result.json")
        with open(out_directory + "/eval_result.json", 'w') as file:
            json.dump(eval_results, file, indent=4)  # 'indent' for pretty-printing
        # break
        
# to_dict()
def clean_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Replace characters in the ASCII range 0-31 (control characters)
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text


def analyse_results(out_directory,accumulate=False,data=None):
    results = []
    with open(out_directory + "eval_result.json", 'r') as file:
        results = json.load(file)  # 'indent' for pretty-printing
    
    # print(results)
    if(data is None):
        data = []
    keys_ = ['model_name', 'question', 'scene_token', 'task_token', \
            'llm_prompt', 'vlm_prompt', 'result', 'merge_feature_layers', \
            'feature_weights', 'merge_head_weights', 'merge_layer_weights_layers', \
            'merge_layer_weights_weights',"get_all_vlm_features_first","merged_head_weights", 'translation']
    measures = ["similarity","lingo_mean","bleu","meteor","rouge","cos","accuracy","nr_samples"]

    for result_file in results:
        if("config" in result_file):
            continue
        
        prediction = []
        try:
            with open(out_directory.replace("../out/processed/","") + result_file, 'r') as file:
                prediction = json.load(file)  # 'indent' for pretty-printing
        except:
            continue
        
        # print(prediction)
        if(len(prediction) == 0):
            continue
        print(result_file)
        entry = prediction[0].copy()
        for key in keys_:
            if(key not in entry):
                entry[key] = ""
            elif("yes换句话戕戕戕℅//" in str(entry[key]) or "yes In other words, c/o//" in str(entry[key])):
                entry[key] = "yes"
                print(result_file,key)
                # print("C"+5)
        for key in measures:
            if(key in results[result_file]):
                entry[key] = results[result_file][key]
                if("yes换句话戕戕戕℅//" in str(results[result_file][key])):
                    print(result_file,key)
                    print("C"+5)
            else:
                entry[key] = ""
                
        if(entry in data):
            continue
        # print(entry["prediction_file"])
        data.append(entry)

    
    if(accumulate == False):
        df = pd.DataFrame(data)
        # Apply clean_text function to the entire DataFrame
        df = df.applymap(clean_text)
        df.to_excel('output.xlsx', index=False, engine='openpyxl')
    else:
        return data

import argparse
parser = argparse.ArgumentParser(description="inference arguments are model_name, val_data_path, llm_prompt_for_vision.")

# Add arguments
parser.add_argument('--dataset_parquet_path', type=str, help='path to the generated dataset .parquet file for the validation set')
args = parser.parse_args()
    
lingo_qa_parquet_path = args.dataset_parquet_path

out_directory = "../out/"

lingo_qa_parquet_path
evaluate(out_directory,lingo_qa_parquet_path,False)
analyse_results(out_directory)



