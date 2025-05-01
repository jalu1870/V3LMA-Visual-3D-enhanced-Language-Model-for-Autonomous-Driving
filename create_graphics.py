import json
import matplotlib.pyplot as plt
import re

# Function to extract the checkpoint number from save_path
def extract_checkpoint(save_path):
    match = re.search(r'checkpoint_(\d+)', save_path)
    if match:
        return int(match.group(1))
    return None

# Load the JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to extract lingo_mean and checkpoint number
def extract_lingo_means(json_data):
    data = []
    for filename, metrics in json_data.items():
        save_path = metrics.get("save_path", "")
        lingo_mean = metrics.get("lingo_mean", None)

        checkpoint = extract_checkpoint(save_path)
        if checkpoint is not None and lingo_mean is not None:
            data.append((checkpoint, lingo_mean))
    
    return sorted(data, key=lambda x: x[0])  # Sort by checkpoint number
import numpy as np
# Function to plot the lingo_mean values
def plot_lingo_mean(data,out_dir,epoch):
    lingo_means = [item["lingo_mean"] for key, item in data.items()]
    checkpoints = np.arange(len(lingo_means))
    
    print(lingo_means)
    plt.plot(checkpoints, lingo_means, marker='o')
    plt.title("Lingo Mean vs Checkpoint")
    plt.xlabel("Checkpoint")
    plt.ylabel("Lingo Mean")
    plt.grid(True)
    plt.savefig(f"{out_dir}epoch_{epoch}.png")
    print(f"Plot saved to {out_dir}{epoch}.png")

def analyse_results(out_directory,epoch):
    results = []
    with open(out_directory + "eval_result_nvidia_all_answers_all.json", 'r') as file:
        results = json.load(file)  # 'indent' for pretty-printing
    
    # print(results)
    data = []
    # keys_ = ['model_name', 'question', 'scene_token', 'task_token', \
    #         'llm_prompt', 'vlm_prompt', 'result', 'merge_feature_layers', \
    #         'feature_weights', 'merge_head_weights', 'merge_layer_weights_layers', \
    #         'merge_layer_weights_weights',"get_all_vlm_features_first","merged_head_weights", 'translation']
    measures = ["similarity","lingo_mean","bleu","meteor","rouge","cos","accuracy","nr_samples"]
    dir_ = {}

    for result_file in results:
        print(result_file)
        # if("results_20250113_" not in result_file and "results_20250114_" not in result_file and "results_20250115_" not in result_file):
        #     continue
        
        prediction = []
        try:
            with open(out_directory + result_file, 'r') as file:
                prediction = json.load(file)  # 'indent' for pretty-printing
        except:
            continue
        
        # print(prediction)
        if(len(prediction) == 0):
            continue
        entry = prediction[0].copy()
        # if("use_same_index" in entry):
        #     continue
        # if(f"epoch_{epoch}" not in entry["save_path"]):
        #     continue
        dir_[entry["save_path"]] = {}
        for key in measures:
            if(key in results[result_file]):
                dir_[entry["save_path"]][key] = results[result_file][key]
                if("yes换句话戕戕戕℅//" in str(results[result_file][key])):
                    print(result_file,key)
                    print("C"+5)
            else:
                entry[key] = ""
        
    return dir_

def extract_checkpoint_number(path):
    # print(path)
    # print("C"+5)
    match = re.search(r'fine_tuned_model_(\d+)', path)
    if match:
        return int(match.group(1))  # Return as an integer for natural sorting
    return -1  # Default value if no match is found

import natsort
def extract_losses_number(path):
    # print(path)
    # print("C"+5)
    match = re.search(r'(\d+).pkl', path)
    if match:
        return int(match.group(1))  # Return as an integer for natural sorting
    return -1  # Default value if no match is found


def extract_epoch_and_timestep(key):
    # Extract the epoch number using regex
    epoch_match = re.search(r'epoch_(\d+)', key)
    if epoch_match:
        epoch = int(epoch_match.group(1))  # Extract epoch number
        # Extract timestep if available (from 'checkpoint_40001_epoch' format)
        timestep_match = re.search(r'checkpoint_(\d+)_epoch', key)
        timestep = int(timestep_match.group(1)) if timestep_match else None
        return (epoch, timestep)
    else:
        epoch_match = re.search(r'(\d+)_epoch', key)
        epoch = int(epoch_match.group(1))  # Extract epoch number
        return (epoch, 500000)
    return (0, None)
def find_local_maxima(sequence,window_size):
    maxima = []
    
    # Loop through the sequence, excluding the first and last elements
    for i in range(1, len(sequence) - 1):
        if sequence[i-1] < sequence[i] > sequence[i+1]:
            if(i > window_size):
                if(sequence[i-window_size] + 0.3 < sequence[i]):
                    # print(sequence[i-window_size],sequence[i])
                    maxima.append(i)  # Store the index of the local maximum
            else:
                maxima.append(i)  # Store the index of the local maximum


    return maxima
import numpy as np
def plot_losses(out_dir,epoch,color):
    run_name = out_dir.split("/")[-2]
    # if(os.path.exists(f"{out_dir}val_loss_{run_name}.svg")):
    #     return 0
    import pickle

    # Replace 'file_path' with the actual path to your pickle file
    data = []
    for epoch_ in range(1,epoch+1):
        try:
            with open(f'{out_dir}losses_{epoch_}.pkl', 'rb') as file:
                data_ = pickle.load(file)
            # print(len(data))
            data = data + data_#[-1]
        except:
            pass
        
    window_size = 5000
    # data = [item for sublist in data for item in sublist][-307000:]
    print(len(data))
    data = [item for sublist in data for item in sublist]


    try:
        print(len(data))
        data = data
        
    except:
        return [],[]
        return 0
    # print(len(data))
    print(out_dir,"len train",len(data))
    train_x_values = None
    if(len(data) != 0):
        # print("C"+5)
        
        running_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        maxima = find_local_maxima(running_avg,window_size)
        # print(data)
        # print(maxima)
        running_avg = running_avg[maxima[-1]+1:]
        # print(window_size, len(running_avg))
        if(len(running_avg) < window_size):
            window_size = len(running_avg)

        # Create an index for the x-axis corresponding to the running average (the first (window_size-1) values will be excluded)
        train_x_values = list(range(window_size, len(running_avg) + 5000))
        # plt.figure()
        plt.plot(train_x_values, running_avg, color="blue", linestyle='-', label="train_loss")#out_dir.split("/")[-2])
        # print(data)
        print(out_dir)
    print(out_dir,data[::window_size])

    run = out_dir.split("/")[-2]

    # print(run)
    # print(out_dir)
    files = os.listdir(out_dir)
    # print(files)
    val_losses = [file for file in files if "losses_" + run in file]
    print(out_dir,"len val", len(val_losses))
    if(len(val_losses) == 0):
        return [],[]
        return 1
    sorted_losses = []
    for epoch_ in range(epoch+1):
        # print(epoch_)

        try:
            step_files = [file for file in val_losses if f"epoch_{epoch_}" in file]
            step_files = natsort.natsorted(step_files)
        except:
            step_files = []

        try:
            epoch_files = [file for file in val_losses if f"_{epoch_}_epoch" in file]
            epoch_files = natsort.natsorted(epoch_files)
        except:
            epoch_files = []
        
        sorted_losses = sorted_losses + step_files + epoch_files
    
    
    # print(len(sorted_losses))
    if(len(sorted_losses) == 0):
        return [],[]
        return 1
    # print("C"+5)
    val_losses_  = []
    for val_loss in sorted_losses:
        with open(f'{out_dir}{val_loss}', 'rb') as file:
            data = pickle.load(file)
        val_losses_.append(data)
        # losses_20250301_152033__checkpoint_100001_epoch_3.
    splitt = out_dir.split("/")
    print(splitt)
    inference_file = sorted_losses[0].replace("..pkl",".pth.json").replace("losses_" + splitt[-2] + "__","")
    print(inference_file)
    # print(val_losses_)
    # val_losses_ = [item for sublist in val_losses_ for item in sublist]
    val_losses_ = [sum(item)/len(item) for item in val_losses_]
    print(val_losses_)
    if(train_x_values is None):
        x_values = np.linspace(0, 100000, len(val_losses_)).astype(int)
    else:
        x_values = np.linspace(0, len(train_x_values), len(val_losses_)).astype(int)
    # print(x_values)
    # print(len(x_values))
    # print(len(x_values),len(val_losses_))
    # print(out_dir,val_losses_[::200])
    plt.plot(x_values, val_losses_, color="green", linestyle='-', label="val loss")
    plt.legend()

    try:
        with open(f"{out_dir}eval_result_nvidia_all_answers_all_new.json", 'r') as f:
            json_ = json.load(f)
    except:
        plt.savefig(f"{out_dir}val_loss_{run_name}.png")
        print(f"Plot saved to {out_dir}val_loss_{run_name}.png")
        return [],[]
        return 1
    
    sorted_dict = dict(sorted(
        json_.items(),
        key=lambda item: (
            extract_epoch_and_timestep(item[0])[0],  # Sort first by epoch
            (extract_epoch_and_timestep(item[0])[1] is None,  # No timestep should come after
            extract_epoch_and_timestep(item[0])[1] if extract_epoch_and_timestep(item[0])[1] is not None else float('inf'))
        )
    ))
    # Sort the dictionary by epoch (first) and timestep (second)
    # sorted_dict = dict(sorted(json_.items(), key=lambda item: extract_epoch_and_timestep(item[0])))

    # Display the sorted dictionary
    print(sorted_dict)

    
    with open(out_dir + inference_file, 'r') as f:
        file_cont = json.load(f)
    
    # print(file_cont)
    model_name = file_cont[0]["model_name"]
    
    lingo_scores = [json_[key]["lingo_mean"] for key in json_]

    if(train_x_values is None):
        lingo_x_values = np.linspace(0, 100000, len(lingo_scores)).astype(int)
    else:
        lingo_x_values = np.linspace(0, len(train_x_values), len(lingo_scores)).astype(int)
    # print(x_values)
    # print(len(x_values))
    print(len(lingo_x_values),len(lingo_scores))
    print(out_dir,lingo_scores[::200])
    plt.twinx()
    plt.ylabel("Lingo score")

    # Plot the second measure on the right y-axis
    color = "red"
    plt.plot(lingo_x_values, lingo_scores, color=color, linestyle='-', label="lingo mean")
    plt.legend()
    # plt.title("val_loss_ vs Checkpoint")
    # plt.xlabel("Checkpoint")
    # plt.ylabel("val_loss_ Mean")
    # plt.grid(True)
    # plt.savefig(f"{out_dir}val_loss_{run_name}.png")
    print(f"Plot saved to {out_dir}val_loss_{run_name}.png")
    # print("C"+5)
    # print("C"+5)
    return model_name, val_losses_, lingo_scores
    return 1

epoch = 3
import os

base_dir = "runs/"
runs = os.listdir(base_dir)#[::-1]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Example colors
colors = [ 'blue', 'green', 'orange', 'purple', 'red', 'yellow', 
    'cyan', 'magenta', 'darkgreen', 'darkblue', 'darkred', 'darkorange', 
    'darkviolet', 'gold', 'lime', 'fuchsia', 'brown', 'pink', 'skyblue', 
    'royalblue', 'turquoise', 'indigo', 'salmon', 'teal', 'chartreuse'
]
c = 0
import shutil
train_data = []
for i,run in enumerate(runs):
    # if(run and "20250226" not in run):# and "20250225" not in run):
    #     continue
    if("checkpoint" in run or "." in run):
        continue
    dirs__ = os.listdir(base_dir + run)
    # if(len(dirs__) < 4):
    #     print(dirs__)
    #     shutil.rmtree(base_dir + run)
    # continue
    plt.figure(figsize=(12, 8))
    plt.title("train_loss vs val_loss vs lingo_score")
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.grid(True)
    print(run)
    out_dir = base_dir + run + "/"
    c = 0
    if(c == len(colors)):
        print("continued: " + run)
        continue
    c_= plot_losses(out_dir,epoch,colors[c])
    if(c_[1] != []):
        config = []
        with open(base_dir + run + "/config_.json", 'r') as f:
            config = json.load(f)
        config_ = config["config"]
        model_name = c_[0]

        merge_feature_layers = config_[3]
        feature_weights = config_[4]
        merge_head_weights = config_[5]
        get_all_vlm_features_first = config_[6]
        apply_on_entire_state = config_[7]
        sum_weight_feature = config_[8]

        resume = config["resume"]
        lr = config["lr"]
        pretrained_weights = config["pretrained_weights"]
            
        dict_ = {
            "model_name":model_name,
            "merge_feature_layers" : merge_feature_layers,
            "feature_weights" : feature_weights,
            "merge_head_weights" : merge_head_weights,
            "get_all_vlm_features_first" : get_all_vlm_features_first,
            "apply_on_entire_state" : apply_on_entire_state,
            "sum_weight_feature" : sum_weight_feature,
            "resume":resume, 
            "lr":lr, 
            "pretrained_weights":pretrained_weights,
            "lora_r":config["lora_r"],
            "lora_alpha":config["lora_alpha"],
            "lora_bias":config["lora_bias"],
            "lora_modules":config["lora_modules"],
            "val_loss":c_[1],
            "lingo_score":c_[2],
        }

        train_data.append(dict_)
    # c += c_
    plt.close()  # Close the first figure
import pandas as pd
df = pd.DataFrame(train_data)
df.to_excel('train_data.xlsx', index=False, engine='openpyxl')
print("C"+5)
plt.legend()  # Show the legend to distinguish between the datasets
print(f"saved to: {base_dir}train_loss_{epoch}_all_first.png")
plt.savefig(f"{base_dir}train_loss_{epoch}_all_first.png")
# Example usage
file_path = 'runs/20250220_180551/eval_result_nvidia_all_answers_all.json'  # Replace with the path to your JSON file
out_dir = 'runs/20250220_180551/'
print("C"+5)
# dir_ = analyse_results(out_dir,epoch)
# print(dir_)
# # Sort the dictionary based on the checkpoint number
# sorted_data = dict(sorted(dir_.items(), key=lambda item: extract_checkpoint_number(item[0])))

# # Display the sorted dictionary
# for path, metrics in sorted_data.items():
#     print(f"{path}: {metrics}")
# plot_lingo_mean(sorted_data,out_dir,epoch)

plot_losses(out_dir,epoch)