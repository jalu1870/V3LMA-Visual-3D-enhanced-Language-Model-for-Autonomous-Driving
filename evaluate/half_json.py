import json
import os

out_directory = "../LLaVA-NeXT/out/"
output_files = os.listdir(out_directory) 

for i, file_name in enumerate(output_files):
    print(file_name)
    if("2412" not in file_name):# and displayed == False): 
        continue
    with open(out_directory + file_name, 'r') as file:
        file_data = json.load(file)

    new_data = []
    for ii, data in enumerate(file_data):
        if(ii % 4 == 0):
            new_data.append(data)
    # print(new_data[0])
    # print(new_data[1])
    # print("C"+5)
    with open(out_directory + "new/" + file_name, 'w') as file:
        json.dump(new_data,file)