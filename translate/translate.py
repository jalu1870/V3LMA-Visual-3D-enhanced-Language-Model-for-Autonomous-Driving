
import json



import re

# Function to check if a character is Chinese
def is_chinese(char):
    # Check if the character is in any of the Chinese Unicode ranges
    return '\u4e00' <= char <= '\u9fff' or \
           '\u3400' <= char <= '\u4DBF' or \
           '\u20000' <= char <= '\u2A6DF'

# Function to find the positions of Chinese characters in English text
def find_chinese_characters(text):
    chinese_positions = []
    for index, char in enumerate(text):
        if is_chinese(char):
            chinese_positions.append(index)
    return chinese_positions

def get_connected_sequences(sequence):
    # Initialize an empty list to store the connected sequences
    connected_sequences = []
    
    
    # Initialize the start of the first sequence
    start_nr = sequence[0]
    
    # Iterate through the sequence starting from the second element
    for i in range(1, len(sequence)):
        # If there's a break in the sequence (difference greater than 1)
        if sequence[i] != sequence[i - 1] + 1:
            # Add the current sequence to the result
            connected_sequences.append((start_nr, sequence[i - 1]))
            # Start a new sequence from the current element
            start_nr = sequence[i]
    
    # Don't forget to add the last sequence
    connected_sequences.append((start_nr, sequence[-1]))
    
    return connected_sequences

def check_translate(out_,trans_dict,model_translate,tokenizer_translate):
    
    # Example usage
    # text = "This is English text with some 中文 characters in it."
    positions = find_chinese_characters(out_)
    # print(len(positions))

    if(len(positions) == 0):
        return out_,trans_dict

    connected_sequences = get_connected_sequences(positions)
    # print(len(connected_sequences))
    # print(f"Positions of Chinese characters: {positions}",connected_sequences)

    # new_out = out_.copy()
    for seq in connected_sequences[::-1]:
        text_ = out_[seq[0]:seq[1]+1]

        
        # article_hi = "避让行人：确保行人安全通行"

        if(text_ in trans_dict):
            decoded = trans_dict[text_]
        else:
            # translate Hindi to French
            tokenizer_translate.src_lang = "zh_CN"
            encoded_hi = tokenizer_translate(text_, return_tensors="pt").to("cuda")
            generated_tokens = model_translate.generate(
                **encoded_hi,
                forced_bos_token_id=tokenizer_translate.lang_code_to_id["en_XX"]
            )
            decoded = tokenizer_translate.batch_decode(generated_tokens, skip_special_tokens=True)
            # print(text_,decoded)
            trans_dict[text_] = decoded
        out_ = out_[:seq[0]] + " " + decoded[0] + out_[seq[1]+1:]
    
    return out_,trans_dict
# print(out_)
# # print(new_out)

# file_name = "out/results_20241127_170327.json"

# with open(file_name, 'r') as file:
#     data = json.load(file)
# out_ = data[-4]["result"]
# print(out_)