import pickle
import numpy as np

# Function to process the file and generate the dictionary
def process_txt_to_dict(file_path):
    res_dict = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_id = None
        current_answers = []

        for line in lines:
            line = line.strip()

            # Check if the line is an ID line
            if line.startswith("id:"):
                if current_id is not None:
                    if len(current_answers) != 20:
                        print(f"ID {current_id} does not have exactly 20 lines of answers. It has {len(current_answers)} lines.")
                    res_dict[current_id] = current_answers

                current_id = line.split(":")[1]
                current_answers = []

            # Check if it's a separator line
            elif line.startswith("---"):
                if current_id is not None:
                    if len(current_answers) != 20:
                        print(f"ID {current_id} does not have exactly 20 lines of answers. It has {len(current_answers)} lines.")
                    res_dict[current_id] = current_answers
                    current_id = None
                    current_answers = []

            # Otherwise, it's an answer line
            elif current_id is not None:
                current_answers.append(line)

        # Add the last ID and answers
        if current_id is not None:
            if len(current_answers) != 20:
                print(f"ID {current_id} does not have exactly 20 lines of answers. It has {len(current_answers)} lines.")
            res_dict[current_id] = current_answers

    return res_dict

# Save the dictionary to a pkl file
def save_dict_to_pkl(data, output_path):
    with open(output_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


# Claim Txt Path. Replace with claim path
input_file_path = "0_claim.txt"
# Original Path for semantic generation
original = np.load("0.pkl",allow_pickle=True)

res_dict = process_txt_to_dict(input_file_path)

final = []
for sample in original:
    sample_id = sample["id"]
    if sample_id not in res_dict:
        continue
    else:
        sample["generations"] = [gen for gen in res_dict[sample_id]]
        final.append(sample)

with open('0_know.pkl', 'wb') as pkl_file:
    pickle.dump(final, pkl_file)


