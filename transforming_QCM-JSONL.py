import json

# Function to convert the JSON questions to JSONL format
def convert_to_jsonl(input_json_file, output_jsonl_file):
    # Open and load the JSON file
    with open(input_json_file, 'r', encoding='utf-8') as infile:
        questions = json.load(infile)

    # List to store the JSONL formatted data
    jsonl_output = []

    # Iterate through each question in the list
    for question in questions:
        # Format the question and answers as a string for the user input
        question_text = f"{question['question']}:\n"
        for option, answer in question['answers'].items():
            question_text += f"{option.upper()}) {answer}\n"
        
        # Format the correct answers (in case there are multiple)
        correct_answers = question['correct_answers']
        if len(correct_answers) > 1:
            correct_answers_text = "Les réponses correctes sont : "
            correct_answers_text += ", ".join([f"{ans.upper()}) {question['answers'][ans]}" for ans in correct_answers])
        else:
            correct_answers_text = '{' + f"""'Réponse correcte':{correct_answers[0].upper()}, 'Explication': {question['answers'][correct_answers[0]]}""" + '}'

        # Create the conversation structure
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": question_text.strip()
                },
                {
                    "role": "assistant",
                    "content": correct_answers_text
                }
            ]
        }

        # Append the conversation to the output list
        jsonl_output.append(conversation)

    # Write the JSONL formatted data to the output file
    with open(output_jsonl_file, 'w', encoding='utf-8') as outfile:
        for entry in jsonl_output:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Example usage
input_json_file = 'FrenchMedMCQA/corpus/dev.json'  # Replace with your input file
output_jsonl_file = 'formatted_conversations_3.jsonl'  # Replace with your desired output file

convert_to_jsonl(input_json_file, output_jsonl_file)


import json

def read_jsonl(file_path):
    try:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def write_jsonl(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")

def concatenate_jsonl_files(file_paths, output_file):
    combined_data = []
    for file_path in file_paths:
        combined_data.extend(read_jsonl(file_path))

    write_jsonl(output_file, combined_data)

# Example usage
file_paths = ['formatted_conversations.jsonl', 'formatted_conversations_2.jsonl', 'formatted_conversations_3.jsonl']
output_file = 'formatted_cconversations_concatenated.jsonl'

concatenate_jsonl_files(file_paths, output_file)
