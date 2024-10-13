import pandas as pd
import json

# Function to read JSON file and create DataFrame
def create_dataframe_from_json(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the entire JSON content

    # Initialize lists to hold the data
    questions = []
    answers_a = []
    answers_b = []
    answers_c = []
    answers_d = []
    answers_e = []
    correct_answers = []
    subject_names = []  # List to hold subject names

    # Iterate through each question in the JSON data
    for item in data:
        questions.append(item['question'])
        answers_a.append(item['answers']['a'])
        answers_b.append(item['answers']['b'])
        answers_c.append(item['answers']['c'])
        answers_d.append(item['answers']['d'])
        answers_e.append(item['answers']['e'])
        
        # Convert correct answers to uppercase letters
        correct_answers.append(', '.join([answer.upper() for answer in item['correct_answers']]))
        
        # Append subject name
        subject_names.append(item['subject_name'])

    # Create a DataFrame
    df = pd.DataFrame({
        'question': questions,
        'answer_A': answers_a,
        'answer_B': answers_b,
        'answer_C': answers_c,
        'answer_D': answers_d,
        'answer_E': answers_e,
        'correct_answers': correct_answers,
        'subject_name': subject_names  # Add subject_name column
    })

    return df

# Example usage
file_path = 'FrenchMedMCQA/corpus/train.json'
df = create_dataframe_from_json(file_path)

# Save the DataFrame to a CSV file
output_file_path = 'multiple_choice_questions_ref.csv'  # Update with your desired output path
df.to_csv(output_file_path, index=False)

# Display the DataFrame
print(df)