import pandas as pd
import time 
import random 
from mistralai import Mistral


api_key = 'TvlAaEIaJYB08ufulDTk4jiRj4HXGQGG' # your API key

model = "mistral-large-latest"
questions_file = 'data/questions_en.csv'  # path to the questions file
reference_file = 'reference_qcm.csv'

df = pd.read_csv(questions_file)
df_ref = pd.read_csv(reference_file)


### Instructions and prompt 

system_instruction = """You are an expert in Medicine and we will be asking you multiple-choice questions in Medicine."""

system_instruction_2 = """
    Answer the following question with the letters of the correct answer. Each question can have one or multiple answers that are right. 
    Your answer must contain the letter of the answers, separated by commas and without any space. An ideal output is like: 'A,B', for instance.
    You merely output the answer as per the instructions you are given.
    You only output the letters that you are asked to provide, e.g. 'A,B,C' or 'C'. Your answer is always sorted alphabetically.
    """

### Prompt with quesstion 
question_prompt = lambda question, possible_answer_a, possible_answer_b, possible_answer_c, possible_answer_d, possible_answer_e: (
    f""" Question: {question}\n

    Possible answers:\n 
    A: {possible_answer_a}\n
    B: {possible_answer_b}\n
    C: {possible_answer_c}\n
    D: {possible_answer_d}\n
    E: {possible_answer_e}\n

    Correct answer(s):
    """
)

### Few-shot inference 
def generate_reference_examples(nb_references):
    reference_formatted = ''
    selected_indexes = random.sample(range(len(df_ref)), nb_references)
    for i in selected_indexes:
        reference_formatted += question_prompt(df_ref["question"][i], df_ref["answer_A"][i], df_ref["answer_B"][i], df_ref["answer_C"][i], df_ref["answer_D"][i], df_ref["answer_E"][i])
        reference_formatted += df_ref['correct_answers'][i].upper()
        reference_formatted += '\n\n'
        i+=1
    return reference_formatted

reference_examples = generate_reference_examples(4)

### Answers - Generation 
answers = []
client = Mistral(api_key=api_key)

for row_idx, row in df.iterrows():
    time.sleep(3)
    prompt = question_prompt(
      row["question"],
      row["answer_A"],
      row["answer_B"],
      row["answer_C"],
      row["answer_D"],
      row["answer_E"]
    )

    chat_response = client.chat.complete(
      model = model,
      messages = [
          {"role": "system","content": system_instruction+system_instruction_2+reference_examples},
          {"role": "user","content": prompt},
      ],
      temperature=0
   )
    answers.append(
        chat_response.choices[0].message.content
    )

# output format is a 2-columns dataframe with exactly 103 rows
output_df = pd.DataFrame(answers, columns=["Answer"])
output_df.index.name = "id"

output_df.to_csv(f'output.csv')
