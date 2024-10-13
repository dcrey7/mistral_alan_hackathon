import pandas as pd
import time 
from datasets import load_dataset
from retrying import retry 
from mistralai import Mistral

hf_token = ''
api_key = 'TvlAaEIaJYB08ufulDTk4jiRj4HXGQGG' # your API key

questions_file = 'data/questions_en.csv'  # path to the questions file
df = pd.read_csv(questions_file)


instruction = """From a question or a statement and answer, tell me if the answer is True or False. Answer must be only True or False, no more words."""

@retry(stop_max_attempt_number=7, wait_fixed=3000)
def question_true_false(instruction, question, possible_answer):
    prompt = f"""{instruction}\n
    Question: {question} \n
    Answer: {possible_answer}
    """

    chat_response = client.chat.complete(
      model = "mistral-large-latest",
      messages = [
          {
              "role": "user",
              "content": prompt
          },
      ],
      temperature=0
   )
    print('T/F Unitaire', chat_response.choices[0].message.content)
    
    return chat_response.choices[0].message.content


def true_answers(instruction, question, answer_a, answer_b, answer_c, answer_d, answer_e):
    true_answers = ''
    answers = {'A':answer_a, 'B':answer_b, 'C':answer_c, 'D':answer_d, 'E':answer_e}
    for answer_key, answer_value in answers.items():
        true_false_answer = question_true_false(instruction, question, answer_value)
        if true_false_answer == 'True':
            true_answers += answer_key

    return true_answers 


answers = []
client = Mistral(api_key=api_key)

for row_idx, row in df.iterrows():
    answer = true_answers(instruction,
      row["question"],
      row["answer_A"],
      row["answer_B"],
      row["answer_C"],
      row["answer_D"],
      row["answer_E"]
    )
    print("Current answer ", answer)
    answers.append(answer)


# output format is a 2-columns dataframe with exactly 103 rows
output_df = pd.DataFrame(answers, columns=["Answer"])
output_df.index.name = "id"

output_df.to_csv("output_TF_en.csv")