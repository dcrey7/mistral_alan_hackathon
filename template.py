import os
import time
import pandas as pd
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
questions_file = "data/questions.csv"

df = pd.read_csv(questions_file)

question_prompt = lambda body, possible_answer_a, possible_answer_b, possible_answer_c, possible_answer_d, possible_answer_e: (
    "Répondez à la question suivante avec les lettres de la bonne réponse. Chaque question peut avoir plusieurs réponses correctes."
    "Il y a plusieurs types de questions"
    "Si "
    "Votre réponse doit contenir la lettre des réponses, séparées par des virgules, sans aucun espace. Un exemple de réponse idéale est : 'A,B'."
    "Vous ne fournissez aucun raisonnement, aucune intuition ni explication de votre réponse. Vous devez simplement donner la réponse conformément aux instructions données."
    "Vous ne donnez que les lettres demandées, par exemple 'A,B,C' ou 'C'. Votre réponse est toujours classée par ordre alphabétique. Vous ne devez pas mettre les lettres dans un ordre différent."
    f"{body}\n"
    f"A: {possible_answer_a}\n"
    f"B: {possible_answer_b}\n"
    f"C: {possible_answer_c}\n"
    f"D: {possible_answer_d}\n"
    f"E: {possible_answer_e}\n"
)

answers = []
client = Mistral(api_key=api_key)

for row_idx, row in df.iterrows():
    print(row_idx)
    time.sleep(3)
    prompt = question_prompt(
        row["question"],
        row["answer_A"],
        row["answer_B"],
        row["answer_C"],
        row["answer_D"],
        row["answer_E"],
    )

    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    answers.append(chat_response.choices[0].message.content)

# output format is a 2-columns dataframe with exactly 103 rows
output_df = pd.DataFrame(answers, columns=["Answer"])
output_df.index.name = "id"

output_df.to_csv("output.csv")
