import pandas as pd

from unsloth import FastLanguageModel

from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer
import transformers
import torch


questions_file = "data/questions.csv"

df = pd.read_csv(questions_file)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    # token = "hf...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
    mapping={
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    },  # ShareGPT style
)
FastLanguageModel.for_inference(model)  


def apply_unsloth_model(prompt):

    messages = [
        # EDIT HERE!
        {"from": "human", "value": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    generated_text = model.generate(
        input_ids=inputs, streamer=text_streamer, max_new_tokens=1024, use_cache=True
    )

    decoded_output = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("\n\n")[-1]
    return decoded_output

def apply_huggingface_model(prompt):


    model1 = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
    messages = [{"role": "user", "content": prompt}]

    tokenizer1 = AutoTokenizer.from_pretrained(model1, chat_template="default")
    prompt = tokenizer1.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model1,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    import pdb ; pdb.set_trace()

    outputs = pipeline(prompt, max_new_tokens=50, do_sample=True, temperature=0.01)
    return outputs[0]["generated_text"]


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


def run_model(model="unsloth", prompt_func=question_prompt):
    if model == "unsloth":
        func = apply_unsloth_model

    elif model=="huggingface":
        func = apply_huggingface_model
    else:
        raise(NotImplementedError())
    

    answers = []

    for row_idx, row in df.iterrows():
        # print(row_idx)
        prompt = prompt_func(
            row["question"],
            row["answer_A"],
            row["answer_B"],
            row["answer_C"],
            row["answer_D"],
            row["answer_E"],
        )


        # import pdb; pdb.set_trace()


        answers.append(func(prompt))

    # answers.append(chat_response.choices[0].message.content)
# 
# output format is a 2-columns dataframe with exactly 103 rows
    output_df = pd.DataFrame(answers, columns=["Answer"])
    output_df.index.name = "id"

    return output_df


if __name__ == "__main__":
    df = run_model(model="huggingface", prompt_func=question_prompt)

    df.to_csv("output1.csv")


# from transformers import AutoTokenizer
# import transformers
# import torch

# model = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
# messages = [{"role": "user", "content": "What is a large language model?"}]

# tokenizer = AutoTokenizer.from_pretrained(model)
# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# outputs = pipeline(prompt, max_new_tokens=50, do_sample=True, temperature=0)
# print(outputs[0]["generated_text"])