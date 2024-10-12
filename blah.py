import pandas as pd
import re

from tqdm import tqdm

# from unsloth import FastLanguageModel

from transformers import TextStreamer
# from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer
import transformers
import torch


questions_file = "data/questions_en.csv"

combinations = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "A,B",
    "A,C",
    "A,D",
    "A,E",
    "B,C",
    "B,D",
    "B,E",
    "C,D",
    "C,E",
    "D,E",
    "A,B,C",
    "A,B,D",
    "A,B,E",
    "A,C,D",
    "A,C,E",
    "A,D,E",
    "B,C,D",
    "B,C,E",
    "B,D,E",
    "C,D,E",
    "A,B,C,D",
    "A,B,C,E",
    "A,B,D,E",
    "A,C,D,E",
    "B,C,D,E",
    "A,B,C,D,E"
]


df = pd.read_csv(questions_file)


# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/Llama-3.2-3B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
#     max_seq_length=2048,
#     dtype=None,
#     load_in_4bit=True,
#     # token = "hf...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="llama-3.1",
#     mapping={
#         "role": "from",
#         "content": "value",
#         "user": "human",
#         "assistant": "gpt",
#     },  # ShareGPT style
# )
# FastLanguageModel.for_inference(model)  


model1 = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"

tokenizer1 = AutoTokenizer.from_pretrained(model1)
pipeline = transformers.pipeline("text-generation",model=model1,torch_dtype=torch.float16,device_map="auto")


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


    messages = [{"role": "user", "content": prompt}]


    new_prompt = tokenizer1.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipeline(new_prompt, max_new_tokens=256, do_sample=True,  temperature=0.7, top_k=50, top_p=0.95)
    return_ =  outputs[0]["generated_text"]

    # Regex to find the pattern "A,C,D,E"
    match = re.findall(r"\b[A-E](?:,[A-E])*\b", return_)
    return match[-1]




question_prompt = lambda body, possible_answer_a, possible_answer_b, possible_answer_c, possible_answer_d, possible_answer_e: (


    f"""
    # Task #
    Your task is to answer a multiple-choice quizz in medicine

    # Context #

    You are given a question and 5 possible answers. At least one is correct, but all can be. 
    Questions can be any of the following question types:

    - situation. You're a doctor and you are given a situation, you have to say how you would act
    - direct question. You're an encyclopedia and you're asked specific details on a disease
    - description. It's different statements and you have to assess which ones are correct.

    # Output #

    You will give a reasoning as to each possible answer and why it belongs to the correct answers (A, B, C, D and E).
    In the end, you will end your reasoning by a simple answer : "My final answer is XXX"

    The XXX will be a comma-concatenated string of all true answers, in alphabetical order, e.g.

    "My final answer is A,C,E" or "My final answer is C" or "My final answer is A,B,C,E" etc.

    # Question #

    Question: {body}

    A: {possible_answer_a}
    B: {possible_answer_b}
    C: {possible_answer_c}
    D: {possible_answer_d}
    E: {possible_answer_e}

    """
)

#     "Answer the following question with the letters of the correct answer. Each question can have multiple answers that are right. "
#     "Your answer must contain the letter of the answers, separated by commas and without any space. An ideal output is like: 'A,B', for instance."
#     "You don't provide any reasoning nor intuition nor explanation of your answer. You merely output the answer as per the instructions you are given."
#     "You only output the letters that you are asked to provide, e.g. 'A,B,C' or 'C'. Your answer is always sorted alphabetically."
#     "Your answer can ONLY be one of {combinations}"
#     "I don't want the answers repeated, only the letters"

#     f"{body}\n"
#     f"A: {possible_answer_a}\n"
#     f"B: {possible_answer_b}\n"
#     f"C: {possible_answer_c}\n"
#     f"D: {possible_answer_d}\n"
#     f"E: {possible_answer_e}\n"
# )


def run_model(model="unsloth", prompt_func=question_prompt):
    if model == "unsloth":
        func = apply_unsloth_model

    elif model=="huggingface":
        func = apply_huggingface_model
    else:
        raise(NotImplementedError())
    

    answers = []

    for row_idx, row in tqdm(df.iterrows()):
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