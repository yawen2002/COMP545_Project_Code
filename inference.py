from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import re
import sys

def load_model(model_name: str):
    """
    Load a Hugging Face causal language model + tokenizer.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    return tokenizer, model

def ask(model, tokenizer, question: str, max_new_tokens=100):
    """
    Generate a response from the model.
    """
    prompt = f"Q: {question}\nA: "

    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
        # do_sample=True,
        # top_p=0.9,
        # temperature=0.8
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):]  # returns only the answer

def contains(answer, words):
    answer_set = set(re.findall(r"[\w']+", answer.lower()))
    word_set = set(re.findall(r"[\w']+", words.lower()))
    return len(answer_set.intersection(word_set)) > 0


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    suffix = "Answer using one word only."
    max_answer_length = 10
    testing = False

    tokenizer, model = load_model(model_name)
    df = pd.read_csv("./dataset.csv")

    if testing:
        (context, question, correct_words, wrong_words) = df.iloc[0]
        prompt = " ".join([context, question, suffix])
        print(prompt)
        print(ask(model, tokenizer, prompt, max_answer_length))
        sys.exit()

    n = df.shape[0]
    ambiguous = []
    nb_correct = 0
    for _, (context, question, correct_words, wrong_words) in tqdm(df.iterrows(), total=n):
        prompt = " ".join([context, question, suffix])
        answer = ask(model, tokenizer, prompt, max_answer_length)
        if contains(answer, correct_words) and contains(answer, wrong_words) or (not contains(answer, correct_words) and not contains(answer, wrong_words)):
            ambiguous.append((prompt, answer))
        elif contains(answer, correct_words):
            nb_correct += 1
    
    print("--- RESULTS ---")
    print(f"{nb_correct}/{n} correct")
    print(f"{n - (nb_correct + len(ambiguous))}/{n} wrong")
    print(f"{len(ambiguous)}/{n} ambiguous")

    if len(ambiguous) > 0:
        print("\nAmbiguous answers:")
        print(ambiguous)
