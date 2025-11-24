from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import re
import sys

def main():
    # ----- CHANGE THESE PARAMETERS -----
    model_name = "qwen/Qwen3-4B-Instruct-2507"
    suffix = "Answer using one word only."
    max_answer_length = 15
    testing = False
    # -----------------------------------

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

def load_model(model_name: str):
    """
    Load a Hugging Face causal language model + tokenizer.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model

def ask(model, tokenizer, prompt: str, max_new_tokens=100):
    """
    Generate a response from the model.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):]

def contains(answer, words):
    """
    Test if an answer string contains any word from another string.
    """
    answer_set = set(re.findall(r"[\w']+", answer.lower()))
    word_set = set(re.findall(r"[\w']+", words.lower()))
    return len(answer_set.intersection(word_set)) > 0


if __name__ == "__main__":
    main()