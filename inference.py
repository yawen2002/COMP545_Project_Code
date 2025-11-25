from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import re

def main():
    # ----- CHANGE THESE PARAMETERS -----
    model_name = "Qwen/Qwen3-8B"
    suffix = "Answer using one word only. Answer: "
    max_answer_tokens = 100
    question_words = "whether"  # To detect when the model just repeats the question instead of answering
    # -----------------------------------

    tokenizer, model = load_model(model_name)
    df = pd.read_csv("./dataset.csv")
    n = df.shape[0]
    ambiguous = []
    nb_correct = 0
    output = pd.DataFrame(columns=["Correct", "Question", "Answer"])
    for i, (context, question, correct_words, wrong_words) in tqdm(df.iterrows(), total=n):
        prompt = " ".join([context, question, suffix])
        extended_prompt = prompt
        answer = ""
        nb_tokens = 0
        is_correct = is_wrong = is_ambiguous = False
        while nb_tokens < max_answer_tokens and not is_correct and not is_wrong and not is_ambiguous:
            nb_tokens += 1
            answer_token = ask(model, tokenizer, extended_prompt, 1)
            answer += answer_token
            extended_prompt += answer_token
            if contains(answer, correct_words):
                is_correct = True
                nb_correct += 1
            elif contains(answer, wrong_words):
                is_wrong = True
            elif contains(answer, question_words):
                is_ambiguous = True

        if is_ambiguous or (not is_correct and not is_wrong):
            ambiguous.append((prompt, answer))
        output.loc[i] = {"Correct": is_correct, "Question": question, "Answer": answer}
        output.to_csv("output.csv", index=False)

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

def ask(model, tokenizer, prompt: str, max_new_tokens=100) -> str:
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

def contains(answer: str, words: str) -> bool:
    """
    Test if an answer string contains any word from another string.
    """
    answer_set = set(re.findall(r"[\w']+", answer.lower()))
    word_set = set(re.findall(r"[\w']+", words.lower()))
    return len(answer_set.intersection(word_set)) > 0


if __name__ == "__main__":
    main()