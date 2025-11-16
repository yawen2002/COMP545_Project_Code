from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    questions = [
        "Jack used a spud bar to break the ice near the dock. Is Jack having a conversation or is he silent?",
        "Lina tripped and spilled the beans across the kitchen floor. Is this statement about food or a secret?",
        "Sora hit the nail on the head with a hammer. Is this statement about carpentry or accuracy?",
        "To watch the parade, Ilya sat on the fence by Main Street. Is the statement about location or indecision?",
        "Rui drew a line in the sand to mark the volleyball court. Did Rui agree or disagree to play volleyball?",
        "The veterinarian let the cat out of the bag. Is this statement about an animal or a secret?",
        "The acrobat hit the hay while landing after her stunt. Is the acrobat asleep or awake?",
        "The campers added fuel to the fire when it began to die down. Are the campers in conflict or working together?",
        "Jane dyed her hair black and blue. Is Jane hurt or unharmed?",
        "Jane's keys were flushed down the drain Are Jane's keys wet or dry?",
        "Jane broke Alex's heart when she took it out of the kiln. Did Alex and Jane have a platonic or romantic relationship?",
        "The parasite gets under your skin. Is the parasite inside the person?",
        "We found the cause of Tom's allergic reaction; it was the icing on the cake. Did Tom have a good time or bad time?",
        "There were two races; in the long run, Andy held a steady pace. Is Andy a sportsman or a career guy?",
    ]
    suffix = " Answer using one word only."
    max_answer_length = 5

    tokenizer, model = load_model(model_name)
    for question in questions:
        print("\nYou:", question + suffix)
        response = ask(model, tokenizer, question + suffix, max_answer_length)
        print("\nModel:", response, "\n")
