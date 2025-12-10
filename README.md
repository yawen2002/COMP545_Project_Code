# Minimal-Context Stress Tests of Figurative Reasoning in LLMs

This project was completed as part of the course COMP545 - Natural Language Understanding with Deep Learning.

Authors:
- Yawen Guo (Student)
- Balaji Ramesh (Student)
- Zarine Ardekani-Djoneidi (Student)
- Verna Dankers (Mentor)

## 1. Dataset Curation

We developed two scripts to curate a dataset of idioms sourced from MAGPIE.

### a) Calculate_Ratio.py

This script calculates the figurative ratio for a single idiom provided by the user at runtime.

$$ \text{Figurative Ratio} = \frac{\text{Number of idiomatic uses}}{\text{Number of idiomatic} + \text{literal uses}} $$

For example: `python3 Calculate_Ratio.py "on the same page"`

### b) Find_High_Figurative_Idioms.py

This script analyzes all idioms in the MAGPIE dataset to identify those that are used predominantly in a figurative sense. It applies several filters and then produces a ranked CSV file containing idioms and their corresponding figurative
ratios.

**Filtering:** Idioms are removed if:
1. they contain fewer than three words, or
2. they have zero literal occurrences

**Sorting:** The idioms are sorted in descending order by figurative ratio.

The output of this script is `high_figurative_idioms.csv`.

## 2. Inference

We implemented two ways of evaluating the models on our curated dataset of sentences with idiomatic expressions. Both approaches work on models from Hugging Face.

### a) Probability-Based Answer Extraction

In `Probability_Approach.ipynb`, we implemented a constrained likelihood evaluation. This approach restricts the output space to a pre-defined set of candidate answers (the correct targets and specific distractors) and selects the candidates with the highest conditional probabilities.

The results for each model are saved in csv files.

### b) Text-Based Answer Extraction

The script `Output_Text_Approach.py` allows models to freely generate text and classifies their responses by matching predefined answer keywords in the output.

The results are saved in csv files.

## 3. Analysis of Results

The notebook `Human_Study_Analysis.ipynb` compares LLM performance and human performance from the human study.

The three surveys used for our human study can be found here:
- [Survey 1](https://docs.google.com/forms/d/e/1FAIpQLSciv4ADmhH80rD9wEhQbjiyFOYEX_FoPhY7QEres4qN6sArWg/viewform?usp=dialog)
- [Survey 2](https://docs.google.com/forms/d/e/1FAIpQLSdeHRwGY1vnJPvINxAa0rQQFZh46v_sttJnkjO1CfL7DZnhFw/viewform?usp=dialog)
- [Survey 3](https://docs.google.com/forms/d/e/1FAIpQLSd6blMBsdbj4cCpFSGJKBLu9fnrU9F-OVkS8icbyjf-gKNBoA/viewform?usp=dialog)

The notebook `Models_Comparisons.ipynb` compares model performance across multiple dimensions, including with and without in-context learning, and using two different answer-extraction methods.
