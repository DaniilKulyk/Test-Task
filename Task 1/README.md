# Task 1. Natural Language Processing. Named entity recognition

## Overview

The goal of the task is to recognize the names of mountains in texts using Named Entity Recognition (NER) models. After training the model, it should find and extract the found names from the raw text.

## Data

After a small search in the Internet was found a dataset on [Kaggle](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset/data), which contains sentences with labeled names of mountains.

## Solution

To solve this problem, the spaCy library was used, based on which one can create one's own NER model with the necessary labels, parameters and architen.
The solution process was as follows:
1. To train the model, the original dataset was processed for problematic characters/values. 
2. The resulting dataset was converted to the format required for training by the library and a new class “MOUNTAIN_NAME” was created
3. A config file was created with all necessary parameters, including the required architecture based on the “roBERTa-Based” model.
4. Trained the model and saved the [best version](https://drive.google.com/drive/folders/1LpS08LlcjQAoGtePH7cQ7Ankeso5QuG2?usp=sharing) of it.

## Improvements

1. Increase the amount of information for training. For ML models, in most cases, increasing the amount of data, including in a "complex form" (in a specific example, this could be several names of mountains that are duplicated or intersected) increases the subsequent accuracy of predicting results.
2. Check other types of NER model architecture
3. Experiment with different training parameters (number of training epochs, batch_size, max_batch_items, max_steps, eval_frequency, etc).

## Working environment
The DataSpell IDE with Python 3.11 was used. All libraries used and their versions are listed in “requirements.txt”.
