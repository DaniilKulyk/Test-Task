import pandas as pd
import ast
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans
import os

# Load dataset
df_m = pd.read_csv('mountain_dataset_with_markup.csv')

# Convert the string representation of lists into real lists
df_m['marker'] = df_m['marker'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Filter rows where 'marker' is not empty (the list contains items)
filtered_df = df_m[df_m['marker'].apply(lambda x: len(x) > 0)]
filtered_df.loc[:, 'marker'] = filtered_df['marker'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

# Prepare training data
training_data = {'class' : ["MOUNTAIN_NAME"], 'annotations' : []}

for _, row in filtered_df.iterrows():
    temp_dict = {'text': row['text'], 'entities': []}
    start = row['marker'][0]
    end = row['marker'][1]
    temp_dict['entities'].append((start, end, "MOUNTAIN_NAME"))
    training_data['annotations'].append(temp_dict)

# Create spaCy model and prepare DocBin for training
nlp = spacy.blank("en")
doc_bin = DocBin()

for training_example in tqdm(training_data['annotations']):
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

# Save training data
doc_bin.to_disk("training_data.spacy")

# Create spaCy config
os.system("python -m spacy init fill-config base_config.cfg config.cfg")

# Train model
os.system("python -m spacy train config.cfg --output ./ --paths.train ./training_data.spacy --paths.dev ./training_data.spacy --gpu-id 0")

