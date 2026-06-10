import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Load model
tokenizer = AutoTokenizer.from_pretrained(
    "models/language_model"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "models/language_model"
)

# Load label mapping
mapping = pd.read_csv(
    "models/label_mapping.csv"
)

# Test text
text = "Hello everyone"

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

# Predict
with torch.no_grad():
    outputs = model(**inputs)

# Probabilities
probs = torch.softmax(
    outputs.logits,
    dim=1
)

prediction = torch.argmax(
    probs,
    dim=1
).item()

confidence = probs[0][prediction].item() * 100

# Convert label to language
language = mapping.loc[
    mapping["Label"] == prediction,
    "Language"
].values[0]

print("Language:", language)
print(f"Confidence: {confidence:.2f}%")