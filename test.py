from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "models/language_model"
)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "models/language_model"
)

print("Model Loaded Successfully!")