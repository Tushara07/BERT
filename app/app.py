import streamlit as st
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Page Config
st.set_page_config(
    page_title="Language Identifier",
    page_icon="🌍",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
.main .block-container {
    max-width: 700px;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Session State
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Clear Function
def clear_text():
    st.session_state.input_text = ""

# Load Model
@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
    "Tushara07/language-identifier-bert"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
    "Tushara07/language-identifier-bert"
    )

    return tokenizer, model


tokenizer, model = load_model()

mapping = pd.read_csv(
    "label_mapping.csv"
)

# UI
st.title("Language Identification System")

st.caption(
    "Detect the language of any text using a fine-tuned BERT model."
)

text = st.text_area(
    "Enter Text",
    height=80,
    placeholder="Type or paste text here...",
    key="input_text"
)

col1, col2, col3 = st.columns([1.5, 1, 4])

detect = col1.button(
    "Detect",
    use_container_width=True
)

col2.button(
    "Clear",
    on_click=clear_text,
    use_container_width=True
)

if detect:

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = (
            probs[0][prediction].item()
            * 100
        )

        language = mapping.loc[
            mapping["Label"] == prediction,
            "Language"
        ].values[0]

        st.success(
            f"Detected Language: {language}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        #st.progress(confidence / 100)