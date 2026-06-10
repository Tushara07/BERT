# Language Identification System Using BERT

## Overview

This project is a Language Identification System that predicts the language of a given text using a fine-tuned BERT model. The application is built with Streamlit and provides real-time language predictions through a simple web interface.

## Dataset

The model was trained using the Language Detection Dataset, which contains text samples from multiple languages. The dataset was used to fine-tune a pre-trained BERT model for multilingual language classification.

Dataset Source:
https://www.kaggle.com/datasets/basilb2s/language-detection

## Supported Languages

The model is trained to identify the following languages:

* English
* French
* Spanish
* Portuguese
* Italian
* Russian
* Swedish
* Dutch
* Danish
* Arabic
* Turkish
* German
* Tamil
* Hindi
* Malayalam
* Kannada
* Greek

The system predicts the most probable language for a given text input and returns the corresponding language label.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* Streamlit
* Pandas
* Scikit-learn

## Project Structure

```text
Language-Identification-System/
│
├── app/
│   └── app.py
├── notebooks/
│   └── bert.ipynb
├── predict.py
├── label_mapping.csv
├── requirements.txt
├── test.py
└── README.md
```

## Methodology

1. Load and preprocess the Language Detection dataset.
2. Encode language labels.
3. Tokenize text using the BERT tokenizer.
4. Fine-tune a pre-trained BERT model for sequence classification.
5. Save the trained model and tokenizer.
6. Deploy the model using Streamlit for real-time predictions.

## Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

## Results

The fine-tuned BERT model is capable of identifying the language of input text across multiple language categories with high accuracy.

## Sample Test Inputs

| Language   | Sample Text                                   |
| ---------- | --------------------------------------------- |
| English    | Hello, how are you doing today?               |
| French     | Bonjour, comment allez-vous aujourd'hui ?     |
| Spanish    | Hola, ¿cómo estás hoy?                        |
| Portuguese | Olá, como você está hoje?                     |
| Italian    | Ciao, come stai oggi?                         |
| German     | Hallo, wie geht es dir heute?                 |
| Dutch      | Hallo, hoe gaat het vandaag met je?           |
| Swedish    | Hej, hur mår du idag?                         |
| Danish     | Hej, hvordan har du det i dag?                |
| Russian    | Привет, как дела сегодня?                     |
| Greek      | Γεια σου, πώς είσαι σήμερα;                   |
| Turkish    | Merhaba, bugün nasılsın?                      |
| Arabic     | مرحباً، كيف حالك اليوم؟                       |
| Hindi      | नमस्ते, आप आज कैसे हैं?                       |
| Tamil      | வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்? |
| Kannada    | ನಮಸ್ಕಾರ, ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಿ?                |
| Malayalam  | നമസ്കാരം, ഇന്ന് നിങ്ങള്‍ക്ക് സുഖമാണോ?         |

These examples can be used to test the language prediction capability of the application after deployment.

