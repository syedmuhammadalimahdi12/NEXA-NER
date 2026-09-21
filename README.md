# 🧠 GRU NER Intelligence

### Named Entity Recognition using a Bidirectional GRU and Streamlit

A professional **AI-powered Named Entity Recognition (NER)** application built using a **Bidirectional GRU neural network**, **TensorFlow/Keras**, and **Streamlit**.

The system analyzes natural language text and identifies important entities such as:

- 🟢 **PERSON** — Names of people
- 🔵 **LOCATION** — Cities, countries, and places
- 🟠 **ORGANIZATION** — Companies, institutions, and organizations

The project also provides an interactive and professional web-based dashboard for model analysis, entity visualization, confidence analysis, and model performance evaluation.

---

## ✨ Features

### 🔎 Intelligent Text Analysis

Enter any English text and let the trained GRU model analyze it automatically.

The application:

- 📝 Accepts custom text
- 🔤 Tokenizes the input
- 🔢 Converts words into numerical IDs
- 🧠 Processes sequences using a Bidirectional GRU
- 🏷️ Predicts NER labels
- 🎨 Highlights detected entities
- 📊 Displays entity confidence
- 🔬 Provides token-level analysis

---

## 🎯 Supported Entity Types

| Entity | Description | Example |
|---|---|---|
| 🟢 PERSON | Names of people | Bill Gates |
| 🔵 LOCATION | Places and geographical locations | London |
| 🟠 ORGANIZATION | Companies and institutions | Microsoft |

---

## 🖥️ Professional Streamlit Dashboard

The application contains multiple interactive sections.

### 📊 Dashboard

The dashboard provides an overview of the trained model, including:

- Model architecture
- NER task information
- Sequence length
- Embedding dimensions
- Entity types
- Accuracy
- Precision
- Recall
- F1 Score
- NER processing pipeline

---

### 🔎 Text Analyzer

The Text Analyzer allows users to enter or paste text and perform real-time NER prediction.

It provides:

- Character count
- Word count
- Sentence count
- Entity highlighting
- Entity count
- Confidence scores
- Detected entity table
- Token-level analysis

Example:

```text
Bill Gates founded Microsoft in the United States.

Possible output:

🟢 Bill Gates
🟠 Microsoft
🔵 United States
📈 Entity Analytics

The application provides visual analytics for:

PERSON
LOCATION
ORGANIZATION

It includes:

📊 Precision comparison
📊 Recall comparison
📊 F1 Score comparison
🥧 Entity distribution
📈 Entity frequency visualization
🎯 Model Performance

The application displays the trained model's evaluation metrics.

Metric	Score
Accuracy	94.41%
Precision	72.65%
Recall	63.36%
F1 Score	67.69%

Note: Accuracy is calculated at the token level, while Precision, Recall, and F1 are entity-level NER metrics. Therefore, these values measure different aspects of model performance.

🧠 Model Architecture

The NER model uses the following architecture:

Input Tokens
     ↓
Embedding Layer
     ↓
Bidirectional GRU
     ↓
Dropout
     ↓
Dense + Softmax
     ↓
NER Labels
Model Configuration
Component	Configuration
Architecture	Bidirectional GRU
Embedding Dimension	128
GRU Units	128
Sequence Length	128
Dropout	0.30
Optimizer	Adam
Framework	TensorFlow / Keras
Task	Named Entity Recognition
🏷️ NER Label Structure

The model uses BIO-style labels.

Label	Meaning
O	Outside an entity
B-PER	Beginning of a person entity
I-PER	Inside a person entity
B-ORG	Beginning of an organization
I-ORG	Inside an organization
B-LOC	Beginning of a location
I-LOC	Inside a location

For example:

Bill       → B-PER
Gates      → I-PER
founded    → O
Microsoft  → B-ORG
📊 Classification Performance

The model achieved the following entity-level results:

Entity	Precision	Recall	F1 Score
🟢 PERSON	70%	55%	62%
🔵 LOCATION	80%	74%	77%
🟠 ORGANIZATION	67%	61%	64%
Overall
Precision: 72.65%
Recall: 63.36%
F1 Score: 67.69%
🛠️ Technologies Used
Programming Language

🐍 Python

Machine Learning
🧠 TensorFlow
Keras
Bidirectional GRU
Embedding Layer
Softmax Classification
Data Processing
NumPy
Pandas
Regular Expressions
Pickle
Visualization
📊 Plotly
Streamlit Charts
User Interface
🎨 Streamlit
Responsive dashboard
Interactive components
Light professional theme
📂 Project Structure
GRU-NER-Streamlit/
│
├── app.py
│
├── gru_ner_model.keras
│
├── word2id.pkl
│
├── id2label.pkl
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
⚙️ Installation
1️⃣ Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
2️⃣ Navigate to the project
cd GRU-NER-Streamlit
3️⃣ Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate
4️⃣ Install dependencies
pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application using:

streamlit run app.py

The application will open in your browser.

Usually:

http://localhost:8501
🧪 Example Input

Try this text inside the Text Analyzer:

Bill Gates founded Microsoft in the United States and later visited Paris. Elon Musk announced that SpaceX would continue developing new spacecraft in Texas. NASA is also working on future missions to the Moon and Mars.

The application will attempt to identify:

🟢 PERSON
Bill Gates
Elon Musk

🔵 LOCATION
United States
Paris
Texas
Moon
Mars

🟠 ORGANIZATION
Microsoft
SpaceX
NASA
🔬 How the System Works

The overall workflow is:

                    USER TEXT
                       │
                       ▼
                📝 Text Input
                       │
                       ▼
                ✂️ Tokenization
                       │
                       ▼
                🔢 Word Encoding
                       │
                       ▼
                📚 Embedding
                       │
                       ▼
                🧠 Bidirectional GRU
                       │
                       ▼
                    💧 Dropout
                       │
                       ▼
                🎯 Dense + Softmax
                       │
                       ▼
                  🏷️ NER Labels
                       │
                       ▼
              🔎 Entity Extraction
                       │
                       ▼
              🎨 Visual Highlighting
📌 Dataset

The model was trained using the CoNLL-2003 Named Entity Recognition dataset.

The original dataset contains multiple NER categories. For this project, the model focuses on:

PERSON
ORGANIZATION
LOCATION
O

The MISC category was mapped to the O class for this implementation.

📈 Model Evaluation

The model was evaluated using:

Test accuracy
Entity-level precision
Entity-level recall
Entity-level F1 score
Classification report

The model achieved approximately:

94.41% test token accuracy

and an overall:

67.69% entity-level F1 score

🎨 User Interface

The application uses a clean, modern, light-themed interface designed for:

🎓 Academic demonstrations
💻 AI/ML project presentations
🧪 Model testing
📊 Performance analysis
🚀 Portfolio demonstrations
🔐 Local Model Files

The application requires the following trained model files:

gru_ner_model.keras
word2id.pkl
id2label.pkl

These files should be located in the same directory as app.py.

🚀 Future Improvements

Possible future improvements include:

🔥 Transformer-based NER
🤗 BERT-based NER
🌍 Multilingual entity recognition
⚡ Faster inference
📱 Mobile-friendly interface
☁️ Cloud deployment
📦 Docker deployment
📚 Larger and more diverse datasets
🎯 Improved PERSON entity recall
🎓 Project Purpose

This project demonstrates how recurrent neural networks can be applied to Natural Language Processing for Named Entity Recognition.

It combines:

Machine Learning + Natural Language Processing + Deep Learning + Interactive Visualization

into a single practical application.

👨‍💻 Author

Syed Muhammad Ali Mahdi

🎓 BS Computer Science
🤖 Artificial Intelligence & Machine Learning
💻 Computer Vision | NLP | Generative AI

⭐ If You Like This Project

If you find this project useful or interesting:

⭐ Star the repository
🍴 Fork the repository
📢 Share the project
💡 Contribute improvements
