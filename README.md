# 🧠 Hamlet Next Word Predictor  

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)
![Docker](https://img.shields.io/badge/Dockerized-Yes-blue)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-green)

This project is a **next-word prediction app** built on Shakespeare’s *Hamlet*. It uses an **LSTM-based Recurrent Neural Network** trained on classical English literature. The goal is to take a user input phrase and predict the **most probable next word**, shown through a clean **Streamlit interface**.

The app is **Dockerized** for reproducibility and includes a **CI/CD pipeline via GitHub Actions** to automatically build and deploy the Docker container.

---

## 🚀 Live Links

- 🌐 **Streamlit App**: [👉 Try it here](https://your-streamlit-app-link.com)
- 🐋 **DockerHub**: [👉 Docker Image] https://hub.docker.com/r/yash43256/next-word-predictor

---

## 🎯 Features

- ✅ Trained on `shakespeare-hamlet.txt` from the NLTK Gutenberg corpus
- ✅ Tokenization with `TextVectorization`
- ✅ LSTM model trained for 100 epochs (accuracy ~60%)
- ✅ Interactive frontend with Streamlit
- ✅ Dockerized and CI/CD enabled with GitHub Actions

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Libraries**: TensorFlow, Keras, NLTK, Streamlit, NumPy
- **Model**: LSTM-based next-word predictor
- **Deployment**: Docker
- **Automation**: GitHub Actions

---

## 📁 Project Structure

```
📦 Next_Word_Predictor/
├── .github/workflows/
│   └── main.yml               # GitHub Actions for CI/CD
│
├── data/
│   └── hamlet.txt             # Raw Shakespeare text
│
├── models/
│   ├── lstm_model.keras       # Trained LSTM model
│   └── vectorizer.keras       # Text vectorizer layer
│
├── app.py                     # Streamlit interface
├── prediction.py              # Prediction logic
├── experiments.py             # Model training and experimentation
├── Dockerfile                 # Docker container config
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to exclude from Git
└── README.md                  # This file
```

---

## 💻 How to Run the App

### ✅ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ash234king/Next_Word_Predictor.git
cd Next_Word_Predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

## 🧠 How It Works

1. **Data Loading** – Load Hamlet using the NLTK Gutenberg corpus.
2. **Preprocessing** – Lowercase and tokenize with Keras `TextVectorization`.
3. **Input Sequences** – Create n-gram sequences for next-word prediction.
4. **Model Architecture**:
   - Embedding layer
   - LSTM (150 units)
   - Dropout
   - LSTM (100 units)
   - Dense output with softmax over vocab
5. **Training** – Trained for 140 epochs for ~60% validation accuracy.
6. **Prediction** – `prediction function` defined in app.py only which processes user input and predicts next word.
7. **Streamlit UI** – `app.py` creates a simple UI to interact with the model.
8. **Docker** – All dependencies are containerized.
9. **CI/CD** – GitHub Actions (`main.yml`) automates Docker builds and deployment.

---

## 📝 Note on Design Decisions

> ⚠️ **Important Note**  
> During experimentation, we tried techniques like `TopKCategoricalSampling`,`labelsmoothening` and `ReduceLROnPlateau`.  
> However, we **chose not to include them** in the final model for the following reasons:
> 
> - **TopKCategoricalSampling**: This method introduced randomness into predictions and often resulted in words that **did not match the sentence context**, reducing the Shakespearean fluency.
> - **ReduceLROnPlateau**: This learning rate scheduler caused the model to focus on **less frequent patterns** or **overfit rare phrases**, making predictions worse.
> 
> ✅ Instead, we trained the model for a fixed **100 epochs** with a stable learning rate.  
> ✅ This slightly overfitted model actually gave **more grammatically and contextually accurate predictions**, which was the primary goal.

---

## 📈 Model Info

- Epochs: 100
- Accuracy: ~60%
- Vocabulary: 5,000 tokens (from Keras vectorizer)
- Input: A text prompt
- Output: Next predicted word

---


## ⚙️ GitHub Actions

A GitHub Actions workflow is defined in:

```
.github/workflows/main.yml
```

On every push or PR:
- 🧪 Run tests
- 🐋 Build Docker image
- 🚀 (Optional) Push image to DockerHub or deploy to cloud

---

## 📜 License

This project is released under the MIT License. Feel free to use, modify, or share with credit.

---

## 🙋‍♂️ Author

**Yashvardhan Singh**  
🎓 B.Tech Mechanical @ IIT Goa  
🔗 [LinkedIn] https://www.linkedin.com/in/yashvardhan-singh-26158028a/ 
💻 [GitHub] https://github.com/ash234king

