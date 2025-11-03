# 🧠 Emotion/Mental State Classification Using NLP

## 📌 Overview

This project focuses on automating the classification of emotional and mental states expressed in text using **Natural Language Processing (NLP)** and **Deep Learning**. The model identifies emotions such as **neutral, positive, romantic, confident, sadness, and fatigue**, helping in areas like **sentiment analysis, mental health monitoring, and social research**.

---

## 🎯 Objective

To develop an AI model that accurately classifies text-based emotions using advanced NLP techniques and deep learning models.

---

## 📂 Dataset

* **Total records:** 416,809
* **Columns:** 3 (Text, Label, and ID)
* **Labels:**

  * `0`: Neutral / Mixed feelings
  * `1`: Mild positive / Reflective emotion
  * `2`: Romantic / Affectionate feeling
  * `3`: Confident / Content / Expressive
  * `4`: Sadness / Depression / Helplessness
  * `5`: Fatigue / Exhaustion / Burnout

---

## ⚙️ Data Preprocessing

* Text cleaning not required (dataset already processed).
* Used **Sentence Transformers** to convert text into embeddings.
* Train-test split for evaluation.
* Tokenization and lemmatization handled within the embedding model.

---

## 🧠 Model Architecture

Multiple NLP models were tested:

* Logistic Regression
* Random Forest
* LSTM / BiLSTM
* Transformer-based (BERT)
* **GRU (selected for final implementation)**

The final **4-layer GRU model** achieved the best accuracy (~78%) and produced reliable emotion predictions.

**GRU Layers:**

1. GRU Layer
2. Critical Layer
3. Hidden Dense Layer
4. Output Layer

---

## 📊 Evaluation Metrics

* **Accuracy:** 78%
* **Precision, Recall, F1-Score**
* **Confusion Matrix** for visualizing misclassifications

Key insights:

* Neutral and Confident emotions showed higher precision.
* Overlap between Sadness and Fatigue was the main challenge.

---

## 🚀 Deployment

The model was deployed using **Flask** for real-time inference and testing.

---

## 🧩 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **scikit-learn**
* **Sentence Transformers**
* **Matplotlib / Seaborn**
* **Google Studio**

---

## 📁 Repository Structure

```
emotion-classification-nlp/
│
├── data/
│   └── emotions_dataset.csv
│
├── notebooks/
│   └── emotion_classification.ipynb
│
├── models/
│   └── gru_model.h5
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── deployment_script.py
│
├── results/
│   ├── confusion_matrix.png
│   └── training_validation_loss.png
│
├── requirements.txt
└── README.md
```

---

## 📈 Future Improvements

* Use more diverse and balanced datasets.
* Experiment with **transformer-based models** (e.g., BERT, RoBERTa).
* Integrate context-aware sentiment tracking for better accuracy.

---

## 👩‍💻 Author

**Sujitha Maganti**
Bachelor’s in Artificial Intelligence and Data Science
📧 magantisujitha1@example.com
