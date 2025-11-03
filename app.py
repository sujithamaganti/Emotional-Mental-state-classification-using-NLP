# app.py
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# 1️⃣ Load the trained GRU model (saved as .h5, not .pkl)
MODEL_PATH = "model_fixed (1).keras"  # 👈 Change this to your actual model file name
model = load_model('model_fixed (1).keras', safe_mode=False)

print("✅ Deep Learning GRU model loaded successfully!")

# 2️⃣ Load SentenceTransformer (same one used in training)
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Sentence Transformer loaded successfully!")

# 3️⃣ Define emotion labels (must match your training order)
class_names = [
    'Neutral / Mixed feelings',
    'Mild positive / reflective emotion',
    'Romantic / affectionate feeling',
    'Confident / content / expressive',
    'Sadness / depression / helplessness',
    'Fatigue / exhaustion / burnout'
]

# 4️⃣ Home route
@app.route('/')
def home():
    if request.method == 'POST':
        # Your prediction logic goes here
        text = request.form['text']
        # ... predict and get result ...
        
        # Then re-render the template with the result
        return render_template('index.html', input_text=text, prediction_text=prediction_result)

    # Handles GET request (initial page load)
    return render_template('index.html', input_text='', prediction_text=None)

# 5️⃣ Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         input_text = request.form['text']
#         if not input_text.strip():
#             return render_template('index.html', prediction_text="Please enter some text.")

#         # Step 1 — Encode user input into 384-dim embedding
#         embedding = encoder.encode([input_text])
#         embedding = np.expand_dims(embedding, axis=1)  # shape: (1, 1, 384)

#         # Step 2 — Predict using trained GRU model
#         prediction = model.predict(embedding)
#         predicted_index = np.argmax(prediction)
#         predicted_label = class_names[predicted_index]

#         return render_template('index.html', prediction_text=f"Predicted Emotion: {predicted_label}")
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    if not input_text.strip():
        return render_template('index.html', prediction_text="Please enter some text.")

    embedding = encoder.encode([input_text])
    embedding = np.expand_dims(embedding, axis=1)
    prediction = model.predict(embedding)[0]

    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = round(prediction[predicted_index] * 100, 1)

    probabilities = {class_names[i].split('/')[0].strip(): round(float(prediction[i]*100), 1) for i in range(len(class_names))}

    return render_template(
        'index.html',
        input_text=input_text,
        emotion_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
        prediction_text=True
    )


if __name__ == '__main__':
    app.run(debug=True)