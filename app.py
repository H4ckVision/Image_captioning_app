from flask import Flask, request, render_template
import os
import numpy as np
import pickle
import uuid
from PIL import Image
from gtts import gTTS
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'

# Load tokenizer and model
tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
model = load_model('model/CNN_LSTM_model_30.keras')

# Load VGG16 for feature extraction
vgg = VGG16()
vgg = Model(vgg.inputs, vgg.layers[-2].output)

# Parameters
max_length = 35  # adjust to your actual value


# Feature extractor
def extract_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg.predict(image, verbose=0)
    return feature


# Caption generation (Greedy)
def predict_caption(model, image_feature, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = ' '.join(in_text.split()[1:-1])  # remove startseq and endseq
    return final_caption


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_path = None
    audio_path = None

    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded."

        file = request.files['file']
        if file.filename == '':
            return "No file selected."

        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs('static/audio', exist_ok=True)

            # Clean previous audio files
            for f in os.listdir('static/audio'):
                os.remove(os.path.join('static/audio', f))

            # Save uploaded image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_path = filepath.replace("\\", "/")

            # Generate caption
            features = extract_features(filepath)
            caption = predict_caption(model, features, tokenizer, max_length)

            # Generate audio using gTTS
            try:
                tts = gTTS(text=caption, lang='en')
                audio_filename = f"{uuid.uuid4()}.mp3"
                audio_path = f"static/audio/{audio_filename}"
                tts.save(audio_path)
                audio_path = audio_path.replace("\\", "/")
            except Exception as e:
                print(f"[ERROR] Audio generation failed: {e}")
                audio_path = None

    return render_template("index.html", caption=caption, image_path=image_path, audio_path=audio_path)


if __name__ == "__main__":
    app.run(debug=True)
