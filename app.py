from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('plant_classifier.h5')

class_names = {
    0: "Aloe Vera",
    1: "Banana",
    2: "Bilimbi",
    3: "Cantaloupe",
    4: "Casava",
    5: "Coconut",
    6:"Corn",
    7:"Cucumber",
    8:"Curcuma",
    9:"Egg Plant",
    10:"Galangal",
    11:"Ginger",
    12:"Guava",
    13:"Kale",
    14:"Longbeans",
    15:"Mango",
    16:"Melon",
    17:"Orange",
    18:"Paddy",
    19:"Pappaya",
    20:"Pepperchilli",
    21:"Pineapploe",
    22:"Pomelo",
    23:"Shallot",
    24:"Soy Beans",
    25:"Spinach",
    26:"Sweet potatoes",
    27:"Tobacco",
    28:"Waterapple",
    29:"Watermelon"
}

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_path = "uploads/" + file.filename
    file.save(img_path)
    
    img = prepare_image(img_path)
    prediction = model.predict(img)
    print(prediction,"prediction")
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(predicted_class)
    plant_name = class_names[predicted_class]
    
    return jsonify({'prediction': plant_name})

if __name__ == '__main__':
    app.run(debug=True)
