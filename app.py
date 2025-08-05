import flask
import os
import model_file
import folium
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import cv2  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #16MB max file size

#creating uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok = True)

model = model_file.create_model()
model.load_weights('wildfire.weights.h5')

wildfire_locations = []

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_map():
    # Create a map centered at the first location or default to (0,0)
    center = wildfire_locations[0] if wildfire_locations else [0, 0]
    m = folium.Map(location=center, zoom_start=13)
    
    # Add markers for all wildfire locations
    for i, (lat, lon) in enumerate(wildfire_locations):
        folium.Marker(
            [lat, lon],
            popup=f'Wildfire #{i+1}',
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(m)
    
    # Save the map to a temporary file
    map_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wildfire_map.html')
    m.save(map_path)
    return map_path

@app.route('/get_map')
def get_map():
    map_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wildfire_map.html')
    return send_file(map_path)

@app.route('/get_locations')
def get_locations():
    return jsonify({
        'locations': wildfire_locations,
        'count': len(wildfire_locations)
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
       return jsonify({'error': 'No file selected'}), 400 

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img = cv2.imread(filepath)
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = img.reshape(1,32,32,3)
            imgs =[]
            imgs.append(img)
            print("img", imgs)

            prediction = model.predict(img)
            result = bool(prediction[0][0] > 0.5)
            confidence = float(prediction [0][0])
            if result == False:
                confidence = 1 - confidence

            os.remove(filepath)

            print(result, confidence)

            return jsonify({
                'prediction': result,
                'confidence': confidence
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    data = request.get_json()
    latitude = float(data.get('latitude'))
    longitude = float(data.get('longitude'))
   
    try:
        # Add new location to in-memory storage
        wildfire_locations.append([latitude, longitude])
        
        # Update the map with all locations
        map_path = create_map()
        return jsonify({
            'success': True,
            'map_url': '/get_map',
            'location_count': len(wildfire_locations)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)