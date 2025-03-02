from flask import Flask, request, send_file
from rembg import remove
from PIL import Image
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/remove-background', methods=['POST'])
def remove_background():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    input_path = 'temp_input.png'
    output_path = 'temp_output.png'

    file.save(input_path)

    with open(input_path, 'rb') as input_file:
        input_image = input_file.read()

    output_image = remove(input_image)

    with open(output_path, 'wb') as output_file:
        output_file.write(output_image)

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
