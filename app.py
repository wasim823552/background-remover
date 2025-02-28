from flask import Flask, request, send_file
import numpy as np
from PIL import Image
import io
import torch
from torchvision import transforms
from model import BackgroundRemovalModel  # Assume you have a pre-trained model

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    # Load your pre-trained model
    model = BackgroundRemovalModel()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Post-process the output
    output_image = transforms.ToPILImage()(output.squeeze().cpu())

    # Save the output image to a bytes buffer
    buf = io.BytesIO()
    output_image.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
