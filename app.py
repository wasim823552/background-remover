from flask import Flask, request, send_file
import io
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained background removal model (example: U^2-Net)
# Replace this with your actual model loading code
def load_model():
    # Example: Load a pre-trained U^2-Net model
    model = torch.hub.load('mateuszbuda/bg-removal', 'u2net')
    model.eval()
    return model

# Preprocess the image for the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize to model input size
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(           # Normalize
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image).unsqueeze(0)

# Remove background using the model
def remove_background(model, image):
    with torch.no_grad():
        output = model(image)
    output = output.squeeze().cpu().numpy()  # Convert to numpy array
    return output

# Enhance image quality (example: simple sharpening)
def enhance_image(image):
    # Example: Apply a simple sharpening filter
    return image.filter(ImageFilter.SHARPEN)

# Flask route for the homepage
@app.route('/')
def home():
    return """
    <h1>Background Remover and Image Enhancer</h1>
    <form action="/process" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Process Image</button>
    </form>
    """

# Flask route to process the image
@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image uploaded", 400

    # Load the uploaded image
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    # Load the model
    model = load_model()

    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Remove background
    output = remove_background(model, input_tensor)

    # Convert output to a PIL image
    output_image = Image.fromarray((output * 255).astype(np.uint8))

    # Enhance the image (optional)
    enhanced_image = enhance_image(output_image)

    # Save the processed image to a bytes buffer
    buf = io.BytesIO()
    enhanced_image.save(buf, format='PNG')
    buf.seek(0)

    # Return the processed image as a downloadable file
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name='processed_image.png')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
