import base64
import io
import os

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

application = Flask(__name__)
# Set max content length for 16MB uploads
application.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Set TRANSFORMERS_CACHE to a writable directory in the container
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"

# Initialize model and processor
# We use a try-except block to log any errors during the long model loading process
try:
    print("Loading Blip2Processor...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    print("Loading Blip2ForConditionalGeneration model...")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    print("Model loading complete.")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a real app, you might want to handle this more gracefully
    # For now, we'll let it fail fast if the model can't load.
    raise e


@application.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the app is running."""
    # A more robust health check might try a quick inference
    return jsonify({"status": "healthy"}), 200


def generate_label(image):
    """Generate caption for the given image"""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    image = image.convert("RGB")

    # device="cuda" is implied by device_map="auto" if a GPU is present
    inputs = processor(image, return_tensors="pt").to(model.device, torch.float16)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def extract_attributes(caption):
    """Extract simple attributes from the generated caption."""
    colors = [
        "red",
        "blue",
        "green",
        "black",
        "white",
        "yellow",
        "pink",
        "orange",
        "brown",
        "gray",
    ]
    color = next((c for c in colors if c in caption.lower()), None)

    clothing_keywords = [
        "dress",
        "shirt",
        "jacket",
        "jeans",
        "pants",
        "skirt",
        "hoodie",
        "coat",
        "shoes",
    ]
    clothing_type = next((c for c in clothing_keywords if c in caption.lower()), None)

    return {"color": color, "type": clothing_type, "caption": caption}


@application.route("/classify", methods=["POST"])
def classify_image():
    """Main classification endpoint."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        image_data = data["image"]
        # Handle both raw base64 and data URI formats
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        caption = generate_label(image)
        attributes = extract_attributes(caption)

        return jsonify(attributes)

    except base64.binascii.Error as e:
        return jsonify({"error": f"Invalid base64 string: {e}"}), 400
    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # This block is not used when running with Gunicorn,
    # but it's good practice to keep for local development.
    # To run locally: flask --app main run --port 8080
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
