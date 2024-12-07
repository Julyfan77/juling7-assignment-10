from flask import Flask, render_template, request, url_for
import numpy as np
from PIL import Image
import open_clip
import torch.nn.functional as F
import io
import base64
import pandas as pd

app = Flask(__name__)

# Initialize CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    # Get query inputs
    text_query = request.form.get("text-query", "")
    image_file = request.files.get("image-query")
    weight = float(request.form.get("weight", 0.5))
    pca_components = int(request.form.get("pca-components", 5))
    
    # Initialize embeddings
    text_embedding = None
    image_embedding = None
    
    df = pd.read_pickle('image_embeddings.pickle')
    # Process text query if provided
    if text_query:
        text = tokenizer([text_query])
        text_embedding = F.normalize(model.encode_text(text))

    # Process image query if provided
    if image_file:
        # Read and preprocess image
        image = Image.open(io.BytesIO(image_file.read()))
        image_tensor = preprocess(image).unsqueeze(0)
        image_embedding = F.normalize(model.encode_image(image_tensor))

    # Combine embeddings if both queries present
    if text_embedding is not None and image_embedding is not None:
        query = F.normalize(weight * text_embedding + (1.0 - weight) * image_embedding)
    elif text_embedding is not None:
        query = text_embedding
    else:
        query = image_embedding

    # Convert query to numpy for similarity calculation
    query_np = query.detach().numpy()

    # Calculate similarities with database embeddings
    similarities = np.dot(np.array(df['embedding'].tolist()), query_np.T).flatten()

    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    results = []
    
    for idx in top_indices:
        # Make sure the image exists in your static/results directory
        source_path = f'coco_images_resized/{df.iloc[idx]["file_name"]}'
        # Create a URL that Flask can serve
        image_url = url_for('static', filename=f'results/{df.iloc[idx]["file_name"]}')
        
        # Copy or save the image to static/results if needed
        import shutil
        import os
        
        # Ensure the results directory exists
        os.makedirs('static/results', exist_ok=True)
        
        # Copy the image to static/results if it doesn't exist
        target_path = f'static/results/{df.iloc[idx]["file_name"]}'
        if not os.path.exists(target_path):
            shutil.copy2(source_path, target_path)
            
        result = {
            'image': image_url,  # Use the URL instead of file path
            'score': float(similarities[idx])
        }
        results.append(result)
    return render_template(
        "index.html",
        results=results,
        text_query=text_query,
        has_image_query=image_file is not None,
        weight=weight,
        pca_components=pca_components
    )

if __name__ == "__main__":
    app.run(debug=True, port=5050)