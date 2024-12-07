from flask import Flask, render_template, request, url_for
import numpy as np
from PIL import Image
import open_clip
import torch.nn.functional as F
from sklearn.decomposition import PCA
import io
import os
import pandas as pd

app = Flask(__name__)

# Load the DataFrame with embeddings
df = pd.read_pickle('image_embeddings.pickle')

# Initialize CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Initialize PCA
pca = None
def init_pca(embeddings, n_components):
    global pca
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    return pca.transform(embeddings)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    try:
        # Get query inputs
        text_query = request.form.get("text-query", "")
        image_file = request.files.get("image-query")
        weight = float(request.form.get("weight", 0.5))  # Text query weight
        use_pca = request.form.get("use-pca") == "true"
        k_components = int(request.form.get("pca-components", 5))
        
        # Initialize variables
        text_embedding_np = None
        image_embedding_np = None
        database_embeddings = np.array(df['embedding'].tolist())
        
        # Process text query if provided
        if text_query:
            text = tokenizer([text_query])
            text_embedding = F.normalize(model.encode_text(text))
            text_embedding_np = text_embedding.detach().numpy()

        # Process image query if provided
        if image_file:
            # Read and preprocess image
            image = Image.open(io.BytesIO(image_file.read()))
            image_tensor = preprocess(image).unsqueeze(0)
            image_embedding = F.normalize(model.encode_image(image_tensor))
            image_embedding_np = image_embedding.detach().numpy()

            if use_pca:
                # Initialize PCA if not done yet
                if pca is None or pca.n_components_ != k_components:
                    database_embeddings = init_pca(database_embeddings, k_components)
                    
                # Transform the query image embedding
                image_embedding_np = pca.transform(image_embedding_np)

        # Combine embeddings if both queries present
        if text_embedding_np is not None and image_embedding_np is not None:
            # If using PCA for image, project text embedding to PCA space
            if use_pca:
                text_embedding_pca = pca.transform(text_embedding_np)
                query = weight * text_embedding_pca + (1.0 - weight) * image_embedding_np
            else:
                query = weight * text_embedding_np + (1.0 - weight) * image_embedding_np
            # Normalize the combined query
            query = query / np.linalg.norm(query)
        elif text_embedding_np is not None:
            query = text_embedding_np
        else:
            query = image_embedding_np

        # Calculate similarities
        similarities = np.dot(database_embeddings, query.T).flatten()

        # Get top 5 results
        top_indices = np.argsort(similarities)[-5:][::-1]
        results = []
        
        # Ensure results directory exists
        os.makedirs('static/results', exist_ok=True)
        
        for idx in top_indices:
            try:
                source_path = f'coco_images_resized/{df.iloc[idx]["file_name"]}'
                target_path = f'static/results/{df.iloc[idx]["file_name"]}'
                
                # Copy image to static directory if it doesn't exist
                if not os.path.exists(target_path):
                    import shutil
                    shutil.copy2(source_path, target_path)
                
                result = {
                    'image': url_for('static', filename=f'results/{df.iloc[idx]["file_name"]}'),
                    'score': float(similarities[idx])
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing result {idx}: {str(e)}")
                continue

        return render_template(
            "index.html",
            results=results,
            text_query=text_query,
            has_image_query=image_file is not None,
            weight=weight,
            use_pca=use_pca,
            pca_components=k_components
        )
        
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5050)