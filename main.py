import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from models.inference import run_inference
from models.degradation import apply_corruption_to_folder  # ✅ corrected

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DEGRADED_FOLDER'] = 'static/outputs/degraded'
app.config['RESTORED_FOLDER'] = 'static/outputs/restored'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEGRADED_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESTORED_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files['image']
        model = request.form['model'].lower()

        if uploaded_file:
            # Generate a unique timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{secure_filename(uploaded_file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            # Apply degradation to uploaded image
            apply_corruption_to_folder(app.config['UPLOAD_FOLDER'], app.config['DEGRADED_FOLDER'])

            # Load the degraded image
            degraded_path = os.path.join(app.config['DEGRADED_FOLDER'], filename)
            degraded_pil = Image.open(degraded_path).convert("RGB")

            # Run inference on degraded image
            restored_images_pil, _ = run_inference(degraded_pil, model)

            # Save restored images
            restored_paths = {}
            for model_key, img in restored_images_pil.items():
                restored_name = f"{model_key.upper()}_{filename}"
                img.save(os.path.join(app.config['RESTORED_FOLDER'], restored_name))
                restored_paths[model_key] = restored_name

            return render_template(
                "index.html",
                filename=filename,
                selected_model=model,
                degraded_image=filename,  # just filename since it's saved with same name
                ground_truth_image=filename,
                restored_images=restored_paths
            )

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
