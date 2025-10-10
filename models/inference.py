import torch
from PIL import Image
import torchvision.transforms as transforms
from models.vae_model import VAE
from models.wgan_model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VAE
vae_model = VAE(latent_dim=128).to(device)
vae_model.load_state_dict(torch.load("models/vae.pth", map_location=device))
vae_model.eval()

# Load WGAN
wgan_model = Generator().to(device)
checkpoint = torch.load("models/wgan.pth", map_location=device)
if 'generator' in checkpoint:
    wgan_model.load_state_dict(checkpoint['generator'])  # checkpoint format
else:
    wgan_model.load_state_dict(checkpoint)               # direct weights
wgan_model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# Common restore function
def restore_image(model, pil_img):
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, tuple):  # VAE returns (recon, mu, logvar)
            output = output[0]
    output = output.squeeze(0).cpu().clamp(0, 1)
    return to_pil(output)

# Run inference
def run_inference(degraded_pil, selected_model):
    results = {}

    # Ensure the degraded image is resized like the models expect
    degraded_resized = degraded_pil.resize((256, 256))

    # Normalize model selection
    selected_model = selected_model.lower()

    if selected_model in ["vae", "all"]:
        results["vae"] = restore_image(vae_model, degraded_resized)

    if selected_model in ["wgan", "all"]:
        results["wgan"] = restore_image(wgan_model, degraded_resized)

    if selected_model in ["pix2pix", "all"]:
        try:
            print(">>> Loading Pix2Pix model...")
            import tensorflow as tf
            from models.pix2pix_model import Pix2PixModel
            import numpy as np

            # Load the saved TensorFlow model from models/pix2pix
            pix2pix = Pix2PixModel("models/pix2pix")
            print("✅ Pix2Pix model loaded.")

            # Convert PIL to OpenCV (BGR)
            img_cv = np.array(degraded_resized)[..., ::-1]

            # Run prediction
            output_np = pix2pix.predict(img_cv)
            print("✅ Pix2Pix prediction complete.")

            # Convert OpenCV (BGR) output back to PIL (RGB)
            output_pil = Image.fromarray(output_np[..., ::-1])
            results["pix2pix"] = output_pil
            print("✅ Pix2Pix image added to results.")
        except Exception as e:
            print(f"❌ Pix2Pix model failed: {e}")


    return results, degraded_resized
