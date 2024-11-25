import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
import lpips

# Initialize LPIPS model for perceptual similarity
loss_fn = lpips.LPIPS(net="alex")

# Function to calculate PSNR
def calculate_psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))

# Function to calculate SSIM
def calculate_ssim(original, restored):
    original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
    return ssim(original, restored, data_range=restored.max() - restored.min())

# Function to calculate LPIPS
def calculate_lpips(original, restored, device):
    original_tensor = transforms.ToTensor()(original).unsqueeze(0).to(device)
    restored_tensor = transforms.ToTensor()(restored).unsqueeze(0).to(device)
    lpips_score = loss_fn(original_tensor, restored_tensor)
    return lpips_score.item()

# Function to load the inpainting model
def load_model(model_path, device):
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to prepare the input image and mask
def prepare_image(image, mask, size=(256, 256)):
    transform = transforms.ToTensor()
    image = image.resize(size)
    mask = mask.resize(size)

    image_tensor = transform(image).unsqueeze(0)
    mask_tensor = transform(mask).unsqueeze(0)

    return image_tensor, mask_tensor

# Function to run inpainting
def run_inpainting(model, image_tensor, mask_tensor):
    with torch.no_grad():
        inpainted_image = model(image_tensor, mask_tensor)
    return inpainted_image

# Function to save tensor to image
def save_image(tensor):
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(image)

# Streamlit session state initialization
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "iteration" not in st.session_state:
    st.session_state.iteration = 1
if "inpainted_results" not in st.session_state:
    st.session_state.inpainted_results = {}

# Sidebar settings
with st.sidebar:
    st.header("Instructions")
    st.write(
        """
        1. **Upload an image** to start.
        2. **Draw on the image** to mark areas for inpainting.
        3. **Run the inpainting** process.
        4. **Repeat the process** if needed.
        """
    )
    st.header("Settings")
    stroke_width = st.slider("Stroke Width (px)", 1, 20, 5)
    drawing_mode = st.selectbox(
        "Drawing Mode", options=["freedraw", "rect", "polygon"], index=0
    )

# Image input: either from file upload or previous iteration
uploaded_image = st.file_uploader("Upload an Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.session_state.original_image = image
    st.session_state.current_image = image
    st.session_state.iteration = 1
    st.session_state.inpainted_results = {}

elif st.session_state.current_image is not None:
    image = st.session_state.current_image
else:
    image = None

# Main UI
if image:
    st.write(f"### Inpainting Iteration: {st.session_state.iteration}")
    st.image(image, caption="Current Image for Inpainting", use_column_width=True)

    # Canvas for mask drawing
    st.write("### Draw on the Image")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red
        stroke_width=stroke_width,
        stroke_color="red",
        background_image=image,
        update_streamlit=True,
        height=image.size[1],
        width=image.size[0],
        drawing_mode=drawing_mode,
        key=f"canvas_{st.session_state.iteration}",
    )

    if st.button("Generate Mask and Inpaint"):
        if canvas_result.image_data is not None:
            with st.spinner("Processing... Please wait"):
                # Generate binary mask
                alpha_channel = (canvas_result.image_data[:, :, 3] > 0).astype(np.uint8)
                mask = np.zeros_like(alpha_channel, dtype=np.uint8)
                contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask, contours, -1, color=1, thickness=cv2.FILLED)

                # Convert mask to an Image for compatibility
                mask_image = Image.fromarray(mask * 255)

                # Display the generated mask
                st.write("### Generated Mask")
                st.image(mask * 255, caption="Generated Mask", use_column_width=True)

                # Load the inpainting model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_path = r"/Users/dhirajsv/Downloads/cap_project/big-lama.pt"  # Update with your model path
                model = load_model(model_path, device)

                if model:
                    image_tensor, mask_tensor = prepare_image(image, mask_image)
                    image_tensor = image_tensor.to(device)
                    mask_tensor = mask_tensor.to(device)

                    # Run the inpainting process
                    inpainted_image_tensor = run_inpainting(model, image_tensor, mask_tensor)
                    inpainted_image = save_image(inpainted_image_tensor)

                    # Save the new inpainted image in session state
                    st.session_state.current_image = inpainted_image

                    # Calculate metrics (always with the original image resized)
                    original_resized = st.session_state.original_image.resize((256, 256))
                    original_np = np.array(original_resized)
                    restored_np = np.array(inpainted_image)
                    psnr_value = calculate_psnr(original_np, restored_np)
                    ssim_value = calculate_ssim(original_np, restored_np)
                    lpips_value = calculate_lpips(original_np, restored_np, device)

                    # Store results for this iteration
                    iteration = st.session_state.iteration
                    st.session_state.inpainted_results[iteration] = {
                        "image": inpainted_image,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "lpips": lpips_value,
                    }

                    st.write("### Inpainted Image")
                    st.image(inpainted_image, caption="Inpainted Image", use_column_width=True)
                    st.write(f"**PSNR:** {psnr_value:.2f}")
                    st.write(f"**SSIM:** {ssim_value:.4f}")
                    st.write(f"**LPIPS:** {lpips_value:.4f}")

                    # Increment iteration
                    st.session_state.iteration += 1

                    # Continue or stop
                    if st.button("Continue Iteration"):
                        st.info("Ready for the next iteration!")
        else:
            st.warning("Please draw on the canvas before generating the mask.")

    st.write("### Download Inpainted Results")
    for iter_num, result in st.session_state.inpainted_results.items():
        st.write(f"#### Iteration {iter_num}")
        st.image(result["image"], caption=f"Inpainted Image - Iteration {iter_num}", use_column_width=True)
        st.write(f"**PSNR:** {result['psnr']:.2f}, **SSIM:** {result['ssim']:.4f}, **LPIPS:** {result['lpips']:.4f}")
        buf = io.BytesIO()
        result["image"].save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label=f"Download Inpainted Image - Iteration {iter_num}",
            data=buf,
            file_name=f"inpainted_iteration_{iter_num}.png",
            mime="image/png",
        )
else:
    st.info("Please upload an image to start the inpainting process.")
