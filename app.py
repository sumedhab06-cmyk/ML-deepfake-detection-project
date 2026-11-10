from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from deepfake.data import image_transform
from deepfake.models import make_model

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "model" / "best.pt"


@st.cache_resource
def load_model():
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = make_model(saved["model_name"], pretrained=False)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved["model_name"]


st.set_page_config(
    page_title="DeepFake Sentinel // Neural Face Scanner",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=Chakra+Petch:wght@400;600;700&family=Rajdhani:wght@500;600;700&display=swap');

    .stApp {
        background-color: #0a0414;
        background-image: 
            radial-gradient(circle at 10% 15%, rgba(255, 0, 127, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 85% 20%, rgba(189, 0, 255, 0.16) 0%, transparent 42%),
            linear-gradient(rgba(10, 4, 20, 0.95), rgba(10, 4, 20, 0.95));
        background-attachment: fixed;
        font-family: 'Rajdhani', sans-serif !important;
        color: #f5e6ff !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Orbitron', monospace, sans-serif !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    @keyframes floaty {
        0%, 100% {
            transform: translateY(0px);
            box-shadow: 0 4px 18px rgba(255, 0, 127, 0.4), 0 0 25px rgba(189, 0, 255, 0.25);
        }
        50% {
            transform: translateY(-5px);
            box-shadow: 0 10px 28px rgba(255, 0, 127, 0.7), 0 0 40px rgba(189, 0, 255, 0.5);
        }
    }

    @keyframes laserSweep {
        0% { top: 0%; opacity: 0.9; }
        50% { top: 96%; opacity: 1; }
        100% { top: 0%; opacity: 0.9; }
    }

    div.stButton > button {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #ff007f 0%, #a855f7 50%, #7928ca 100%) !important;
        border: 1px solid #ff77e9 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        animation: floaty 3.5s ease-in-out infinite !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-8px) scale(1.04) !important;
        box-shadow: 0 0 30px #ff007f, 0 0 60px #bd00ff !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(25, 11, 40, 0.7);
        border: 2px dashed #ff007f;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.2);
    }

    .cyber-card {
        background: rgba(22, 10, 36, 0.78);
        border: 1px solid rgba(255, 42, 133, 0.35);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(189, 0, 255, 0.15);
    }

    .image-scanner-box {
        position: relative;
        display: inline-block;
        width: 100%;
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid #ff007f;
        box-shadow: 0 0 25px rgba(255, 0, 127, 0.45);
    }

    .laser-bar {
        position: absolute;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent, #ff007f, #ffffff, #bd00ff, transparent);
        box-shadow: 0 0 15px #ff007f, 0 0 25px #bd00ff;
        animation: laserSweep 3s ease-in-out infinite;
        z-index: 10;
        pointer-events: none;
    }

    .glow-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'Chakra Petch', sans-serif;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: 1px solid;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 25px; padding: 25px 15px; background: radial-gradient(circle, rgba(157,78,221,0.18) 0%, rgba(255,0,127,0.1) 60%, transparent 100%); border-radius: 24px; border: 1px solid rgba(255,42,133,0.3); box-shadow: 0 0 35px rgba(255,0,127,0.25);">
        <div style="display: flex; justify-content: center; align-items: center; gap: 12px; margin-bottom: 10px;">
            <span class="glow-pill" style="background: rgba(255,0,127,0.15); color: #ff77e9; border-color: #ff007f;">DEEPFAKE SENTINEL</span>
            <span class="glow-pill" style="background: rgba(189,0,255,0.15); color: #d8b4fe; border-color: #bd00ff;">NEURAL SCANNER</span>
        </div>
        <h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #ffffff 0%, #ff77e9 45%, #c084fc 80%, #ffffff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0;">
            FORENSIC FACE IMAGE INSPECTOR
        </h1>
        <p style="color: #d8b4fe; font-size: 1.1rem; max-width: 750px; margin: 0 auto;">
            Upload a cropped facial image to verify whether it matches authentic facial characteristics or deep learning synthesis artifacts.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not CHECKPOINT.is_file():
    st.error("Missing model/best.pt. Run training, then build this deployment package with package_deployment.py.")
    st.stop()

upload = st.file_uploader("UPLOAD A CROPPED FACE IMAGE (JPG, PNG)", type=["jpg", "jpeg", "png"])
if upload is not None:
    try:
        image = Image.open(upload).convert("RGB")
        if image.width * image.height > 25_000_000:
            st.error("Image is too large. Please use a cropped face image.")
            st.stop()

        col_img, col_res = st.columns([1, 1.3], gap="large")

        with col_img:
            st.markdown(
                """
                <div class="cyber-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-family: 'Orbitron'; font-size: 0.82rem; color: #ff77e9; font-weight: 700;">TARGET FACE RETICLE</span>
                        <span class="glow-pill" style="background: rgba(255,0,127,0.15); color: #ff77e9; border-color: #ff007f;">LOCKED</span>
                    </div>
                    <div class="image-scanner-box">
                        <div class="laser-bar"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(image, caption="Uploaded face target", use_container_width=True)

        model, name = load_model()
        with torch.inference_mode():
            tensor = image_transform()(image).unsqueeze(0)
            probability = torch.softmax(model(tensor), dim=1)[0, 1].item()

        is_fake = probability >= 0.5

        with col_res:
            if is_fake:
                st.markdown(
                    f"""
                    <div class="cyber-card" style="border-color: #ff0055; box-shadow: 0 0 30px rgba(255,0,85,0.4); text-align: center;">
                        <span class="glow-pill" style="background: rgba(255,0,85,0.2); color: #ff77aa; border-color: #ff0055;">DEEPFAKE DETECTED</span>
                        <h2 style="color: #ff2a85; margin: 10px 0;">SYNTHETIC MANIPULATION</h2>
                        <div style="font-family: 'Orbitron'; font-size: 2.2rem; color: #ffffff; font-weight: 800;">
                            {probability:.1%}
                        </div>
                        <p style="color: #fca5a5;">Estimated Synthetic Probability &bull; Architecture: <b>{name}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="cyber-card" style="border-color: #00f5d4; box-shadow: 0 0 30px rgba(0,245,212,0.4); text-align: center;">
                        <span class="glow-pill" style="background: rgba(0,245,212,0.2); color: #00f5d4; border-color: #00f5d4;">AUTHENTIC CONFIRMED</span>
                        <h2 style="color: #00f5d4; margin: 10px 0;">REAL HUMAN FACE</h2>
                        <div style="font-family: 'Orbitron'; font-size: 2.2rem; color: #ffffff; font-weight: 800;">
                            {(1 - probability):.1%}
                        </div>
                        <p style="color: #a7f3d0;">Estimated Authenticity Confidence &bull; Architecture: <b>{name}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(f"Model checkpoint: {name} &bull; Fixed threshold: 0.5. Research demo only.")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not read that image: {exc}")
