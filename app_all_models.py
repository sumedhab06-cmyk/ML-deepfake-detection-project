import base64
import json
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch

from deepfake.data import image_transform
from deepfake.models import make_model

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

MODELS_CONFIG = {
    "MobileNetV3-Small": {
        "key": "mobilenet_v3",
        "tag": "ULTRA-FAST EDGE",
        "color": "#ff2a85",
        "params": "1.52M",
        "desc": "Lightweight inverted residual architecture optimized for fast inference.",
    },
    "EfficientNet-B0": {
        "key": "efficientnet_b0",
        "tag": "COMPOUND BALANCED",
        "color": "#d946ef",
        "params": "4.01M",
        "desc": "Uniformly scaled depth, width, and resolution for high parameter efficiency.",
    },
    "ResNet-50": {
        "key": "resnet50",
        "tag": "DEEP RESIDUAL",
        "color": "#a855f7",
        "params": "23.51M",
        "desc": "Deep bottleneck residual representations with high feature discrimination.",
    },
}

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepFake Sentinel // Neural Face Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Futuristic Cyberpunk / Gaming CSS Styles (Pink & Purple Neon)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800;900&family=Chakra+Petch:wght@400;500;600;700&family=Rajdhani:wght@500;600;700&family=Press+Start+2P&display=swap');

    :root {
        --neon-pink: #ff007f;
        --neon-pink-glow: #ff2a85;
        --neon-purple: #9d4edd;
        --neon-violet: #bd00ff;
        --cyber-cyan: #00f5d4;
        --neon-crimson: #ff0055;
        --bg-dark: #0a0414;
        --card-bg: rgba(22, 10, 36, 0.78);
        --card-border: rgba(255, 42, 133, 0.35);
    }

    /* Overall App Background & Typography */
    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            radial-gradient(circle at 10% 15%, rgba(255, 0, 127, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 85% 20%, rgba(189, 0, 255, 0.16) 0%, transparent 42%),
            radial-gradient(circle at 50% 85%, rgba(157, 78, 221, 0.12) 0%, transparent 50%),
            linear-gradient(rgba(10, 4, 20, 0.95), rgba(10, 4, 20, 0.95));
        background-attachment: fixed;
        font-family: 'Rajdhani', sans-serif !important;
        color: #f5e6ff !important;
    }

    /* Headings Gaming Font */
    h1, h2, h3, h4, .cyber-font {
        font-family: 'Orbitron', monospace, sans-serif !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    p, span, label, div {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.02rem;
        letter-spacing: 0.5px;
    }

    /* Floating Floaty Animation for Buttons */
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

    /* Flash Shimmer Effect */
    @keyframes flashSweep {
        0% { left: -120%; }
        100% { left: 160%; }
    }

    /* Neon Pulse Glow */
    @keyframes neonPulse {
        0%, 100% {
            text-shadow: 0 0 10px #ff007f, 0 0 20px #ff007f, 0 0 35px #bd00ff, 0 0 60px #bd00ff;
        }
        50% {
            text-shadow: 0 0 15px #ff77e9, 0 0 30px #ff2a85, 0 0 50px #bd00ff, 0 0 85px #bd00ff;
        }
    }

    /* Laser Scanner Beam */
    @keyframes laserSweep {
        0% { top: 0%; opacity: 0.9; }
        50% { top: 96%; opacity: 1; }
        100% { top: 0%; opacity: 0.9; }
    }

    /* Float Animation for Info Cards */
    @keyframes cardFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }

    /* Streamlit Button Styling: Glowy & Floaty */
    div.stButton > button {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #ff007f 0%, #a855f7 50%, #7928ca 100%) !important;
        border: 1px solid #ff77e9 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        animation: floaty 3.5s ease-in-out infinite !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        position: relative !important;
        overflow: hidden !important;
        cursor: pointer !important;
    }

    div.stButton > button:hover {
        transform: translateY(-8px) scale(1.04) !important;
        box-shadow: 0 0 30px #ff007f, 0 0 60px #bd00ff, 0 0 90px #ff00a0 !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #ff2a85 0%, #bd00ff 100%) !important;
    }

    div.stButton > button:active {
        transform: translateY(2px) scale(0.98) !important;
    }

    /* Cyber Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(20, 9, 34, 0.6);
        padding: 8px 12px;
        border-radius: 14px;
        border: 1px solid rgba(255, 42, 133, 0.25);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        font-size: 0.92rem !important;
        color: #d8b4e2 !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        border: 1px solid transparent !important;
        transition: all 0.25s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ff77e9 !important;
        border-color: rgba(255, 42, 133, 0.5) !important;
        box-shadow: 0 0 15px rgba(255, 42, 133, 0.3) !important;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(255, 0, 127, 0.4) 0%, rgba(189, 0, 255, 0.4) 100%) !important;
        border: 1px solid #ff2a85 !important;
        box-shadow: 0 0 20px rgba(255, 42, 133, 0.55), inset 0 0 10px rgba(189, 0, 255, 0.4) !important;
    }

    /* File Uploader Cyber Container */
    [data-testid="stFileUploader"] {
        background: rgba(25, 11, 40, 0.7);
        border: 2px dashed #ff007f;
        border-radius: 16px;
        padding: 18px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.2);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #bd00ff;
        box-shadow: 0 0 25px rgba(189, 0, 255, 0.4), inset 0 0 15px rgba(255, 0, 127, 0.15);
    }

    /* Cyber Glass Cards */
    .cyber-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 20px;
        position: relative;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(189, 0, 255, 0.15);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }

    .cyber-card:hover {
        border-color: #ff2a85;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5), 0 0 30px rgba(255, 42, 133, 0.35);
        transform: translateY(-3px);
    }

    /* Reticle & Scanner Box for Image Display */
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

    /* Stat Glow Pills */
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

    .glow-pill-pink {
        background: rgba(255, 0, 127, 0.15);
        color: #ff77e9;
        border-color: #ff007f;
        box-shadow: 0 0 10px rgba(255, 0, 127, 0.4);
    }

    .glow-pill-purple {
        background: rgba(189, 0, 255, 0.15);
        color: #d8b4fe;
        border-color: #bd00ff;
        box-shadow: 0 0 10px rgba(189, 0, 255, 0.4);
    }

    .glow-pill-cyan {
        background: rgba(0, 245, 212, 0.15);
        color: #00f5d4;
        border-color: #00f5d4;
        box-shadow: 0 0 10px rgba(0, 245, 212, 0.4);
    }

    /* Custom Neon Progress Bar */
    .cyber-meter-track {
        width: 100%;
        height: 12px;
        background: rgba(40, 15, 60, 0.7);
        border-radius: 8px;
        border: 1px solid rgba(255, 42, 133, 0.3);
        overflow: hidden;
        margin: 10px 0;
        position: relative;
    }

    .cyber-meter-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 0 15px currentColor;
    }

    /* Banner Consensus Alerts */
    .verdict-banner-fake {
        background: linear-gradient(135deg, rgba(255, 0, 85, 0.25) 0%, rgba(189, 0, 255, 0.2) 100%);
        border: 2px solid #ff0055;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 0 35px rgba(255, 0, 85, 0.45), inset 0 0 20px rgba(255, 0, 85, 0.2);
        animation: cardFloat 3s ease-in-out infinite;
    }

    .verdict-banner-real {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.2) 0%, rgba(157, 78, 221, 0.2) 100%);
        border: 2px solid #00f5d4;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 0 35px rgba(0, 245, 212, 0.4), inset 0 0 20px rgba(0, 245, 212, 0.15);
        animation: cardFloat 3s ease-in-out infinite;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d0519;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#ff007f, #bd00ff);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #ff77e9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_deepfake_model(model_name: str):
    checkpoint_path = ROOT / "outputs" / model_name / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing for {model_name} at {checkpoint_path}. Train the model first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_all_metrics():
    metrics = {}
    for display_name, meta in MODELS_CONFIG.items():
        metrics_file = ROOT / "outputs" / meta["key"] / "test_metrics.json"
        if metrics_file.is_file():
            with open(metrics_file, "r") as f:
                metrics[display_name] = json.load(f)
    return metrics


@st.cache_data
def load_training_histories():
    histories = {}
    for display_name, meta in MODELS_CONFIG.items():
        hist_file = ROOT / "outputs" / meta["key"] / "history.csv"
        if hist_file.is_file():
            histories[display_name] = pd.read_csv(hist_file)
    return histories



# -----------------------------------------------------------------------------
# Top Hero Header with Neon Gaming HUD
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 25px; padding: 25px 15px; background: radial-gradient(circle, rgba(157,78,221,0.18) 0%, rgba(255,0,127,0.1) 60%, transparent 100%); border-radius: 24px; border: 1px solid rgba(255,42,133,0.3); box-shadow: 0 0 35px rgba(255,0,127,0.25);">
        <div style="display: flex; justify-content: center; align-items: center; gap: 12px; margin-bottom: 10px;">
            <span class="glow-pill glow-pill-pink">CYBER HUD V2.4</span>
            <span class="glow-pill glow-pill-purple">DEEPFAKE CLASSIFIER</span>
        </div>
        <h1 style="font-size: 2.7rem; font-weight: 900; background: linear-gradient(135deg, #ffffff 0%, #ff77e9 45%, #c084fc 80%, #ffffff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0; text-shadow: 0 0 25px rgba(255,42,133,0.6);">
            DEEPFAKE SENTINEL // MULTI-MODEL SCANNER
        </h1>
        <p style="color: #d8b4fe; font-size: 1.15rem; font-weight: 600; letter-spacing: 1px; max-width: 820px; margin: 0 auto;">
            Simultaneous multi-architecture facial forensic inference powered by MobileNetV3-Small, EfficientNet-B0, and ResNet-50.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_scanner, tab_benchmarks, tab_dossier = st.tabs(
    [
        "NEURAL SCANNER",
        "PERFORMANCE MATRIX",
        "ARCHITECTURE & DATASET DOSSIER",
    ]
)

# =============================================================================
# TAB 1: NEURAL SCANNER (LIVE INFERENCE)
# =============================================================================
with tab_scanner:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <h3 style="color: #ff77e9; margin: 0; font-size: 1.25rem;">
                INPUT FACE SOURCE & LIVE INFERENCE ENGINE
            </h3>
            <span class="glow-pill glow-pill-purple">READY FOR INPUT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Custom File Uploader
    uploaded_file = st.file_uploader(
        "UPLOAD A CROPPED FACIAL IMAGE (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a tightly cropped face image formatted at 224x224 (or scaled automatically).",
    )

    # Determine image to analyze
    current_image = None
    image_origin_title = None

    if uploaded_file is not None:
        try:
            current_image = Image.open(uploaded_file).convert("RGB")
            image_origin_title = f"Uploaded File: {uploaded_file.name}"
        except Exception as exc:
            st.error(f"Failed to decode uploaded image: {exc}")

    # If no image chosen yet, show an inviting prompt
    if current_image is None:
        st.markdown(
            """
            <div class="cyber-card" style="text-align: center; padding: 45px 20px; border: 2px dashed rgba(255,42,133,0.4); margin-top: 15px;">
                <h3 style="color: #ff77e9; margin-bottom: 10px;">NEURAL TARGETING SYSTEM ARMED</h3>
                <p style="color: #c084fc; max-width: 600px; margin: 0 auto; font-size: 1.1rem;">
                    <b>Drag & drop or browse a cropped facial image</b> above to activate the triple-engine forensic inspection pipeline.
                </p>
                <div style="margin-top: 20px; display: flex; justify-content: center; gap: 12px;">
                    <span class="glow-pill glow-pill-pink">ResNet-50</span>
                    <span class="glow-pill glow-pill-purple">EfficientNet-B0</span>
                    <span class="glow-pill glow-pill-cyan">MobileNetV3-Small</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Run inference across models
        if current_image.width * current_image.height > 25_000_000:
            st.error("Image dimensions exceed 25MP limit. Please provide a cropped face.")
            st.stop()

        col_img, col_preds = st.columns([1, 1.4], gap="large")

        with col_img:
            st.markdown(
                f"""
                <div class="cyber-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-family: 'Orbitron'; font-size: 0.82rem; color: #ff77e9; font-weight: 700;">
                            TARGET FACE RETICLE
                        </span>
                        <span class="glow-pill glow-pill-pink">LOCKED</span>
                    </div>
                    <div class="image-scanner-box">
                        <div class="laser-bar"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(
                current_image,
                caption=f"{image_origin_title} ({current_image.width}x{current_image.height} px)",
                width="stretch",
            )

            st.markdown(
                f"""
                <div style="margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,42,133,0.25); border-radius: 10px; padding: 8px 12px;">
                        <span style="font-size: 0.75rem; color: #a855f7; text-transform: uppercase;">Color Space</span>
                        <div style="font-family: 'Orbitron'; font-size: 0.95rem; color: #ffffff;">RGB (3-Ch)</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(189,0,255,0.25); border-radius: 10px; padding: 8px 12px;">
                        <span style="font-size: 0.75rem; color: #a855f7; text-transform: uppercase;">Resolution</span>
                        <div style="font-family: 'Orbitron'; font-size: 0.95rem; color: #ffffff;">{current_image.width} × {current_image.height}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_preds:
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="color: #ff77e9; margin: 0; font-size: 1.15rem;">
                        FORENSIC INFERENCE ENSEMBLE
                    </h3>
                    <span class="glow-pill glow-pill-cyan">ACTIVE</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Transform image to tensor
            tensor = image_transform()(current_image).unsqueeze(0).to(device)

            model_results = []
            for display_name, meta in MODELS_CONFIG.items():
                try:
                    model = load_deepfake_model(meta["key"])
                    with torch.inference_mode():
                        logits = model(tensor)
                        fake_prob = torch.softmax(logits, dim=1)[0, 1].item()
                    prediction = "Fake" if fake_prob >= 0.5 else "Real"
                    model_results.append(
                        {
                            "name": display_name,
                            "key": meta["key"],
                            "tag": meta["tag"],
                            "params": meta["params"],
                            "prediction": prediction,
                            "probability": fake_prob,
                            "color": meta["color"],
                        }
                    )
                except Exception as exc:
                    st.warning(f"Engine {display_name} unavailable: {exc}")

            if model_results:
                fake_count = sum(r["prediction"] == "Fake" for r in model_results)
                real_count = len(model_results) - fake_count
                avg_prob = sum(r["probability"] for r in model_results) / len(model_results)

                # High-Tech Consensus Verdict Banner
                if fake_count >= 2:
                    st.markdown(
                        f"""
                        <div class="verdict-banner-fake">
                            <div style="display: inline-block; padding: 4px 14px; border-radius: 20px; background: rgba(255, 0, 85, 0.3); border: 1px solid #ff0055; font-family: 'Orbitron'; font-size: 0.85rem; color: #ff77aa; margin-bottom: 8px;">
                                DEEPFAKE DETECTED
                            </div>
                            <h2 style="color: #ff2a85; font-size: 1.85rem; margin: 5px 0 8px 0; text-shadow: 0 0 20px #ff0055;">
                                SYNTHETIC FACE DETECTED
                            </h2>
                            <p style="color: #fca5a5; font-size: 1.05rem; margin: 0; font-weight: 600;">
                                Ensemble Agreement: <b>{fake_count}/{len(model_results)} Models Flagged As Manipulated</b> &bull; Mean Synthetic Risk: <b>{avg_prob:.1%}</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="verdict-banner-real">
                            <div style="display: inline-block; padding: 4px 14px; border-radius: 20px; background: rgba(0, 245, 212, 0.25); border: 1px solid #00f5d4; font-family: 'Orbitron'; font-size: 0.85rem; color: #00f5d4; margin-bottom: 8px;">
                                AUTHENTICITY CONFIRMED
                            </div>
                            <h2 style="color: #00f5d4; font-size: 1.85rem; margin: 5px 0 8px 0; text-shadow: 0 0 20px #00f5d4;">
                                AUTHENTIC HUMAN FACE
                            </h2>
                            <p style="color: #a7f3d0; font-size: 1.05rem; margin: 0; font-weight: 600;">
                                Ensemble Agreement: <b>{real_count}/{len(model_results)} Models Verified As Real</b> &bull; Mean Authentic Confidence: <b>{(1 - avg_prob):.1%}</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

                # Individual Model Cards
                for res in model_results:
                    pred_color = "#ff2a85" if res["prediction"] == "Fake" else "#00f5d4"
                    fill_pct = int(res["probability"] * 100)
                    fill_gradient = (
                        "linear-gradient(90deg, #ff007f, #ff2a85)"
                        if res["prediction"] == "Fake"
                        else "linear-gradient(90deg, #9d4edd, #00f5d4)"
                    )

                    st.markdown(
                        f"""
                        <div class="cyber-card" style="margin-bottom: 12px; padding: 16px 20px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <div>
                                    <span style="font-family: 'Orbitron'; font-size: 1.05rem; font-weight: 800; color: #ffffff;">
                                        {res["name"]}
                                    </span>
                                    <span class="glow-pill glow-pill-purple" style="margin-left: 8px; font-size: 0.72rem;">
                                        {res["tag"]} &bull; {res["params"]}
                                    </span>
                                </div>
                                <span style="font-family: 'Orbitron'; font-size: 1rem; font-weight: 900; color: {pred_color}; text-shadow: 0 0 10px {pred_color};">
                                    [{res["prediction"].upper()}]
                                </span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; color: #d8b4fe; margin-top: 4px;">
                                <span>Synthetic Likelihood</span>
                                <span style="font-family: 'Orbitron'; font-weight: 700; color: #ffffff;">{res["probability"]:.1%}</span>
                            </div>
                            <div class="cyber-meter-track">
                                <div class="cyber-meter-fill" style="width: {fill_pct}%; background: {fill_gradient}; color: {pred_color};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown(
            """
            <div style="margin-top: 25px; padding: 14px 18px; background: rgba(255, 42, 133, 0.08); border-left: 4px solid #ff007f; border-radius: 8px;">
                <span style="font-family: 'Orbitron'; font-size: 0.85rem; color: #ff77e9; font-weight: 700;">RESEARCH NOTICE:</span>
                <span style="color: #e9d5ff; font-size: 0.95rem; margin-left: 8px;">
                    This prototype is trained on Celeb-DF v2 facial crops and benchmarked on CACD deepfakes. Predictions indicate similarity to known manipulation patterns and are not calibrated for courtroom forensic deployment.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 2: PERFORMANCE MATRIX & BENCHMARK ANALYTICS
# =============================================================================
with tab_benchmarks:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h3 style="color: #ff77e9; margin: 0 0 6px 0;">
                BENCHMARK PERFORMANCE MATRIX & FORENSIC METRICS
            </h3>
            <p style="color: #c084fc; margin: 0; line-height: 1.6;">
                Trained on <b>28,942 samples</b> (~28,000 GPU-trained facial crops) out of <b>45,035 total dataset samples</b> (~45,000 total across Celeb-DF v2 & CACD). Evaluated on the held-out cross-source test set of <b>8,819 unseen test facial images</b> (5,063 authentic CACD faces vs 3,756 face-swapped deepfakes).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_metrics = load_all_metrics()

    if all_metrics:
        # Comparison Table
        table_rows = []
        for name, m in all_metrics.items():
            cm = m.get("confusion_matrix_real_fake")
            if cm and len(cm) == 2 and len(cm[0]) == 2:
                tn = cm[0][0]
                fp = cm[0][1]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            elif "specificity" in m:
                specificity = m["specificity"]
            elif "balanced_accuracy" in m and "recall_fake" in m:
                specificity = 2 * m["balanced_accuracy"] - m["recall_fake"]
            else:
                specificity = 0.0

            table_rows.append(
                {
                    "Architecture": name,
                    "Parameters": f"{m.get('parameter_count', 0):,}",
                    "Test Accuracy": f"{m.get('accuracy', 0):.2%}",
                    "Balanced Acc": f"{m.get('balanced_accuracy', 0):.2%}",
                    "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                    "Fake Precision": f"{m.get('precision_fake', 0):.2%}",
                    "Fake Recall": f"{m.get('recall_fake', 0):.2%}",
                    "Specificity": f"{specificity:.2%}",
                    "Fake F1-Score": f"{m.get('f1_fake', 0):.4f}",
                    "Latency / Batch": f"{m.get('seconds_per_image_batch_amortized', 0)*1000:.2f} ms",
                }
            )

        df_metrics = pd.DataFrame(table_rows)

        # High-Tech Metric Cards Summary
        st.markdown(
            """
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px;">
                <div class="cyber-card" style="border-color: #ff007f;">
                    <div style="font-family: 'Chakra Petch'; color: #ff77e9; font-size: 0.8rem; text-transform: uppercase;">Top Benchmark ROC-AUC</div>
                    <div style="font-family: 'Orbitron'; font-size: 1.8rem; color: #ffffff; font-weight: 800; margin: 4px 0;">97.66%</div>
                    <div style="font-size: 0.82rem; color: #a855f7;">ResNet-50 Leader</div>
                </div>
                <div class="cyber-card" style="border-color: #bd00ff;">
                    <div style="font-family: 'Chakra Petch'; color: #d8b4fe; font-size: 0.8rem; text-transform: uppercase;">Top Test Accuracy</div>
                    <div style="font-family: 'Orbitron'; font-size: 1.8rem; color: #ffffff; font-weight: 800; margin: 4px 0;">92.26%</div>
                    <div style="font-size: 0.82rem; color: #a855f7;">ResNet-50 / EfficientNet-B0</div>
                </div>
                <div class="cyber-card" style="border-color: #ff2a85;">
                    <div style="font-family: 'Chakra Petch'; color: #ff77aa; font-size: 0.8rem; text-transform: uppercase;">Top Specificity</div>
                    <div style="font-family: 'Orbitron'; font-size: 1.8rem; color: #ffffff; font-weight: 800; margin: 4px 0;">92.38%</div>
                    <div style="font-size: 0.82rem; color: #a855f7;">EfficientNet-B0 (True Real Rate)</div>
                </div>
                <div class="cyber-card" style="border-color: #00f5d4;">
                    <div style="font-family: 'Chakra Petch'; color: #00f5d4; font-size: 0.8rem; text-transform: uppercase;">Fastest Edge Latency</div>
                    <div style="font-family: 'Orbitron'; font-size: 1.8rem; color: #ffffff; font-weight: 800; margin: 4px 0;">2.44 ms</div>
                    <div style="font-size: 0.82rem; color: #a855f7;">MobileNetV3-Small (1.52M params)</div>
                </div>
                <div class="cyber-card" style="border-color: #ff77e9;">
                    <div style="font-family: 'Chakra Petch'; color: #ff77e9; font-size: 0.8rem; text-transform: uppercase;">Trained / Total Samples</div>
                    <div style="font-family: 'Orbitron'; font-size: 1.65rem; color: #ffffff; font-weight: 800; margin: 4px 0;">28,942 / 45,035</div>
                    <div style="font-size: 0.82rem; color: #a855f7;">~28k GPU-Trained &bull; 8,819 Test Split</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(df_metrics, width="stretch", hide_index=True)

    # ROC Curves & Confusion Matrices
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="color: #ff77e9; margin: 0 0 14px 0;">
            ROC CURVES & CONFUSION MATRICES BY MODEL
        </h3>
        """,
        unsafe_allow_html=True,
    )

    model_sel = st.selectbox(
        "SELECT ARCHITECTURE TO INSPECT FORENSIC CURVES",
        options=list(MODELS_CONFIG.keys()),
        index=1,
    )

    selected_key = MODELS_CONFIG[model_sel]["key"]
    output_dir = ROOT / "outputs" / selected_key
    cm_path = output_dir / "confusion_matrix.png"
    roc_path = output_dir / "roc_curve.png"

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown(
            f"""
            <div class="cyber-card" style="text-align: center; margin-bottom: 10px;">
                <div style="font-family: 'Orbitron'; color: #ff77e9; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px;">
                    CONFUSION MATRIX // {model_sel.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if cm_path.is_file():
            st.image(str(cm_path), width="stretch")
        else:
            st.info("Confusion matrix plot not found in outputs directory.")

    with col_roc:
        st.markdown(
            f"""
            <div class="cyber-card" style="text-align: center; margin-bottom: 10px;">
                <div style="font-family: 'Orbitron'; color: #00f5d4; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px;">
                    ROC-AUC CURVE // {model_sel.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if roc_path.is_file():
            st.image(str(roc_path), width="stretch")
        else:
            st.info("ROC Curve plot not found in outputs directory.")

    # Training Curves Interactive Explorer
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="color: #ff77e9; margin: 0 0 14px 0;">
            TRAINING LOSS & VALIDATION CONVERGENCE HISTORY
        </h3>
        """,
        unsafe_allow_html=True,
    )

    histories = load_training_histories()
    if histories:
        metric_choice = st.radio(
            "SELECT CONVERGENCE METRIC TO PLOT",
            options=["train_loss", "accuracy", "roc_auc", "f1_fake"],
            format_func=lambda x: {
                "train_loss": "Training Loss",
                "accuracy": "Validation Accuracy",
                "roc_auc": "Validation ROC-AUC",
                "f1_fake": "Validation F1-Score (Fake Class)",
            }[x],
            horizontal=True,
        )

        chart_data = pd.DataFrame()
        for name, df_hist in histories.items():
            if metric_choice in df_hist.columns:
                chart_data[name] = df_hist[metric_choice]

        chart_data.index = chart_data.index + 1
        chart_data.index.name = "Epoch"

        st.line_chart(chart_data)


# =============================================================================
# TAB 3: ARCHITECTURE & DATASET DOSSIER
# =============================================================================
with tab_dossier:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h3 style="color: #ff77e9; margin: 0 0 8px 0;">
                PROJECT DOSSIER, METHODOLOGY & DATASET ARCHITECTURE
            </h3>
            <p style="color: #c084fc; margin: 0;">
                Complete specifications regarding the deep learning pipeline, dataset partitioning, and forensic generalization protocol.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_doss1, col_doss2 = st.columns(2, gap="large")

    with col_doss1:
        st.markdown(
            """
            <div class="cyber-card" style="margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <h4 style="color: #ff77e9; margin: 0;">DATASET DISTRIBUTION & PARTITIONS</h4>
                </div>
                <p style="color: #e9d5ff; font-size: 0.98rem; line-height: 1.6;">
                    The forensic dataset encompasses a comprehensive scale of <b>45,035 facial crops</b> (~45,000 total images: 23,171 authentic human faces vs 21,864 synthetic manipulated faces) partitioned across Celeb-DF v2 and CACD:
                </p>
                <ul style="color: #d8b4fe; font-size: 0.95rem; line-height: 1.8;">
                    <li><b>GPU Training Split:</b> 28,942 images (~28,000 samples: 14,471 authentic real + 14,471 reconstructed synthetic).</li>
                    <li><b>Disjoint Validation Split:</b> 7,274 images (~7,000 samples: 3,637 authentic real + 3,637 reconstructed synthetic).</li>
                    <li><b>Held-Out Test Split (CACD):</b> 8,819 images (~8,800 samples: 5,063 authentic CACD + 3,756 face-swapped deepfakes).</li>
                    <li><b>Total Unified Dataset Pool:</b> 45,035 total images.</li>
                </ul>
                <div style="margin-top: 10px; padding: 10px 14px; background: rgba(0, 245, 212, 0.08); border-left: 3px solid #00f5d4; border-radius: 6px; font-size: 0.88rem; color: #a7f3d0;">
                    <b>GPU Training Hardware Protocol:</b> When training on GPU hardware, <code>train.py</code> processes the <b>28,942 training samples</b> out of the <b>45,035 total dataset</b>, followed by evaluation on the <b>8,819 held-out test samples</b>.
                </div>
            </div>

            <div class="cyber-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <h4 style="color: #00f5d4; margin: 0;">VIDEO-LEVEL DISJOINT SPLITS</h4>
                </div>
                <p style="color: #e9d5ff; font-size: 0.98rem; line-height: 1.6;">
                    To prevent data leakage caused by identical background frames or temporal correlations, the validation split is strictly segregated at the <b>source-video level</b>:
                </p>
                <p style="color: #c084fc; font-size: 0.92rem; line-height: 1.5;">
                    4 out of 5 videos per identity are allocated for training, while the remaining video is held out exclusively for validation. Paired original and reconstructed frames are kept together in identical splits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_doss2:
        st.markdown(
            """
            <div class="cyber-card" style="margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <h4 style="color: #bd00ff; margin: 0;">MODEL ARCHITECTURE COMPARISON</h4>
                </div>
                <div style="margin-bottom: 14px;">
                    <span style="font-family: 'Orbitron'; color: #ff77e9; font-weight: 700;">1. MobileNetV3-Small (1.52M params)</span>
                    <p style="color: #e9d5ff; font-size: 0.92rem; margin: 2px 0 10px 0;">
                        Employs hardware-aware NAS, inverted residuals, and linear bottlenecks. Ideal for mobile or edge deployment with an impressive ~2.44 ms per-sample inference.
                    </p>
                </div>
                <div style="margin-bottom: 14px;">
                    <span style="font-family: 'Orbitron'; color: #d946ef; font-weight: 700;">2. EfficientNet-B0 (4.01M params)</span>
                    <p style="color: #e9d5ff; font-size: 0.92rem; margin: 2px 0 10px 0;">
                        Compound scaling coefficient jointly optimizes network depth, channel width, and input resolution. Attains 92.21% accuracy and 97.51% ROC-AUC on CACD test.
                    </p>
                </div>
                <div>
                    <span style="font-family: 'Orbitron'; color: #a855f7; font-weight: 700;">3. ResNet-50 (23.51M params)</span>
                    <p style="color: #e9d5ff; font-size: 0.92rem; margin: 2px 0 0 0;">
                        50-layer deep residual network with identity shortcut connections preventing vanishing gradients. Highest overall test ROC-AUC (97.66%) and fake recall (94.33%).
                    </p>
                </div>
            </div>

            <div class="cyber-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <h4 style="color: #ff007f; margin: 0;">INPUT NORMALIZATION SPECIFICATION</h4>
                </div>
                <p style="color: #e9d5ff; font-size: 0.92rem; line-height: 1.5;">
                    All facial images undergo bicubic interpolation to 224&times;224 resolution followed by standard ImageNet distribution normalization:
                </p>
                <code style="display: block; background: rgba(0,0,0,0.4); padding: 8px 12px; border-radius: 6px; color: #ff77e9; font-family: monospace; font-size: 0.85rem;">
                    mean = [0.485, 0.456, 0.406]<br>
                    std  = [0.229, 0.224, 0.225]
                </code>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid rgba(255, 42, 133, 0.25); color: #c084fc; font-family: 'Chakra Petch', sans-serif;">
        DEEPFAKE SENTINEL &bull; ADVANCED FORENSIC COMPUTING &bull; PINK & PURPLE NEON HUD EDITION
    </div>
    """,
    unsafe_allow_html=True,
)
