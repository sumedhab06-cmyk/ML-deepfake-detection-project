"""Create a small standalone Streamlit deployment ZIP from a trained checkpoint."""
import argparse
import shutil
import zipfile
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package one trained model into a standalone web app")
    parser.add_argument("--model", choices=("mobilenet_v3", "efficientnet_b0", "resnet50"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checkpoint = root / "outputs" / args.model / "best.pt"
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint {checkpoint}; run train.py first")
    target = root / "deployment_ready"
    target.mkdir(exist_ok=True)
    (target / "deepfake").mkdir(exist_ok=True)
    (target / "model").mkdir(exist_ok=True)
    for name in ("app.py", "requirements.txt"):
        shutil.copy2(root / "deployment_template" / name, target / name)
    for name in ("__init__.py", "data.py", "models.py"):
        shutil.copy2(root / "deepfake" / name, target / "deepfake" / name)
    shutil.copy2(checkpoint, target / "model" / "best.pt")
    (target / "README.md").write_text(
        "# DeepFake Face Image Detection Demo\n\n"
        "Run locally: `pip install -r requirements.txt`, then `streamlit run app.py`.\n\n"
        "For Streamlit Community Cloud, upload the contents of this folder to a GitHub repository, "
        "then create an app at share.streamlit.io using `app.py` as the entrypoint. "
        "Set Python to 3.11 or 3.12. Keep the `model/best.pt` checkpoint in the repository. "
        "This is an image classifier for cropped faces and cannot provide forensic certainty.\n",
        encoding="utf-8")
    archive = root / f"DeepFake_Deployment_{args.model}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for file in sorted(target.rglob("*")):
            if file.is_file():
                output.write(file, file.relative_to(target))
    print(f"Created {archive}. Upload its extracted contents (not the ZIP) to your app repository.")
