from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"
LOCAL_DIR = Path.home() / "models" / "GigaChat3-10B-A1.8B-bf16"


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("MODEL DOWNLOADED TO:", LOCAL_DIR)


if __name__ == "__main__":
    main()