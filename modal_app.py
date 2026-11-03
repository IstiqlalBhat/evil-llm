"""Evil LLM on Modal, so the fixed link always works: evil_server.py on a T4 like Kaggle's, switched on by the first
request and off again after 10 minutes without one, so an idle demo costs nothing. Modal's $30 of free usage a month is
about 50 hours of chatting. Deploy: modal deploy modal_app.py, then point the fixed link (kaggle/worker.js) at the URL
it prints (see README). The visitor log's Discord webhook is the Modal secret evil-llm's EVIL_DISCORD_WEBHOOK."""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORER = "https://github.com/lyramakesmusic/neuron-explorer"
COMMIT = "e566e83e3f9fe00f85438231a788f90e83e2af43"  # the explorer version kaggle/map3d.patch applies to

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "wamerican")  # wamerican: the English word list the map's neuron labels are checked against
    # the versions it runs with on the Mac
    .pip_install("torch==2.14.1", "transformers==5.20.1", "numpy==2.5.4", "fastapi==0.146.0", "uvicorn==0.54.0")
    .env({"HF_HOME": "/hf"})
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')\"")
    .env({"HF_HUB_OFFLINE": "1"})  # the weights are in the image: don't ask Hugging Face for a newer version each start
    .run_commands(f"git clone -q {EXPLORER} /evil-llm/neuron-explorer && git -C /evil-llm/neuron-explorer checkout -q {COMMIT}")
    .add_local_file(os.path.join(HERE, "kaggle", "map3d.patch"), "/evil-llm/map3d.patch", copy=True)
    .run_commands("git -C /evil-llm/neuron-explorer apply ../map3d.patch")
    .add_local_file(os.path.join(HERE, "evil_server.py"), "/evil-llm/evil_server.py")
    .add_local_dir(os.path.join(HERE, "ui"), "/evil-llm/ui")
)
app = modal.App("evil-llm", image=image)


# one GPU at most, like Kaggle: visitors take turns (the server writes one reply at a time), and the bill can't multiply
@app.function(gpu="T4", secrets=[modal.Secret.from_name("evil-llm")], max_containers=1, scaledown_window=10 * 60,
              timeout=30 * 60)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/evil-llm")
    import evil_server  # loads the model onto the GPU
    return evil_server.app
