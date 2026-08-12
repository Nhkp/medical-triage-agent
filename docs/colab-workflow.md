# Google Colab T4 workflow

Use `notebooks/colab_training.ipynb` when Kaggle is inconvenient or when VSCode is attached to
a Colab Jupyter runtime.

## Runtime

- Select a GPU runtime, preferably T4.
- The project targets Python 3.12+; the notebook stops early on older runtimes.
- Do not run `uv sync --extra training` in Colab. Colab already provides a CUDA PyTorch build,
  and reinstalling Torch can waste disk or break the runtime.
- Install only the missing training dependencies with `pip`, as shown in the notebook.

## Secrets

Add a Colab secret named `HF_TOKEN` with read access to
`Lokhidor/medical-triage-dataset`. The notebook can also prompt for the token with `getpass`,
but storing it as a Colab secret is preferred.

## Flow

1. Clone or update the GitHub repo under `/content/medical-triage-agent`.
2. Install missing dependencies without replacing Torch.
3. Download the private Hugging Face dataset with `huggingface_hub.snapshot_download`.
4. Run dataset audit and summary.
5. Run the tiny SFT smoke job on T4.
6. Increase sample counts only after the smoke run is stable.

Generated datasets, adapters, checkpoints, and evaluation files stay in the Colab runtime or
are pushed to private Hugging Face repos. Do not commit them to git.
