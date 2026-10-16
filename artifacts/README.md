# Local model artifacts

Selected PyTorch checkpoints are stored in models/MODEL/seed_SEED.pt after local training and finalization. They are excluded from Git. The small manifest JSON files record selected members, checksums, threshold and validation metrics. Search checkpoints are also stored locally under results/RUN/MODEL/CANDIDATE and are ignored.

To recreate weights on another computer, prepare the dataset/embeddings and run the shared training CLI. Each member uploads their artifact manifest and model-specific results; the combined figures are published after integration. Do not fabricate weights or copy literature scores into local result files.
