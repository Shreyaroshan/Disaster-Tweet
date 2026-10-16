# Team model handoff

The repository administrator pushes the archive and shared infrastructure first. Each assigned member reviews, runs, explains and pushes their model implementation, final configuration, methodology card, results and artifact manifest. All implementations and the technical report were initially AI-assisted; a push alone is not evidence that the member wrote or reviewed the implementation. Record actual changes and review work in the contribution statement.

| Assigned reviewer | Model | Files to push from the handoff package |
|---|---|---|
| Member 1 (user) | CNN | `src/disaster_tweets/models/cnn.py`, `configs/cnn.json`, `docs/models/cnn.md`, `results/final_2026_10_16/cnn/`, any packaged replication/retained results and training log, `artifacts/models/cnn/manifest.json` |
| Member 2 | MLP | `src/disaster_tweets/models/mlp.py`, `configs/mlp.json`, `docs/models/mlp.md`, `results/final_2026_10_16/mlp/`, any packaged replication/retained results and training log, `artifacts/models/mlp/manifest.json` |
| Member 3 | Encoder-only Transformer | `src/disaster_tweets/models/transformer.py`, `configs/transformer.json`, `docs/models/transformer.md`, `results/final_2026_10_16/transformer/`, any packaged replication/retained results and training log, `artifacts/models/transformer/manifest.json` |
| Member 4 | LSTM | `src/disaster_tweets/models/lstm.py`, `configs/lstm.json`, `docs/models/lstm.md`, `results/final_2026_10_16/lstm/`, any packaged replication/retained results and training log, `artifacts/models/lstm/manifest.json` |

These assignments follow the latest user instruction and supersede the synopsis assignment table. The four named team members in the synopsis are Shreya Roshan, Tanisha Mathur, Adithya Kini and Arnab Sinha. The mapping of names to member numbers should be confirmed by the team before signatures; no unconfirmed mapping is treated as an individual contribution claim.

## Member workflow

1. Clone or pull the reorganized shared repository.
2. Create a personal branch, for example `member2-mlp`.
3. Copy the files from the appropriate handoff ZIP into the identical repository paths.
4. Set up the environment and run the preparation command in README.
5. Run `python src/train.py --model MODEL --run member_MODEL_review` and inspect the validation results. Model weights and datasets remain local.
6. Read the relevant methodology section, verify the forward pass, loss, padding handling and selection policy, and make any actual corrections needed.
7. Commit your model/configuration/methodology files, packaged model-specific results and artifact manifest, plus actual review changes. Use the exact git add command in your package instructions. Open a pull request or push the branch according to the team workflow.
8. After all four modules are integrated, run the shared tests. Test evaluation is a separate locked step; do not repeatedly use its score to tune the model.

The existing final metrics are already measured locally and included in the member packages. The combined comparison, figures, verification log and final report are held separately in `handoff/Integration_Report_Package.zip` until all four member contributions are integrated. The correction removes these outputs from the current GitHub tree, but the earlier commit remains in history. A member need not claim those earlier generated runs as their own work. Document their actual review, reproduction, debugging and writing.

## Files to keep local

Do not commit the virtual environment, raw/processed CSV datasets, GloVe downloads, NumPy embedding arrays or PyTorch checkpoints. Do not re-upload the old improve/tune/report scripts to the active src folder. Their historical copies belong in archive; the active code uses the shared preprocessing/training modules.

## Completion state on GitHub

The model registry uses lazy imports. The shared repository can be prepared before all model files arrive; training a missing model gives a clear awaiting-upload error. The four-model test suite and full inference become runnable after the members push all four modules. The administrator does not fabricate commits under the members' identities.
