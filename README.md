# Kannada-English hate speech detection and target identification (HASTIKA @ ICON 2026)

Code for the working note *Ensembling TF-IDF and Multilingual Transformers for Hate Speech Detection and
Target Identification in Kannada-English Code-Mixed Text* (HASTIKA shared task at ICON 2026).

The systems average the class probabilities of a word + character TF-IDF logistic regression and fine-tuned
multilingual transformers (MuRIL, XLM-RoBERTa, multilingual BERT).

| Task | Models in the submitted ensemble | Test macro-F1 |
|---|---|---|
| A: Hate / Non-Hate | TF-IDF + MuRIL (weights 0.75 / 0.25) | 0.7915 |
| B: six target categories | TF-IDF + MuRIL + XLM-R + mBERT (weights 0.797 / 0.079 / 0.057 / 0.067) | 0.6520 |

Author: Suhaan Francis, Manipal Institute of Technology, Manipal Academy of Higher Education.

## Data

The HASTIKA data are **not included** and must not be redistributed (see the terms of the shared task).
Download the files from the organizers' repository, <https://github.com/shankarb14/Hastika-ICON2026>, and put
them in `data/`. See `data/README.md`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
# For GPU training install the CUDA build of PyTorch, e.g.:
# pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

The experiments were run with Python 3.12.10, PyTorch 2.6.0 (CUDA 12.4), Transformers 5.16.1, scikit-learn 1.9.0,
pandas 3.0.5 and NumPy 2.5.2 on a laptop with an NVIDIA RTX 4050 (6 GB).

## Pipeline

All scripts are run from the repository root. `--task` is `A` or `B`. A fixed 85/15 stratified split of the
training data (seed 42) is used as the development split; the transformer checkpoints and the ensemble weights
are selected on it.

### Quick check (about a minute)

```bash
python -m src.finetune train --task A --model google/muril-base-cased --train data/binary_train.csv \
       --out-dir outputs/smoke --epochs 1 --max-train-samples 64
```

### Task A

```bash
# TF-IDF: dev probabilities for the weight search, test probabilities from a model trained on all training data
python -m src.tfidf dev  --task A --train data/binary_train.csv --min-df 2 --out outputs/A_tfidf_dev.npy
python -m src.tfidf test --task A --train data/binary_train.csv --test data/binary_test.csv --out outputs/A_tfidf_test.npy

# MuRIL, trained in three stages on the same model (stages 1 and 2 continue from the last epoch)
python -m src.finetune train --task A --model google/muril-base-cased --train data/binary_train.csv \
       --out-dir outputs/A_muril_s1 --epochs 4 --lr 2e-5 --warmup-ratio 0.1
python -m src.finetune train --task A --model google/muril-base-cased --train data/binary_train.csv \
       --out-dir outputs/A_muril_s2 --init-checkpoint outputs/A_muril_s1/last.pt --epochs 2 --lr 1e-5
python -m src.finetune train --task A --model google/muril-base-cased --train data/binary_train.csv \
       --out-dir outputs/A_muril_s3 --init-checkpoint outputs/A_muril_s2/last.pt --epochs 6 --lr 1e-5
python -m src.finetune predict --task A --model google/muril-base-cased --checkpoint outputs/A_muril_s3/best.pt \
       --test data/binary_test.csv --out outputs/A_muril_test.npy

# Ensemble: search the weight on the dev split, then apply (the submission used 0.75 / 0.25)
python -m src.ensemble search --task A --train data/binary_train.csv \
       --dev-probs outputs/A_tfidf_dev.npy outputs/A_muril_s3/dev_probs_best.npy --out outputs/A_weights.json
python -m src.ensemble apply --task A --test data/binary_test.csv \
       --test-probs outputs/A_tfidf_test.npy outputs/A_muril_test.npy \
       --weights 0.75 0.25 --out outputs/A_predictions.csv
```

Note: the epoch count of stage 2 above is a placeholder; set it to the value used for your run.

### Task B

```bash
python -m src.tfidf dev  --task B --train data/multiclass_train.csv --out outputs/B_tfidf_dev.npy
python -m src.tfidf test --task B --train data/multiclass_train.csv --test data/multiclass_test.csv \
       --out outputs/B_tfidf_test.npy

# Transformers with class-weighted cross-entropy (MuRIL: 8 epochs, XLM-R: 15, mBERT: 8)
python -m src.finetune train --task B --model google/muril-base-cased --train data/multiclass_train.csv \
       --out-dir outputs/B_muril --epochs 8 --class-weights
python -m src.finetune train --task B --model xlm-roberta-base --train data/multiclass_train.csv \
       --out-dir outputs/B_xlmr --epochs 15 --class-weights
python -m src.finetune train --task B --model bert-base-multilingual-cased --train data/multiclass_train.csv \
       --out-dir outputs/B_mbert --epochs 8 --class-weights

python -m src.finetune predict --task B --model google/muril-base-cased --checkpoint outputs/B_muril/best.pt \
       --test data/multiclass_test.csv --out outputs/B_muril_test.npy
python -m src.finetune predict --task B --model xlm-roberta-base --checkpoint outputs/B_xlmr/best.pt \
       --test data/multiclass_test.csv --out outputs/B_xlmr_test.npy
python -m src.finetune predict --task B --model bert-base-multilingual-cased --checkpoint outputs/B_mbert/best.pt \
       --test data/multiclass_test.csv --out outputs/B_mbert_test.npy

# Ensemble (probability files and weights are in the order TF-IDF, MuRIL, XLM-R, mBERT)
python -m src.ensemble search --task B --train data/multiclass_train.csv \
       --dev-probs outputs/B_tfidf_dev.npy outputs/B_muril/dev_probs_best.npy \
                   outputs/B_xlmr/dev_probs_best.npy outputs/B_mbert/dev_probs_best.npy \
       --n-trials 3000 --out outputs/B_weights.json
python -m src.ensemble apply --task B --test data/multiclass_test.csv \
       --test-probs outputs/B_tfidf_test.npy outputs/B_muril_test.npy outputs/B_xlmr_test.npy outputs/B_mbert_test.npy \
       --weights 0.797 0.079 0.057 0.067 --out outputs/B_predictions.csv
```

### Scoring

Once gold labels are available (the organizers released them after the competition):

```bash
python -m src.evaluate --task A --pred outputs/A_predictions.csv --gold data/binary_test_gold.csv
python -m src.evaluate --task B --pred outputs/B_predictions.csv --gold data/multiclass_test_gold.csv
```

The prediction files have the `id,label` format required by the shared task.

## Notes on reproducibility

- The reported numbers were produced in Jupyter notebooks during the competition. The scripts here are a cleaned
  version of the same pipeline. The preprocessing, the development split and the TF-IDF models reproduce the
  notebook results exactly (for example, dev macro-F1 0.8097 for Task A with `--min-df 2` and 0.6303 for Task B).
- Transformer training is not bit-for-bit reproducible: the original runs did not fix the PyTorch seeds, GPU kernels
  are not fully deterministic, and each configuration was run once. Expect small differences, and re-tune the
  ensemble weights on your own dev probabilities.
- The Task A MuRIL checkpoint in the submission came from several continued runs in one notebook; the three-stage
  recipe above follows that history.
- The submitted systems did not use the released validation labels for training.

## Citation

Please cite the HASTIKA dataset paper and the shared task overview paper:

```bibtex
@article{kavatagi2025hastika,
  title={HASTIKA: hate speech and target identification in Kannada-English code-mixed text},
  author={Kavatagi, Sanjana and Rachh, Rashmi},
  journal={Language Resources and Evaluation},
  volume={59},
  number={3},
  pages={2811--2856},
  year={2025},
  publisher={Springer}
}
```

## License

Add a license for the code here. The HASTIKA data are covered by the terms of the shared task, not by this repository.
