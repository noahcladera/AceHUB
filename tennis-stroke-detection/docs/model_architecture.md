# Model Architecture

The tennis stroke detection model uses sequential data (pose landmarks over time) to classify each frame or segment into specific stroke types or a "no-stroke" class.

## LSTM Backbone
- **Layers**: 2 LSTM layers with 256 hidden units each
- **Dropout**: 0.2 to reduce overfitting
- **Bidirectional**: True, to capture forward/backward temporal context

## Transformer Alternative
We also experiment with a Transformer-based approach. The input embeddings are the pose landmarks from each frame. 
- **Attention Heads**: 8
- **Embedding Dim**: 256
- **Positional Encoding**: Standard sine/cosine approach

## Classification Head
- A fully connected layer projecting from the hidden size to the output classes (e.g., stroke vs. no-stroke, or stroke types).
- Loss function is cross-entropy.

---

For more details, see the `model_config.yaml` and refer to each submodule in `models/`:
- `models/base.py` – The base class containing shared logic.
- `models/feature_extractors/pose_encoder.py` – Encodes pose landmarks into feature vectors.
- `models/sequence/lstm.py` – Implementation of the LSTM-based sequence model.
- `models/sequence/transformer.py` – Implementation of the Transformer-based sequence model.