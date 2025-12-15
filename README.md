# Shakespeare Transformer Text Generator

This project trains a small Transformer-based language model on Shakespeare text using TensorFlow/Keras, then generates new text given a prompt.

It demonstrates:

- Text preprocessing with `TextVectorization`
- Sequence creation for language modeling
- A custom Transformer block (Multi-Head Attention + feedforward)
- Training with `SparseCategoricalCrossentropy`
- Autoregressive text generation with temperature sampling

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

pip install -r requirements.txt
