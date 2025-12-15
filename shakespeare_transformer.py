"""
Shakespeare Transformer Text Generator

Trains a small Transformer-based language model on Shakespeare text using
TensorFlow/Keras and generates text from a given prompt.
"""

# ======================
# Imports
# ======================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    MultiHeadAttention,
    Dense,
    LayerNormalization,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping


# ======================
# Dataset
# ======================

# Download Shakespeare dataset
path_to_file = get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)
text = open(path_to_file, "rb").read().decode(encoding="utf-8")

# Quick preview
print(text[:1000])

# Preprocess the dataset
vocab_size = 10000
seq_length = 100

# Adapt TextVectorization to full text
vectorizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
)
text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1)
vectorizer.adapt(text_ds)

# Vectorize the text as a single 1D sequence of tokens
vectorized_text = vectorizer([text])[0]
print("Vectorized text shape:", vectorized_text.shape)
print("First 10 vectorized tokens:", vectorized_text.numpy()[:10])


# ======================
# Create input and target sequences
# ======================

def create_sequences(token_ids: np.ndarray, seq_length: int):
    """
    Create input and target sequences using a sliding window over 1D token_ids.
    """
    input_seqs = []
    target_seqs = []

    for i in range(len(token_ids) - seq_length):
        input_seq = token_ids[i : i + seq_length]
        target_seq = token_ids[i + 1 : i + seq_length + 1]
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

    return np.array(input_seqs), np.array(target_seqs)


X, Y = create_sequences(vectorized_text.numpy(), seq_length)

print("Number of sequences generated:", len(X))
print("Sample input sequence:", X[0] if len(X) > 0 else "No sequences generated")

# Safety checks
assert X.size > 0, "Input data X is empty"
assert Y.size > 0, "Target data Y is empty"

X = tf.convert_to_tensor(X, dtype=tf.int32)
Y = tf.convert_to_tensor(Y, dtype=tf.int32)
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)


# ======================
# Build the Transformer Model
# ======================

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        # Self-attention block
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]
        self.dense = Dense(vocab_size)  # logits over vocabulary

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def positional_encoding(self, seq_length, embed_dim):
        positions = np.arange(seq_length)[:, np.newaxis]
        dims = np.arange(embed_dim)[np.newaxis, :]
        angle_rads = self.get_angles(positions, dims, embed_dim)

        # Apply sin to even indices in the array; cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]

        x = self.embedding(inputs)
        x = x + self.pos_encoding[:, :seq_len, :]

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)

        output = self.dense(x)  # (batch, seq_len, vocab_size) logits
        return output


# Hyperparameters
embed_dim = 256
num_heads = 4
ff_dim = 512
num_layers = 4

# Build the Transformer model
model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length)

# Build the model by passing a dummy input
_ = model(tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32))

# Compile the model (note: from_logits=True because Dense has no softmax)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn)

# Show model summary
model.summary()


# ======================
# Train the Transformer model
# ======================

# Use a subset to keep training time reasonable
X_train = X[:10000]
Y_train = Y[:10000]

# Early stopping if loss does not improve
early_stopping = EarlyStopping(
    monitor="loss", patience=2, restore_best_weights=True
)

history = model.fit(
    X_train,
    Y_train,
    epochs=2,
    batch_size=32,
    callbacks=[early_stopping],
)

# Plot training loss
plt.plot(history.history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


# ======================
# Text generation
# ======================

def generate_text(model, start_string, num_generate=100, temperature=1.0):
    """
    Generate text autoregressively from a trained Transformer model.

    Args:
        model: Trained TransformerModel.
        start_string: Prompt text to start generation.
        num_generate: Number of tokens to generate.
        temperature: Softmax temperature for sampling (lower = more greedy).

    Returns:
        Generated text string (prompt + generated tokens).
    """
    # Vectorize the start string
    input_eval = vectorizer([start_string]).numpy()  # shape (1, L)

    # Pad or truncate to seq_length
    if input_eval.shape[1] < seq_length:
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
    elif input_eval.shape[1] > seq_length:
        input_eval = input_eval[:, -seq_length:]

    input_eval = tf.convert_to_tensor(input_eval, dtype=tf.int32)

    text_generated = []
    vocab = vectorizer.get_vocabulary()

    for _ in range(num_generate):
        # Get logits for all positions in the sequence: (1, seq_len, vocab_size)
        logits = model(input_eval, training=False)

        # Use only the last time step: (1, vocab_size)
        logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / float(temperature)

        # Sample from the distribution
        predicted_id = tf.random.categorical(logits, num_samples=1)[0, 0].numpy()

        # Append predicted token
        text_generated.append(vocab[predicted_id])

        # Update input_eval to include the new token, keeping the last seq_length tokens
        input_eval_np = input_eval.numpy()
        input_eval_np = np.append(input_eval_np, [[predicted_id]], axis=1)
        input_eval_np = input_eval_np[:, -seq_length:]
        input_eval = tf.convert_to_tensor(input_eval_np, dtype=tf.int32)

    return start_string + " " + " ".join(text_generated)


if __name__ == "__main__":
    start_string = "To be, or not to be"
    generated_text = generate_text(model, start_string, temperature=0.7)
    print("\n=== Generated Text ===")
    print(generated_text)
