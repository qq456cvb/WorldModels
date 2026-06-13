# World Models

A from-scratch implementation of **World Models** ([Ha & Schmidhuber, NeurIPS 2018](https://arxiv.org/abs/1803.10122) — interactive version at [worldmodels.github.io](https://worldmodels.github.io/)), applied to OpenAI Gym's `CarRacing-v0`.

The agent is split into the paper's three components — **V** (vision), **M** (memory), and **C** (controller):

| Component | File | Model |
| --- | --- | --- |
| **V** — Vision | `VAE.py` | Convolutional **VAE** that compresses each 64×64×3 frame into a 32-D latent `z` |
| **M** — Memory | `MDN-RNN.py` | **LSTM + Mixture Density Network** that predicts the next latent `z` given the current `z` and action |
| **C** — Controller | `CMA-ES.py` | **CMA-ES** optimizer used to train the (small) controller |

## How the pieces fit

1. **VAE (`VAE.py`)** — a 4-layer conv encoder produces a 32-D Gaussian latent (`mean`/`log_var`) with the reparameterization trick, and a conv-transpose decoder reconstructs the frame. Trained with reconstruction + KL loss on random `CarRacing-v0` rollouts. Running the file directly loads a checkpoint and shows live frame-vs-reconstruction windows.

2. **MDN-RNN (`MDN-RNN.py`)** — encodes rollout frames to `z` with the trained VAE (via a `FusedModel` that wires VAE + RNN together), then trains an LSTM (hidden size 256) whose MDN head outputs a mixture of `MIX_GAUSSIANS = 5` Gaussians per latent dimension over length-`SEQ_LEN = 20` sequences. Loss is the negative log-likelihood of the next latent under the predicted mixture.

3. **CMA-ES (`CMA-ES.py`)** — a self-contained NumPy implementation of Covariance Matrix Adaptation Evolution Strategy (with rank-µ update, evolution paths, and step-size control), the black-box optimizer the paper uses for the controller. Run directly, it sanity-checks itself on a convex quadratic test function.

## Setup & usage

Built on **tensorpack**, **TensorFlow 1.x** (`tf.contrib.slim` / `tf.contrib.rnn`), **OpenAI Gym** (`CarRacing-v0`), **OpenCV**, and **NumPy**.

```bash
# 1. Train the VAE (edit VAE.py to call train())
python VAE.py

# 2. Train the MDN-RNN on top of the VAE encoder
python MDN-RNN.py

# 3. Try the CMA-ES optimizer on its built-in test function
python CMA-ES.py
```

The MDN-RNN loads the VAE checkpoint from `train_log/auto_encoder/checkpoint`; train the VAE first. Checkpoints and logs are written under `train_log/`.

## Notes & scope

- This is a TensorFlow 1.x / tensorpack-era project and needs that legacy stack (`tf.contrib`) to run.
- The VAE and MDN-RNN (V and M) are implemented and trainable; the CMA-ES controller is provided as a standalone optimizer and is not yet wired into an end-to-end controller-training loop on the environment.
