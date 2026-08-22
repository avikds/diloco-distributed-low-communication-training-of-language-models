"""
DiLoCo: Distributed Low-Communication Training of Language Models

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_model_params
import numpy as np

def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    """
    Initialize parameters for a 2-layer MLP.

    Shapes:
        W1: (input_dim, hidden_dim)
        b1: (hidden_dim,)
        W2: (hidden_dim, output_dim)
        b2: (output_dim,)

    All parameters are np.float64 arrays.
    Weights are initialized from a seeded NumPy RNG, while
    biases are initialized to zero.
    """
    rng = np.random.default_rng(seed)

    params = {
        "W1": rng.standard_normal((input_dim, hidden_dim)).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": rng.standard_normal((hidden_dim, output_dim)).astype(np.float64),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }

    return params

# Step 2 - relu
def relu(x):
    """Apply element-wise Rectified Linear Unit (ReLU)."""
    return np.maximum(x, 0)

# Step 3 - model_forward
def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    z1 = x @ params["W1"] + params["b1"]
    h1 = relu(z1)
    logits = h1 @ params["W2"] + params["b2"]

    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "logits": logits,
    }

    return logits, cache

# Step 4 - softmax
def softmax(logits):
    """Compute numerically stable row-wise softmax."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Step 5 - cross_entropy_loss
def cross_entropy_loss(logits, labels):
    """Compute mean numerically-stable cross-entropy loss."""
    # Numerically stable log-sum-exp:
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = (
        max_logits
        + np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
    )

    # Log probability of the true class
    true_class_logits = logits[np.arange(logits.shape[0]), labels]
    true_class_log_probs = true_class_logits - log_sum_exp[:, 0]

    # Mean negative log-likelihood
    return -np.mean(true_class_log_probs)

# Step 6 - model_backward
def model_backward(params, cache, labels):
    """Compute gradients of the mean softmax cross-entropy loss."""
    x = cache["x"]
    z1 = cache["z1"]
    h1 = cache["h1"]
    logits = cache["logits"]

    batch_size = x.shape[0]

    # Softmax probabilities
    probs = softmax(logits)

    # Gradient of cross-entropy w.r.t. logits
    d_logits = probs.copy()
    d_logits[np.arange(batch_size), labels] -= 1.0
    d_logits /= batch_size

    # Output layer: logits = h1 @ W2 + b2
    dW2 = h1.T @ d_logits
    db2 = np.sum(d_logits, axis=0)

    # Backprop through output layer
    dh1 = d_logits @ params["W2"].T

    # Backprop through ReLU
    dz1 = dh1 * (z1 > 0)

    # Hidden layer: z1 = x @ W1 + b1
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

# Step 7 - init_adamw_state
def init_adamw_state(params):
    """Initialize AdamW first/second moment state and step counter."""
    m = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    v = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    return {
        "m": m,
        "v": v,
        "t": 0,
    }

# Step 8 - update_adam_moments (not yet solved)
# TODO: implement

# Step 9 - bias_correct_moments (not yet solved)
# TODO: implement

# Step 10 - adam_param_step (not yet solved)
# TODO: implement

# Step 11 - decoupled_weight_decay (not yet solved)
# TODO: implement

# Step 12 - clone_params (not yet solved)
# TODO: implement

# Step 13 - scale_params (not yet solved)
# TODO: implement

# Step 14 - subtract_params (not yet solved)
# TODO: implement

# Step 15 - average_params (not yet solved)
# TODO: implement

# Step 16 - iid_shard_dataset (not yet solved)
# TODO: implement

# Step 17 - noniid_shard_dataset (not yet solved)
# TODO: implement

# Step 18 - sample_worker_batch (not yet solved)
# TODO: implement

# Step 19 - local_train_step (not yet solved)
# TODO: implement

# Step 20 - inner_train_worker (not yet solved)
# TODO: implement

# Step 21 - init_outer_optimizer (not yet solved)
# TODO: implement

# Step 22 - update_outer_momentum (not yet solved)
# TODO: implement

# Step 23 - nesterov_param_update (not yet solved)
# TODO: implement

# Step 24 - compute_outer_gradient (not yet solved)
# TODO: implement

# Step 25 - run_diloco_round (not yet solved)
# TODO: implement

# Step 26 - train_diloco (not yet solved)
# TODO: implement

# Step 27 - train_synchronous_baseline (not yet solved)
# TODO: implement

# Step 28 - evaluate_loss (not yet solved)
# TODO: implement

# Step 29 - classification_accuracy (not yet solved)
# TODO: implement

# Step 30 - communication_savings (not yet solved)
# TODO: implement

