"""Neural network forward pass — feedforward and Elman RNN.

Weights are flat arrays, unpacked inside the forward pass.
Compatible with vmap over population dimension.
"""

import jax.numpy as jnp


def sigmoid(x):
    x = jnp.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + jnp.exp(-x))


def nn_forward(inputs, weights, h_prev, params):
    """Forward pass for feedforward or Elman RNN.

    Args:
        inputs: (input_size,) sensor + vision values
        weights: (num_weights,) flat weight array
        h_prev: (hidden_size,) previous hidden state (zeros for feedforward)
        params: StaticParams

    Returns:
        (left, right): wheel commands in [-1, 1]
        h_new: updated hidden state
    """
    in_size = params.input_size
    hid = params.hidden_size
    out = params.output_size
    is_elman = params.is_elman

    # Unpack weights
    idx = 0

    # W_xh: input -> hidden
    W_xh = weights[idx:idx + in_size * hid].reshape(in_size, hid)
    idx += in_size * hid

    # W_hh: hidden -> hidden (Elman only, but always allocated for static shapes)
    W_hh = weights[idx:idx + hid * hid].reshape(hid, hid)
    idx += hid * hid

    # b_h: hidden bias
    b_h = weights[idx:idx + hid]
    idx += hid

    # W_ho: hidden -> output
    W_ho = weights[idx:idx + hid * out].reshape(hid, out)
    idx += hid * out

    # b_o: output bias
    b_o = weights[idx:idx + out]

    # Forward pass
    pre_h = inputs @ W_xh + b_h
    # Add recurrent contribution (zeroed out for feedforward via is_elman flag)
    recurrent = h_prev @ W_hh
    pre_h = pre_h + jnp.where(is_elman, recurrent, 0.0)

    # Hidden activation
    h_new = jnp.where(params.use_sigmoid, sigmoid(pre_h), jnp.tanh(pre_h))

    # Output
    pre_o = h_new @ W_ho + b_o
    output = jnp.where(params.use_sigmoid, sigmoid(pre_o), jnp.tanh(pre_o))

    left = jnp.clip(output[0], -1.0, 1.0)
    right = jnp.clip(output[1], -1.0, 1.0)

    return (left, right), h_new


def num_weights(input_size, hidden_size, output_size):
    """Total number of weights (always includes W_hh for static shape).

    Feedforward: W_hh is allocated but zeroed / ignored via is_elman flag.
    Elman: W_hh is used.
    """
    return (input_size * hidden_size +    # W_xh
            hidden_size * hidden_size +    # W_hh
            hidden_size +                  # b_h
            hidden_size * output_size +    # W_ho
            output_size)                   # b_o
