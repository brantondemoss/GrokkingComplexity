import jax
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', False)

# Then before your model creation, add debug prints
print("Available devices:", jax.devices())
import jax.numpy as jnp
import numpy as onp
from jax import jit, value_and_grad, random, tree_util
import jax.lax as lax
import optax
import wandb
import bz2
import gzip
import io
from io import BytesIO
import pickle
import lzma
from scipy.stats import entropy
from multiprocessing import Pool
from PIL import Image
import matplotlib.pyplot as plt
import math
import sys
from bayes_opt import BayesianOptimization

import argparse


from model import TransformerDecoder, MLP
from data_gen import generate_modular_arithmetic_data, get_mini_batches

# Define helper functions for compression and logging
def gzip_compress(data, compresslevel=6):
    with io.BytesIO() as byte_io:
        with gzip.GzipFile(fileobj=byte_io, mode='w', compresslevel=compresslevel) as f:
            f.write(data.tobytes())
        return len(byte_io.getvalue())


def bzip2_compress(data):
    compressed = bz2.compress(data.tobytes())
    return len(compressed)

def normalize_matrix(matrix):
    min_val = onp.min(matrix)
    max_val = onp.max(matrix)
    return (matrix - min_val) / (max_val - min_val)

def coarse_grain(matrix, bins):
    min_val, max_val = matrix.min(), matrix.max()
    bin_width = (max_val - min_val) / bins
    normalized_matrix = (matrix - min_val) / (bin_width + epsilon)
    return onp.floor(normalized_matrix).astype(int)

def coarse_grain_gzip(matrix, bins=10):
    coarse_grained_matrix = coarse_grain(matrix, bins)
    return gzip_compress(coarse_grained_matrix)

def compute_histogram(data, bins=10):
    hist, _ = onp.histogram(data, bins=bins)
    return hist

def compute_entropy(hist):
    return entropy(hist + 1e-9)  # Add a small value to avoid log(0)

def get_layer_names(params_tree):
    layer_names = []
    def _recursive_helper(param_dict, prefix=""):
        for name, param in param_dict.items():
            full_name = f"{prefix}/{name}" if prefix else name
            if isinstance(param, dict) or isinstance(param, jax.tree_util.DictKey):
                _recursive_helper(param, full_name)
            else:
                layer_names.append(full_name)
    _recursive_helper(params_tree)
    return layer_names

def extract_matrices(param_dict):
    matrices = []
    def recurse(items):
        if isinstance(items, dict):
            for item in items.values():
                recurse(item)
        elif isinstance(items, onp.ndarray):
            matrices.append(items)
    recurse(param_dict)
    return matrices

def wandb_callback(train_loss, val_loss, train_accuracy, val_accuracy, params, 
                   inputs, targets, val_inputs, val_targets,step, noise_key=None):
    
    # choosing 1.3 as the base of the log, we are looking at everything on 
    # steps axis on log scale, this lpoks good
    noise_key, subkey = jax.random.split(noise_key)

    k = int(math.log(train_config["epochs"]) / math.log(1.3))
    steps = sorted(list(set([int(1.3**n) for n in range(k+1)])))
    if int(step) in steps:
        log_data = {
            'Losses.TrainLoss': float(train_loss),
            'Losses.TestLoss': float(val_loss),
            'Losses.Train Acc': float(train_accuracy),
            'Losses.Test Acc': float(val_accuracy)
        }
        
        epsilons = [10**-n for n in range(0,2)]
        algos = ["bzip2"] #gzip
        for epsilon in epsilons:
                complexities, bin_size = get_complexity(params, epsilon, inputs, targets, acc=False)
                log_data[f'bin_size@{epsilon}'] = float(bin_size)
                for algo in algos:
                    log_data[f'complexity/{algo}@{epsilon}'] = int(complexities[algo])

        '''
        epsilons = [10**-n for n in range(0,5)]
        algos = ["bzip2"] #gzip
        for epsilon in epsilons:
                complexities, bin_size = get_complexity(params, epsilon, inputs, targets, acc=False, noise=True, noise_key=noise_key)
                log_data[f'noise_bin_size@{epsilon}'] = float(bin_size)
                for algo in algos:
                    log_data[f'noise/{algo}_complexity@{epsilon}'] = int(complexities[algo])
        '''

        epsilons = [2**-n for n in range(-2,5)]
        algos = ["bzip2"] #gzip
        for epsilon in epsilons:
                #complexities, thresh = get_svd_complexity(params, epsilon, inputs, targets, acc=False)
                """complexities, thresh = optimize_svd_quantization(params, epsilon, inputs, targets, 
                              min_threshold=0.0, max_threshold=0.995, 
                              min_bin_size=1e-6, max_bin_size=1e-1, 
                              max_iters=50, tolerance=1e-6)"""
                
                # PUT THIS BACK
                complexities, thresh, coarse_train_loss, coarse_test_loss = bayes_optimize_svd_quantization(params, epsilon, inputs, targets, 
                              min_threshold=0.0, max_threshold=1.0, 
                              min_bin_size=1e-3, max_bin_size=1.0, 
                              n_iter=50, rng=noise_key)
                log_data[f'svd_threshold@{epsilon}'] = float(thresh)
                for algo in algos:
                    log_data[f'svd/{algo}_complexity@{epsilon}'] = int(complexities[algo])
                    log_data[f'svd/{algo}_trainloss@{epsilon}'] = float(coarse_train_loss)
                    log_data[f'svd/{algo}_testloss@{epsilon}'] = float(coarse_test_loss)

        layer_names = get_layer_names(params)
        gzip_sizes = []
        bzip2_sizes = []
        #zfp_sizes = []
        #zfp_sizes_noise = []
        entropies_dict = {}
        svd_entropies = []
        svd_variance = []
        param_svd_entropies = []
        rank_svd_entropies = []
        entropy_noise = []
        coarse_gzip_sizes_dict = {}
        coarse_gzip_sizes_noise = []
        norms = []
        frob_norms = []
        total_params = 0
        svd_ranks = 0
        noise_scale = train_config["noise_scale"]  # Adjust as necessary
        bin_precisions = [1e-1, 1e-2, 1e-3, 1e-4]  # Define your precisions here
        effective_ranks = []

        for precision in bin_precisions:
            entropies_dict[precision] = []
            coarse_gzip_sizes_dict[precision] = []

        for name, param in zip(layer_names, jax.tree_util.tree_leaves(params)):
            param = onp.array(param)
            
            #if len(param.shape) > 1:
            #    log_data[f"{name}_image_native"]=wandb.Image(param)
            #    log_data[f"{name}_image"]=wandb.Image(get_img_buf(param))
            num_params = param.size
            total_params += num_params

            any_nan = onp.isnan(param).any()
            frob_norm = onp.linalg.norm(param)
            norm = frob_norm/num_params
            
            frob_norms.append(frob_norm)
            norms.append(norm)

            param_range = param.max() - param.min()

            # only calculate SVD for matrices
            if len(param.shape) > 1:
                svd_ent, rank, variance, effective_rank = svd_entropy(param)
                log_data[f"{name}_svd_entropy"]= svd_ent
                log_data[f"{name}_effective_rank"]= effective_rank
                effective_ranks.append(effective_rank*num_params)
            log_data[f"{name}_histogram"] = wandb.Histogram(np_histogram=onp.histogram(param, bins=30))
            
            #print(name,'param max,min,range',param.max(),param.min(),param_range)
            '''
            for precision in bin_precisions:
                bins = max(int(param_range / precision),1)
                #print("num_bins=",bins,"at precision",precision)
                hist = compute_histogram(param, bins)
                param_entropy = compute_entropy(hist)
                entropies_dict[precision].append(param_entropy * num_params)
                log_data[f"{name}_entropy_{precision}precision"] = float(param_entropy)

                coarse_gzip_size = coarse_grain_gzip(param, bins)
                coarse_gzip_sizes_dict[precision].append(coarse_gzip_size)
                log_data[f"{name}_coarse_gzip_{precision}precision"] = int(coarse_gzip_size)
            

            # Calculate entropy and coarse gzip at noise bin threshold
            noise_bins = max(int((param.max() - param.min()) / noise_scale),1)
            hist_noise = compute_histogram(param, noise_bins)
            entropy_noise_value = compute_entropy(hist_noise) * num_params
            entropy_noise.append(entropy_noise_value)
            coarse_gzip_size_noise = coarse_grain_gzip(param, noise_bins)
            coarse_gzip_sizes_noise.append(coarse_gzip_size_noise)

            log_data[f"{name}_entropy_noise"] = float(compute_entropy(hist_noise))
            log_data[f"{name}_coarse_gzip_noise"] = int(coarse_gzip_size_noise)

            
            if len(param.shape) > 1:
                svd_ent, rank, variance, effective_rank = svd_entropy(param)
                svd_ranks += rank
                svd_entropies.append(svd_ent)
                svd_variance.append(variance)
                param_svd_entropies.append(svd_ent * num_params)
                rank_svd_entropies.append(svd_ent * rank)
                log_data[f"{name}_svd_entropy"] = float(svd_ent)
                log_data[f"{name}_svd_variance"] = float(variance)
            '''

            gzip_size = gzip_compress(param)
            bzip2_size = bzip2_compress(param)
            #zfp_size = zfp_compress(param)
            #zfp_size_noise = zfp_compress(param, tolerance=noise_scale)

            gzip_sizes.append(gzip_size)
            bzip2_sizes.append(bzip2_size)
            #zfp_sizes.append(zfp_size)
            #zfp_sizes_noise.append(zfp_size_noise)

            #log_data[f"{name}_histogram"] = wandb.Histogram(hist_noise.tolist())
            log_data[f"{name}_gzip"] = int(gzip_size)
            log_data[f"{name}_bzip2"] = int(bzip2_size)
            #log_data[f"{name}_zfp"] = int(zfp_size)
            #log_data[f"{name}_zfp_noise"] = int(zfp_size_noise)
            log_data[f"{name}_norm"] = float(norm)
            log_data[f"{name}_frob_norm"] = float(frob_norm)

        # Compute sums and mean
        log_data['total_gzip_size'] = sum(gzip_sizes)
        log_data['total_bzip2_size'] = sum(bzip2_sizes)
        #log_data['total_zfp_size'] = sum(zfp_sizes)
        #log_data['total_zfp_size_noise'] = sum(zfp_sizes_noise)
        #log_data['total_coarse_gzip_size_noise'] = sum(coarse_gzip_sizes_noise)
        #log_data['total_svd_entropy'] = sum(svd_entropies)
        #log_data['total_svd_entropy_rank'] = sum(rank_svd_entropies) / svd_ranks
        #log_data['total_svd_entropy_params'] = sum(param_svd_entropies) / total_params
        #log_data['total_svd_variance'] = sum(svd_variance)
        log_data['norms'] = wandb.Histogram(norms)
        log_data['frob_norms'] = wandb.Histogram(frob_norms)
        log_data['mean_norm'] = sum(norms) / len(norms)
        log_data['mean_frob_norm'] = sum(frob_norms) / len(frob_norms)
        log_data['effective_rank'] = sum(effective_ranks) / total_params
        

        #for precision in bin_precisions:
        #    log_data[f'total_coarse_gzip_size_{precision}precision'] = sum(coarse_gzip_sizes_dict[precision])
        #    log_data[f'weighted_entropy_sum_{precision}precision'] = sum(entropies_dict[precision]) / total_params

        #log_data['weighted_entropy_noise_sum'] = sum(entropy_noise) / total_params

        wandb.log(step=int(step), data=log_data)


def svd_entropy(matrix):
    s = onp.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    normalized_s = s / onp.sum(s)
    svd_entropy = -onp.sum(normalized_s * onp.log(normalized_s + 1e-9)) # Avoid log(0)
    effective_rank = onp.exp(svd_entropy) # https://core.ac.uk/download/pdf/147929764.pdf effective rank is exp(svd_ent)
    return svd_entropy, s.size, onp.var(s), effective_rank
        

def get_img_buf(weights):
    fig, ax = plt.subplots()
    cax = ax.matshow(weights, cmap='viridis')
    fig.colorbar(cax)
    ax.axis('off')

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
     # Load the buffer as an image
    img = Image.open(buf)

    # Convert the image to a numpy array
    img_array = onp.array(img)
    
    # Close the plot to free resources
    plt.close(fig)

    return img_array

@jit
def update_step(params, opt_state, inputs, targets):
    loss, grads = value_and_grad(clipped_cross_entropy_loss)(params, inputs, targets)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

@jit
def noisy_update_step(params, opt_state, noise_key, inputs, targets):
    noise_params = add_noise_to_params(params, noise_key, variance=train_config["noise_scale"])
    loss, grads = value_and_grad(clipped_cross_entropy_loss)(noise_params, inputs, targets)
    #loss, grads = value_and_grad(compute_noised_loss)(params, noise_params, inputs, targets)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

epsilon = 1e-9
def clipped_cross_entropy_loss(params, inputs, targets, cutoff=1.0, penalty_scale=10):
    logits = model.apply(params, inputs)
    probs = jax.nn.softmax(logits, axis=-1)

    # Add epsilon for numerical stability
    clipped_probs = jnp.clip(probs, epsilon, 1.0 - epsilon)
    log_probs = jnp.log(clipped_probs)

    # Only consider the last element of the sequence for each example in the batch
    log_probs = log_probs[:, -1, :]

    labels = jax.nn.one_hot(targets, vocab_size)
    loss = -jnp.sum(log_probs * labels) / labels.shape[0]

    # Add weight norm penalty
    if train_config["cutoff_penalty"]:
        #penalty = weight_norm_penalty(params, cutoff=train_config["norm_cutoff"])#, penalty_scale=penalty_scale)
        penalty = old_weight_norm_penalty(params, cutoff=train_config["norm_cutoff"])
        loss = loss + penalty
    if train_config["spectral_penalty"] != 0.0:
        penalty = spectral_entropy_penalty(params)
        loss = loss + train_config["spectral_penalty"]*penalty
    return loss

@jax.jit
def matrix_l2(param, cutoff):
    """Apply custom regularization to a parameter."""
    mask = jnp.abs(param) > cutoff
    return jnp.where(mask, (jnp.abs(param) - cutoff) ** 2, 0.0).sum()

@jax.jit
def weight_norm_penalty(params, cutoff):
    """Apply custom regularization to all parameters in the pytree."""
    return tree_util.tree_reduce(
        lambda x, y: x + y,
        tree_util.tree_map(lambda p: matrix_l2(p, cutoff), params),
        0.0
    )

def spectral_entropy_penalty(params, epsilon=1e-10):
    def spectral_entropy(param):
        if param.ndim >= 2:  # Only apply SVD on matrices
            U, S, Vt = jnp.linalg.svd(param, full_matrices=False)
            normalized_s = S / jnp.sum(S)
            return -jnp.sum(normalized_s * jnp.log(normalized_s + epsilon))
        return 0.0

    def reduce_fn(acc, param):
        return acc + spectral_entropy(param)

    num_matrices = sum(1 for param in jax.tree_util.tree_leaves(params) if param.ndim >= 2)
    total_penalty = jax.tree_util.tree_reduce(reduce_fn, params, initializer=0.0)
    mean_penalty = total_penalty / num_matrices if num_matrices > 0 else 0.0

    return mean_penalty


@jax.jit
def old_weight_norm_penalty(params, cutoff=0.1, penalty_scale=1): 
    #was cutoff=12.0 and penalty_scale = 1e-4 for non-avg case
    def penalty_fn(param):
        #norm = jnp.linalg.norm(param)
        #penalty = penalty_scale * (norm - cutoff)**2

        avg_norm = jnp.linalg.norm(param) / param.size
        penalty = penalty_scale * (avg_norm - cutoff)**2
        return penalty

    def zero_penalty_fn(param):
        return 0.0

    penalties = [
        lax.cond(jnp.linalg.norm(param)/param.size > cutoff, penalty_fn, zero_penalty_fn, param)
        for param in jax.tree_util.tree_leaves(params)
    ]

    return jnp.sum(jnp.array(penalties))


#@jax.jit
def compute_accuracy(params, inputs, targets):
    logits = model.apply(params, inputs)
    predictions = jnp.argmax(logits, axis=-1)

    # Only consider the last element of the sequence for each example in the batch
    predictions = predictions[:, -1]

    return jnp.mean(predictions == targets)

@jax.jit
def add_noise_to_params(params, key, variance=1e-3):
    def add_noise(param, key):
        noise = random.normal(key, shape=param.shape) * variance
        return param + noise
    
    leaves, treedef = jax.tree_util.tree_flatten(params)
    num_leaves = len(leaves)
    keys = random.split(key, num_leaves)
    noisy_leaves = [add_noise(leaf, k) for leaf, k in zip(leaves, keys)]
    noisy_params = jax.tree_util.tree_unflatten(treedef, noisy_leaves)
    return noisy_params

@jax.jit
def compute_noised_loss(params, noise_params, inputs, targets, cutoff=1.0, penalty_scale=1e-1):
    logits = model.apply(noise_params, inputs)
    probs = jax.nn.softmax(logits, axis=-1)

    # Add epsilon for numerical stability
    clipped_probs = jnp.clip(probs, epsilon, 1.0 - epsilon)
    log_probs = jnp.log(clipped_probs)

    # Only consider the last element of the sequence for each example in the batch
    log_probs = log_probs[:, -1, :]

    labels = jax.nn.one_hot(targets, vocab_size)
    loss = -jnp.sum(log_probs * labels) / labels.shape[0]

    # Add weight norm penalty
    penalty = weight_norm_penalty(params, cutoff=cutoff, penalty_scale=penalty_scale)
    return loss + penalty

def quantize_params(params, bin_size):
    def quantize(param):
        return jnp.round(param / bin_size) * bin_size

    return jax.tree_util.tree_map(quantize, params)

@jax.jit
def compute_loss(params, inputs, targets):
    logits = model.apply(params, inputs)
    probs = jax.nn.softmax(logits, axis=-1)
    #clipped_probs = jnp.clip(probs, epsilon, 1.0 - epsilon)
    log_probs = jnp.log(probs)
    log_probs = log_probs[:, -1, :]
    labels = jax.nn.one_hot(targets, vocab_size)
    loss = -jnp.sum(log_probs * labels) / labels.shape[0]
    return loss

def svd_truncate(matrix, threshold, components=False):
    m, n = matrix.shape
    assert len(matrix.shape) == 2, "Only 2D matrices are supported"

    U, S, Vt = jnp.linalg.svd(matrix, full_matrices=False)
    total_sum = jnp.sum(S)
    cumulative_sum = jnp.cumsum(S)
    threshold_value = threshold * total_sum
    num_singular_values = int(jnp.searchsorted(cumulative_sum, threshold_value))
    k = num_singular_values + 1  # Include the singular value at index `num_singular_values`

    if k >= len(S):
        return matrix  # Return original matrix if threshold is too high

    if components:
        mn = m * n
        if k * (m + n) < mn:
            truncated_U = U[:, :k]
            truncated_S = S[:k]
            truncated_Vt = Vt[:k, :]
            return truncated_U, truncated_S, truncated_Vt
        else:
            return matrix
    else:
        truncated_S = S.at[k:].set(0)
        truncated_matrix = U @ jnp.diag(truncated_S) @ Vt
        return truncated_matrix

def apply_svd_truncate(params, threshold, components=False):
    def truncate_fn(x):
        if x.ndim >= 2:  # Only apply SVD on matrices
            return svd_truncate(x, threshold, components=components)
        return x
    return tree_util.tree_map(truncate_fn, params)

def serialize_svd_components(params):
    svd_components = []
    def collect_svd_components(x):
        if x[1] is not None:  # Only collect if it's an SVD tuple
            svd_components.append(x)
    tree_util.tree_map(collect_svd_components, params)
    return svd_components

def get_initial_max_bin_size(params):
    max_value = 2*max(jnp.max(jnp.abs(param)) for param in jax.tree_util.tree_leaves(params))
    return max_value

def get_compressed_sizes(obj):

    buffer = BytesIO()
    results = {}
    params = jax.tree_util.tree_flatten(obj)
    # gzip compression
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write(pickle.dumps(params))

    results['gzip'] = len(buffer.getvalue())
    buffer.seek(0)
    buffer.truncate(0)
    
    # bzip2 compression
    compressed_data = bz2.compress(pickle.dumps(params))
    results['bzip2'] = len(compressed_data)
    buffer.seek(0)
    buffer.truncate(0)
    
    return results

def get_complexity(params, epsilon, inputs, targets, initial_bin_size=None, 
                   max_iters=50, tolerance=1e-12, acc=False, noise=False, noise_key=None):
    if acc:
        original_loss = compute_accuracy(params, inputs, targets)
    else:
        original_loss = compute_loss(params, inputs, targets)
    
    min_bin_size = 1e-12  # Smallest bin size to consider
    max_bin_size = get_initial_max_bin_size(params)  # Dynamically set based on parameter values
    
    

    if initial_bin_size is not None:
        best_bin_size = initial_bin_size
    else:
        best_bin_size = min_bin_size

    #print(f"Initial original loss: {original_loss}, max bin size: {max_bin_size}, best bin size {best_bin_size}" )

    #log_min_bin_size = jnp.log(min_bin_size)
    #log_max_bin_size = jnp.log(max_bin_size)
    
    for _ in range(max_iters):
        mid_bin_size = (min_bin_size + max_bin_size) / 2
        quantized_params = quantize_params(params, mid_bin_size)
        if noise:
            quantized_params = add_noise_to_params(quantized_params, noise_key, variance=train_config["noise_scale"])
        if acc:
            quantized_loss = compute_accuracy(quantized_params, inputs, targets)
        else:
            quantized_loss = compute_loss(quantized_params, inputs, targets)

        #print(f"{_}: Bin:{mid_bin_size:.2f}, Loss:{quantized_loss:.5f}")
        
        if jnp.abs(original_loss - quantized_loss) < epsilon:
            best_bin_size = mid_bin_size
            min_bin_size = mid_bin_size
        else:
            max_bin_size = mid_bin_size
        
        if max_bin_size - min_bin_size < tolerance:
            break

    quantized_params = jax.tree_util.tree_map(lambda x: jnp.where(x == -0.0, 0.0, x), quantize_params(params, best_bin_size))
    return get_compressed_sizes(quantized_params), best_bin_size

def print_param_shapes(params, message):
    print(f"\n{message}:")
    def print_fn(path, x):
        print(f"{path}: {x.shape}")
    tree_util.tree_map_with_path(print_fn, params)

def get_svd_complexity(params, epsilon, inputs, targets, initial_threshold=0.99, 
                       max_iters=50, tolerance=1e-3, acc=False):
    if acc:
        original_loss = compute_accuracy(params, inputs, targets)
    else:
        original_loss = compute_loss(params, inputs, targets)
    
    min_threshold = 0.0
    max_threshold = 1.0
    
    best_threshold = initial_threshold
    
    for _ in range(max_iters):
        mid_threshold = (min_threshold + max_threshold) / 2
        svd_params = apply_svd_truncate(params, mid_threshold)
        #print_param_shapes(svd_params, "SVD Truncated Params")
        
        if acc:
            svd_loss = compute_accuracy(svd_params, inputs, targets)
        else:
            svd_loss = compute_loss(svd_params, inputs, targets)
        
        if svd_loss - original_loss < epsilon: #originally: if jnp.abs(original_loss - svd_loss) < epsilon:
            best_threshold = mid_threshold
            max_threshold = mid_threshold
        else:
            min_threshold = mid_threshold
        
        if max_threshold - min_threshold < tolerance:
            break
    
    svd_params = apply_svd_truncate(params, best_threshold, components=True)
    #svd_components = serialize_svd_components(svd_params)
    results = get_compressed_sizes(svd_params)
    #results['threshold'] = best_threshold
    return results, best_threshold



#####################################     NEW SVD CODE         ############################################

def svd_truncate(matrix, threshold):
    U, S, Vt = jnp.linalg.svd(matrix, full_matrices=False)
    total_sum = jnp.sum(S)
    cumulative_sum = jnp.cumsum(S)
    k = int(jnp.searchsorted(cumulative_sum, threshold * total_sum))
    return U[:, :k], S[:k], Vt[:k, :]

def quantize(x, bin_size):
    rounded = jnp.round(x / bin_size) * bin_size
    rounded = jnp.where(rounded == -0.0, 0.0, rounded)
    return rounded

def apply_svd_truncate_quantize(params, threshold, bin_size, components=False):
    def truncate_quantize_fn(x):
        if jnp.any(jnp.isnan(x)):
            print("Nans in array")
            print("Original:", x)
        if x.ndim >= 2:
            m, n = x.shape
            U, S, Vt = svd_truncate(x, threshold)
            k = len(S)
            if k * (m + n) < m * n:
                #print('original matrix shape:', x.shape)
                #print('SVD components:', U.shape, S.shape, Vt.shape)
                if components:
                    return quantize(U @ jnp.diag(S), bin_size), quantize(Vt, bin_size)
                    #return quantize(U, bin_size), quantize(S, bin_size), quantize(Vt, bin_size)
                else:
                    return quantize(U @ jnp.diag(S), bin_size) @ quantize(Vt, bin_size)
        return quantize(x, bin_size)
    return tree_util.tree_map(truncate_quantize_fn, params)

def optimize_svd_quantization(params, epsilon, inputs, targets, 
                              min_threshold=0.0, max_threshold=0.995, 
                              min_bin_size=1e-6, max_bin_size=1e-1, 
                              max_iters=50, tolerance=1e-6):
    original_loss = compute_loss(params, inputs, targets)
    best_threshold = max_threshold
    best_bin_size = get_initial_max_bin_size(params)
    best_params = params

    #print(jax.tree_util.tree_flatten_with_path(params))

    for _ in range(int(max_iters//2)):
        mid_threshold = (min_threshold + max_threshold) / 2
        mid_bin_size = jnp.sqrt(min_bin_size * max_bin_size) # Geometric mean
        
        current_params = apply_svd_truncate_quantize(params, mid_threshold, mid_bin_size)
        current_loss = compute_loss(current_params, inputs, targets)
        
        if current_loss < original_loss + epsilon:
            best_threshold = mid_threshold
            max_threshold = mid_threshold
            best_params = current_params
        else:
            min_threshold = mid_threshold
        
        if (max_threshold - min_threshold < tolerance):
            break
    
    for _ in range(int(max_iters//2)):
        mid_bin_size = jnp.sqrt(min_bin_size * max_bin_size) # Geometric mean
        current_params = apply_svd_truncate_quantize(params, mid_threshold, mid_bin_size)
        current_loss = compute_loss_svd(current_params, inputs, targets)
        
        if current_loss < original_loss + epsilon:
            best_bin_size = mid_bin_size
            min_bin_size = mid_bin_size
            best_params = current_params
        else:
            max_bin_size = mid_bin_size
        
        if (max_bin_size / min_bin_size < 1 + tolerance):
            break
    
    best_params = apply_svd_truncate_quantize(params, best_threshold, best_bin_size, components=True)
    best_params = jax.tree_util.tree_map(lambda x: jnp.where(x == -0.0, 0.0, x), best_params)

    print(best_params)
    results = get_compressed_sizes_svd(best_params)
    return results, best_threshold
    #return best_params, best_threshold, best_bin_size

def get_compressed_sizes_svd(params):
    def prepare_for_compression(x):
        if isinstance(x, tuple):
            return jnp.concatenate([x[0].ravel(), x[1].ravel()])
        return x.ravel()
    flat_params = tree_util.tree_map(prepare_for_compression, params)
    return get_compressed_sizes(flat_params)


#################################################################################### BAYESIAN OPTIMIZATION

def bayes_optimize_svd_quantization(params, epsilon, inputs, targets,
                              min_threshold=0.0, max_threshold=1.0,
                              min_bin_size=1e-5, max_bin_size=1e-1,
                              n_iter=50, init_points=10, rng=None):
    original_loss = compute_loss(params, inputs, targets)

    def objective_function(bin_size_normalized, threshold_normalized):
        # Denormalize threshold (linear scale)
        threshold = threshold_normalized * (max_threshold - min_threshold) + min_threshold
        
        # Denormalize bin_size (logarithmic scale)
        bin_size = math.exp(bin_size_normalized * (math.log(max_bin_size) - math.log(min_bin_size)) + math.log(min_bin_size))

        print(f"Bin Size: {bin_size}, Threshold: {threshold}")
        
        current_params = apply_svd_truncate_quantize(params, threshold, bin_size)
        current_loss = compute_loss(current_params, inputs, targets)
        
        if current_loss > original_loss + epsilon or jnp.isnan(current_loss) or jnp.isinf(current_loss):
            #print('got loss:', current_loss)
            #print(current_params)
            return -3e5  # Return infinity if the loss constraint is violated
        
        compress_params = apply_svd_truncate_quantize(params, threshold, bin_size, components=True)
        compress_params = jax.tree_util.tree_map(lambda x: jnp.where(x == -0.0, 0.0, x), compress_params)
        compressed_size = get_compressed_sizes_svd(compress_params)["bzip2"]
        
        return -compressed_size
    # Define the parameter space (normalized to [0, 1])
    pbounds = {
        'bin_size_normalized': (0, 1),
        'threshold_normalized': (0, 1),
    }

    # Initialize and run the Bayesian optimization
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        #replace random_state with jax rng key we passed in
        random_state=int(rng[0])
    )

    optimizer.probe(
        params={'bin_size_normalized': 0.0, 'threshold_normalized': 1.0},
        lazy=False
    )
    optimizer.probe(
        params={'bin_size_normalized': 0.1, 'threshold_normalized': 0.98},
        lazy=False
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )

    # Denormalize the best parameters
    best_threshold = optimizer.max['params']['threshold_normalized'] * (max_threshold - min_threshold) + min_threshold
    best_bin_size = math.exp(optimizer.max['params']['bin_size_normalized'] * (math.log(max_bin_size) - math.log(min_bin_size)) + math.log(min_bin_size))

    # get loss of best params, on train and test data
    best_params = apply_svd_truncate_quantize(params, best_threshold, best_bin_size)
    inputs, targets = mini_batch[:, :-1], mini_batch[:, -1]
    train_loss = compute_loss(best_params, train_data[:, :-1], train_data[:, -1])
    test_loss = compute_loss(best_params, test_data[:, :-1], test_data[:, -1])

    

    # Apply the best parameters to get the final result
    best_params = apply_svd_truncate_quantize(params, best_threshold, best_bin_size, components=True)
    best_params = tree_util.tree_map(lambda x: jnp.where(x == -0.0, 0.0, x), best_params)

    results = get_compressed_sizes_svd(best_params)
    return results, best_threshold, train_loss, test_loss#, best_bin_size



####################################################################################



def getargs():
    # Create the parser
    parser = argparse.ArgumentParser(description="Get run options from cmd line")

    # Add the argument with type=int to ensure it's an integer
    parser.add_argument('--rng', type=int, help='rng seed', default=0)
    parser.add_argument('--epochs', type=int, help='num epochs', default=10000)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--wd', type=float, help='weight decay', default=0.0)
    parser.add_argument('--ours', type=float, help='our regularization method', default=0.0)
    parser.add_argument('--spectral', type=float, help='spectral entropy penalty', default=0.0)
    parser.add_argument('--noise', type=float, help='noise scale', default=0.0)
    parser.add_argument('--cutoff', type=float, help='param cutoff', default=None)
    parser.add_argument('--operation', type=int, choices=[0, 1, 2, 3, 4, 5], 
                        help="The type of modular arithmetic operation to generate data for. \n" +\
                        "0: addition, 1: subtraction, 2: multiplication, 3: division, 4: x2_plus_y2, 5: x2_plus_xy_plus_y2")
    parser.add_argument('--architecture', type=str, choices=["transformer", "mlp"], help='which architecture to use', default="transformer")
    parser.add_argument('--mod', type=int, default=113, help='mod value for modular arithmetic')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the number argument is provided and is an integer
    #if args.rng is not None:
    #    print(f"The provided rng seed is: {args.number}")
    #else:
    #    print("No rng seed argument provided.")
    #    
    return args

if __name__ == "__main__":
    args = getargs()
    # Configuration for the transformer and training

    if args.ours != 0.0:
        args.wd = 1.0
        args.spectral = 1e-1
        if args.operation == 0 or args.operation == 2:
            args.noise = 1e-2
        else:
            args.noise = 1e-3
    
    operation_map = {
        0: 'addition',
        1: 'subtraction',
        2: 'multiplication',
        3: 'division',
        4: 'x2_plus_y2',
        5: 'x2_plus_xy_plus_y2'
    }
    operation_name = operation_map[args.operation]
    
    train_fraction_map = {
        'addition': 0.2,
        'subtraction': 0.3,
        'multiplication': 0.2,
        'division': 0.3,
        'x2_plus_y2': 0.4,
        'x2_plus_xy_plus_y2': 0.6
    }
    train_fraction = train_fraction_map[operation_name]
    
    transformer_config = {
        "mod": args.mod, #113
        "vocab_size": args.mod+2, #115
        "d_model": 128,
        "d_head": 32,
        "d_mlp": 512,
        "nhead": 4,
        "n_layers": 2 if (args.operation == 1 or args.operation == 3) else 1,
    }
    
    mlp_config = {
        "mod": args.mod,
        "vocab_size": args.mod+2,
        "d_model": 128,
        "d_mlp": 512,
        "n_layers": 3,
        "fn_activation": "relu",
    }

    train_config = {
        "wandb_log": True,
        "epochs": args.epochs,
        "lr_init": args.lr,
        "noise_scale": args.noise,
        "norm_cutoff": args.cutoff,
        "w_decay": args.wd,
        "noise_reg": bool(args.noise > 0.0),
        "cutoff_penalty": bool(args.cutoff is not None),
        "rng_key": args.rng,
        "spectral_penalty": args.spectral,
        "dataset": operation_name,
        "architecture": args.architecture
    }
    
    # Initialize wandb
    wandb.init(project="grok-sept26")
    wandb.config.update(train_config)
    wandb.config.update(transformer_config)

    if train_config["architecture"] == "transformer":
        mod = transformer_config["mod"]
        vocab_size = transformer_config["vocab_size"]
        d_model = transformer_config["d_model"]
        d_head = transformer_config["d_head"]
        d_mlp = transformer_config["d_mlp"]
        nhead = transformer_config["nhead"]
        n_layers = transformer_config["n_layers"]
    elif train_config["architecture"] == "mlp":
        mod = mlp_config["mod"]
        vocab_size = mlp_config["vocab_size"]
        d_model = mlp_config["d_model"]
        d_mlp = mlp_config["d_mlp"]
        n_layers = mlp_config["n_layers"]
        fn_activation = mlp_config["fn_activation"]

    # Data Preparation
    data = generate_modular_arithmetic_data(operation_name, mod=mod, vocab_size=vocab_size)
    train_size = int(train_fraction * len(data))
    #data = generate_all_mod_addition_data(mod, vocab_size)
    data = jax.random.permutation(jax.random.PRNGKey(train_config["rng_key"]), data)
    train_data = data[:train_size]
    test_data = data[train_size:]
    mini_batch_size = min(512, len(train_data) // 2)

    # Initialize model and optimizer
    if train_config["architecture"] == "transformer":
        model = TransformerDecoder(d_model=d_model, d_head=d_head, d_mlp=d_mlp, nhead=nhead, vocab_size=vocab_size, n_layers=n_layers, layernorm=True)
    elif train_config["architecture"] == "mlp":
        model = MLP(d_model=d_model, d_mlp=d_mlp, vocab_size=vocab_size, n_layers=n_layers, fn_activation=fn_activation)
    rng = jax.random.PRNGKey(train_config["rng_key"])
    rng_noise = jax.random.PRNGKey(train_config["rng_key"])
    params = model.init(rng, jnp.array(train_data[:, :-1]))
    optimizer = optax.adamw(learning_rate=train_config["lr_init"], 
                            b1=0.9, 
                            b2=0.98, 
                            eps=1e-08,
                            weight_decay=train_config["w_decay"])
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(train_config["epochs"]):
        # Shuffle the training data at the beginning of each epoch
        rng, subkey = jax.random.split(rng)
        train_data = jax.random.permutation(subkey, train_data)
        
        for mini_batch in get_mini_batches(train_data, mini_batch_size, subkey):
            inputs, targets = mini_batch[:, :-1], mini_batch[:, -1]
            
            rng_noise, subkey = jax.random.split(rng_noise)
            if train_config["noise_reg"]:
                params, opt_state, train_loss = noisy_update_step(params, opt_state, subkey, inputs, targets)
            else:
                params, opt_state, train_loss = update_step(params, opt_state, inputs, targets)
                

        k = int(math.log(train_config["epochs"]) / math.log(1.3))
        steps = sorted(list(set([int(1.3**n) for n in range(k+1)])))
        if int(epoch) in steps:
            train_accuracy = compute_accuracy(params, train_data[:, :-1], train_data[:, -1])

            # Validation
            val_inputs, val_targets = test_data[:, :-1], test_data[:, -1]
            val_loss = clipped_cross_entropy_loss(params, val_inputs, val_targets)
            val_accuracy = compute_accuracy(params, val_inputs, val_targets)

            if train_config["wandb_log"]:
                wandb_callback(train_loss, val_loss, train_accuracy, val_accuracy, params, 
                            inputs=inputs, targets=targets, val_inputs=val_inputs, 
                            val_targets=val_targets,step=epoch, noise_key=subkey)