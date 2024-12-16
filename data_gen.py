import jax
import jax.numpy as jnp

def modular_inverse(a, mod):
    t, new_t = 0, 1
    r, new_r = mod, a
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        return None  # No inverse exists
    if t < 0:
        t = t + mod
    return t

def get_mini_batches(data, batch_size, key):
    data_size = len(data)
    num_batches = data_size // batch_size
    data = jax.random.permutation(key, data)
    for i in range(num_batches):
        yield data[i*batch_size:(i+1)*batch_size]
    if data_size % batch_size != 0:
        yield data[num_batches*batch_size:]

def generate_modular_arithmetic_data(operation, mod=113, vocab_size=115):
    a_values = jnp.repeat(jnp.arange(mod), mod)
    b_values = jnp.tile(jnp.arange(mod), mod)
    operation_symbol = mod
    
    if operation == 'addition':
        c_values = (a_values + b_values) % mod

    elif operation == 'subtraction':
        c_values = (a_values - b_values) % mod

    elif operation == 'multiplication':
        c_values = (a_values * b_values) % mod

    elif operation == 'division':
        a_list, b_list, c_list = [], [], []
        for a in range(mod):
            for b in range(1, mod):  # Skip b=0 to avoid division by zero
                inv_b = modular_inverse(b, mod)
                if inv_b is not None:
                    a_list.append(a)
                    b_list.append(b)
                    c_list.append((a * inv_b) % mod)
        a_values = jnp.array(a_list)
        b_values = jnp.array(b_list)
        c_values = jnp.array(c_list)

    elif operation == 'x2_plus_y2':
        c_values = (a_values**2 + b_values**2) % mod

    elif operation == 'x2_plus_xy_plus_y2':
        c_values = (a_values**2 + a_values * b_values + b_values**2) % mod

    else:
        raise ValueError("Invalid operation specified")

    equal_symbol = jnp.full(a_values.shape, mod+1)

    sequences = jnp.stack([a_values, jnp.full(a_values.shape, operation_symbol), b_values, equal_symbol, c_values], axis=1)

    return sequences

def generate_modular_arithmetic_data2(operation, mod=113, vocab_size=115):
    a_values = jnp.arange(mod)
    b_values = jnp.arange(mod)
    a_mesh, b_mesh = jnp.meshgrid(a_values, b_values)
    a_flat = a_mesh.flatten()
    b_flat = b_mesh.flatten()
    operation_symbol = mod

    if operation == 'addition':
        c_values = (a_flat + b_flat) % mod
    elif operation == 'subtraction':
        c_values = (a_flat - b_flat) % mod
    elif operation == 'multiplication':
        c_values = (a_flat * b_flat) % mod
    elif operation == 'division':
        # Avoid division by zero
        mask = b_flat != 0
        a_div = a_flat[mask]
        b_div = b_flat[mask]
        inv_b = jnp.array([pow(b, -1, mod) for b in b_div])  # Modular inverse
        c_values = (a_div * inv_b) % mod
        a_flat = a_div
        b_flat = b_div
    elif operation == 'x2_plus_y2':
        c_values = (a_flat**2 + b_flat**2) % mod
    elif operation == 'x2_plus_xy_plus_y2':
        c_values = (a_flat**2 + a_flat * b_flat + b_flat**2) % mod
    else:
        raise ValueError("Invalid operation specified")
    
    # Create unique tuples
    unique_tuples = jnp.unique(jnp.column_stack((a_flat, b_flat, c_values)), axis=0)
    a_values, b_values, c_values = unique_tuples.T
    
    equal_symbol = jnp.full(a_values.shape, mod+1)
    sequences = jnp.column_stack([a_values, jnp.full(a_values.shape, operation_symbol), b_values, equal_symbol, c_values])
    return sequences