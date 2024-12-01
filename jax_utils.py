import einops
import jax

def random_tree(tree_definition, key):
    keys = jax.random.split(key, tree_definition.num_leaves)
    return jax.tree.unflatten(tree_definition, keys)

def tree_stack(trees):
    return jax.tree.map(lambda *arrays: jax.numpy.stack(arrays), *trees)

def tree_pack(trees, pattern):
    return jax.tree.map(lambda *arrays: einops.pack(arrays, pattern)[0], *trees)
