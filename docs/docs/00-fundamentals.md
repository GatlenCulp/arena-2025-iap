# Chapter 0: Fundamentals [(source)](https://arena-chapter0-fundamentals.streamlit.app/)

> MLAB felt like I was doing actual AI safety technical research, rather than just impersonating what I thought an AI safety technical researcher would be doing

[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library

Author: [Callum McDougall](https://www.perfectlynormal.co.uk/)

## Notes on Style

I'm only taking notes on things I'm not familiar with.

## [0.0.1] Prerequisites - Core Concepts / Knowledge

### ReLU vs Sigmoids

Helps with vanishing gradient problem and moer efficient.

### Info Theory

Recommended book: [Elements of Info Theory](http://staff.ustc.edu.cn/~cgong821/Wiley.Interscience.Elements.of.Information.Theory.Jul.2006.eBook-DDU.pdf).

And to understand [KL Divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence)

<!-- TODO: I probably should read the above. Not super familiar with KL divergence. Also relies on understanding entropy. -->

Note: They recommend streamlit over Dash.

## [0.0.2] Prerequisites - Einops, Einsum, & Tensors

### Einops Exercises

### Broadcasting

**Broadcasting** - Copying along dimensions in element-wise operations to make sure they have the same shape

Ex:

1. Prepend "dummy dimensions" of size `1` to the tensor with less dimensions (ex: `data.shape = (N, k)` + vector `vec.shape = (k)`. To make sure num dims match on an operations between the two, `vec` becomes `(1, k)`)
1. Repeat along dimensions until shape matches for BOTH tensors (ex: `vec` is repeated `N` times to become `(N, k)`)

#### Exercises

**Q**

```python
x = t.ones((3, 1, 5))
y = t.ones((1, 4, 5))

z = x + y
```

**A**

```python
z.shape = (3, 4, 5)
```

✅

______________________________________________________________________

**Q**

```python
x = t.ones((8, 2, 6))
y = t.ones((8, 2))

z = x + y
```

**A**

```python
z.shape = (8, 2, 6)
```

❌

**A (Revised)**
Invalid. Dimension is added to the FRONT, not end. So `y` becomes `(1, 8, 2)`

```python
x = t.ones((8, 2, 6))
y = t.ones((2, 6))

z = x + y
```

**A**

```python
z.shape = (8, 2, 6)
```

✅

(Not continuing)
