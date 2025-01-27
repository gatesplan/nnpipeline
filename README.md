# Neural Network Pipeline

Simple network building pipeline for PyTorch. (torch>=2.1,<2.2)


## How to start?

1. install `pip install fish-nnpipeline` and `import nnpipeline`

2. create a pipe base object. (ex: `layer = nnpipeline.LinearExponentialEncoder(10, 5))

3. chain it to `torch.nn.sequential` or do whatever you want. all pipeline object follows `torch.nn.Module`..
   (at least `forward` method exists.)


## Explain of each pipeline object

1. Layers

1.1. `LinearExponentialEncoder` : Generate linear layers semi-automatically. You should give `in_features`, `out_features` and the class do the rest.

You can control narrowing node ratio by `compression_rate` parameter. (default 0.618) Also can use noramlization, dropout.

You cannot alter order between linear and others like norm, activation, dropout. It has been fixed (linear -> norm -> activation -> dropout) because I'm super lazy.

```aiignore
lee = LinearExponentialEncoder(100, 34)
```

```aiignore
# what lee is..
LinearExponentialEncoder(
  (layers): Sequential(
    (0): Linear(in_features=100, out_features=61, bias=True)
    (1): BatchNorm1d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=61, out_features=37, bias=True)
    (4): BatchNorm1d(37, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Linear(in_features=37, out_features=34, bias=True)
    (7): BatchNorm1d(34, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
  )
)
```

1.2. `LinearExponentialDecoder` : Encoder is narrowing down linear nodes. Decoder is expanding nodes. It's like a mirror of encoder.

Those linear encoder/decoder work based on exponentially inceasing/decreasing sequence that start from `in_features` to `out_features`. As you can see above example of `lee`, it works just like normal torch.nn.Module object. So you can put this in `nn.Sequential` or something.

1.3. `LinearCylinder` : Cylinder is a simple multi-layer module that has same input, output features. It looks just like a cylinder so the name is also cylinder.


2. Glues

2.1. `LinearJoint` LYou can concatenate multiple pipeline output to one single pipeline using `torch.cat`. yes you can do this by yourself, but I prefer this way more.


3. Compositions

3.1. `LinearExponentialComposition`: You can direcly use multiple pipes and join them manually, or you can simply define COMPOSITION class.

```aiignore
l1 = LinearExponentialEncoder(100, 35)
l2 = LinearExponentialEncoder(150, 40)
l3 = LinearExponentialEncoder(120, 30)

lec = LinearExponentialComposition([l1, l2, l3], 80)
print(lec)
```

```aiignore
# I'm lec!
LinearExponentialComposition(
  (pipes): ModuleList(
    (0): LinearExponentialEncoder(
      (layers): Sequential(
        (0): Linear(in_features=100, out_features=61, bias=True)
                ...
        (8): ReLU()
      )
    )
    (1): LinearExponentialEncoder(
      (layers): Sequential(
        (0): Linear(in_features=150, out_features=92, bias=True)
                ...
        (8): ReLU()
      )
    )
    (2): LinearExponentialEncoder(
      (layers): Sequential(
        (0): Linear(in_features=120, out_features=74, bias=True)
                ...
        (8): ReLU()
      )
    )
  )
  (joint): LinearJoint()
  (encoder): LinearExponentialEncoder(
    (layers): Sequential(
      (0): Linear(in_features=105, out_features=80, bias=True)
      (1): BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
)
```

4. No more contents. I'm lazy. I'll add more later.
