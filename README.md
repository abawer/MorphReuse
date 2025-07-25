# MorphReuse

**Shaped Weight Sharing with Per-Layer Scaling for Lighter Computational Networks**  

[DOI Placeholder – Zenodo Link Here](https://zenodo.org)


## Overview


This repository explores **MorphReuse**, an efficient parameter-sharing technique that replaces network layers with a shared trainable core and per-layer scaling adapters. Key features:

- **Shared Core MLP**: Central trainable transformation module
- **Per-Layer Scaling**: Learnable scaling factors for layer-specific adaptation
- **Residual Connections**: Identity + scaled transformation for stable training
- **CPU-Optimized**: Designed for accessible experimentation on standard hardware

Benchmarked across vision and text tasks:
- **Image Classification**: MNIST, FashionMNIST, CIFAR10
- **Text Classification**: SST-2 sentiment analysis


## Method by Dataset

Evaluate three training strategies—**Baseline**, **MorphReuse**, and **LoRA**—on three datasets: MNIST, FashionMNIST, CIFAR-10, and SST-2. Each method modifies which layers are trained or reused.


### Image Classification (MNIST, FashionMNIST, CIFAR10)

| Method       | Architecture                          | Trainable Components               |
|--------------|---------------------------------------|------------------------------------|
| **Baseline** | Full MLP/ConvNet                      | All parameters                     |
| **MorphReuse**| Shared core + scaling adapters       | Core + adapters only (10-26% params)|
| **LoRA**     | Low-Rank Adapters                     | Adapter matrices only (1-2% params)|

**Key Implementation**:
```python
class MorphReuseCore(nn.Module):
    def __init__(self, dim=128, expand=2.5):
        super().__init__()
        hidden = int(dim * expand)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        self.scale_param = nn.Parameter(torch.tensor([-3.0]))  # α-residual control

    def forward(self, x):
      transformed = self.net(x)
      scale = torch.sigmoid(self.scale_param) * 2  # 0-2 range
      return x + scale * transformed

class MorphReuseAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, shared_core):
        super().__init__()
        self.core = shared_core
        bottleneck = shared_core.net[0].in_features
        self.in_proj = nn.Linear(in_dim, bottleneck, bias=False)
        self.out_proj = nn.Linear(bottleneck, out_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        return self.out_proj(self.core(self.in_proj(x)))
```

**Baseline**  
- Two-layer MLP classifier. All weights are trained end-to-end.

**MorphReuse**  
- Both MLP layers are replaced with a shared MLP core and per-layer scaling adapters.  
- Outer wrappers are frozen; only the core and adapters are trained.

**LoRA**  
- Injects LoRA adapters into each `nn.Linear`.  
- Base weights are frozen; only adapters are trainable.


#### CIFAR-10 (RGB Images 3×32×32, Conv Backbone + Classifier MLP)

Same MLP as above on top of a Convolutional Backbone net for feature extraction.

**Baseline**  
- Convolutional feature extractor followed by a fully connected MLP classifier.  
- All layers are trained.

**MorphReuse**  
- The convolutional extractor backcbone is trained.
- The classifier is replaced by two shared MorphReuse layers with scaling adapters.  
- Only the shared core and extractor backbone is trained, adaptor weights are frozen.

**LoRA**  
- The convolutional extractor backcbone is trained.
- LoRA adapters are added to the classifier MLP.  
- Only LoRA layers are trained out of the MLP.



### SST-2 (Text Sentiment Classification)

| Method       | Architecture                          | Trainable Components               |
|--------------|---------------------------------------|------------------------------------|
| **Baseline** | Pretrained TinyBERT LLM                  | Three top layers                   |
| **MorphReuse**| Pretrained TinyBERT + Surrogate Shared Weights for three top layers + Non-Linear activation       | Surrogate Weights|
| **LoRA**     | Pretrained TinyBERT+Low-Rank Adapters for three top layer  | LoRA residual weights|


**TinyBERT** is a distilled version of BERT developed by Huawei Noah’s Ark Lab. It significantly reduces model size and inference time while retaining much of the original BERT performance.
In this project, TinyBERT is used as a frozen LLM backbone for SST-2 sentiment classification, with MorphReuse or LoRA applied on the top layers.

- **Model Name**: `huawei-noah/TinyBERT_General_4L_312D`
- **Architecture**: 4 transformer layers, 312 hidden size, 12 attention heads
- **Pretrained on**: General-domain data
- **Use Case**: Efficient fine-tuning and inference for downstream NLP tasks like SST-2
- **Parameter Count**: ~14.5M
- **Tokenizer**: BERT-compatible (`bert-base-uncased`)



**Key Implementation**:
```python

# Simplified MorphReuse wrapper for shared weights
class MorphReuseLinearWrapperShared(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.core = nn.GELU()
        self.shared_w = SharedWeight(fin, fout) # Use SharedWeight

    def forward(self, x):
        return self.core(self.shared_w(x))

class SharedWeight(nn.Module):
    """
    Singleton storage for one (weight, bias) pair per unique shape.
    """
    _cache = {}          # key=(in_f, out_f)  ->  nn.ParameterDict
    _usage = {}

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        key = (in_f, out_f)
        if key not in self._cache:               # create once
            w = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.xavier_uniform_(w)
            b = nn.Parameter(torch.zeros(out_f))
            self._cache[key] = nn.ParameterDict({"sweight": w, "sbias": b})
        self.params = self._cache[key]           # shared reference
        self._usage[key] = self._usage.get(key, 0) + 1
        self.scale = nn.Parameter(torch.ones(1))   # per-layer

    def forward(self, x):
        return self.scale * Fn.linear(x, self.params["sweight"], self.params["sbias"])
```

**Baseline**  
- Uses `huawei-noah/TinyBERT_General_4L_312D`.  
- Unfreezes top 2 transformer encoder layers + classifier head.  
- All unfrozen parts are trainable.

**MorphReuse**  
- All encoder layers are frozen.  
- The top 2 layers and classifier are wrapped with `MorphReuseLinearWrapperShared`.  
- Shared core is used across `q`, `k`, `v`, intermediate, output, and classifier projections.  
- Only the core and adapters are trained.

**LoRA**  
- Applies LoRA to top 2 layers and classifier.  
- Base TinyBERT is frozen.  
- Only LoRA modules are trained.



## Comparative Results


### Colab Notebook Specs
```
OS: Linux-6.1.123+-x86_64-with-glibc2.35
CPU: AMD EPYC 7B12 (2×2.25 GHz)
RAM: 12977 MB
Python: 3.11.13
PyTorch: 2.3.1+cpu
Device: cpu
```

### Prerequisits
```
pip install -U --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu \
   -f https://download.pytorch.org/whl/torch_stable.html \
   "datasets<2.19" transformers "peft<0.7.0" huggingface_hub fsspec accelerate
```

### Tested Datasets


| Dataset      | Type                    | Classes                      | Shape      | Usage                                 |
|--------------|-------------------------|------------------------------|------------|----------------------------------------|
| **MNIST**        | Handwritten digits      | 10 (digits 0–9)              | 1×28×28    | Image classification (MLP/CNN)        |
| **FashionMNIST** | Clothing images         | 10 (shirts, shoes, etc.)     | 1×28×28    | Realistic alt. to MNIST               |
| **CIFAR-10**     | Natural object photos   | 10 (airplane, dog, etc.)     | 3×32×32    | Image classification (CNN)            |
| **SST-2**        | Movie review sentences  | 2 (positive / negative)      | 64 Max Sequence Len (padded)   | Text classification (LLM fine-tuning) |



### 1. MNIST (MLP for flatten 1×28×28 image)

<img width="1990" height="1572" alt="image" src="https://github.com/user-attachments/assets/ddee55e1-fa87-40dc-bbb7-d6d362a27cd3" />



### 2. FashionMNIST (MLP Classifier for flatten 1×28×28 image)


<img width="1990" height="1572" alt="image" src="https://github.com/user-attachments/assets/281556e2-4b8d-4afe-8535-3e59deda2d7b" />



### 3. CIFAR-10 (ConvNet + MLP Classifier)

<img width="1990" height="1572" alt="image" src="https://github.com/user-attachments/assets/d2bede36-598a-49c9-b481-c5b409d59dce" />



### 4. SST-2 (Pretrained Headless TinyBERT + Trained 3 top layers)

<img width="1990" height="1572" alt="image" src="https://github.com/user-attachments/assets/a7883e3a-75c4-4312-a80f-f19d59de4681" />


## References

- [TinyBERT on Hugging Face](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
- [Paper: TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
- Parameter-efficient fine-tuning: [LoRA paper](https://arxiv.org/abs/2106.09685)
- Weight sharing in transformers: e.g., ALBERT [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)  

