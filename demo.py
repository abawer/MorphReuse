# Copyright 2025 Amit Bawer
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0 


# ========================================================
# Colab MorphReuse demo (Baseline vs MorphReuse vs LoRA)
#
# FOR CPU-ONLY EXECUTION
#
# Run first:
#
# pip install -U --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu \
#   -f https://download.pytorch.org/whl/torch_stable.html \
#   "datasets<2.19" transformers "peft<0.7.0" huggingface_hub fsspec accelerate
# ========================================================
import gc
import os
import platform
import time
import numpy as np
import matplotlib.pyplot as plt
import re
import multiprocessing
from collections import Counter
# Set up multi-threading
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = str(multiprocessing.cpu_count())
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fn
import torchvision.datasets as tvds
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

# ========== DEVICE CONFIGURATION (FORCED CPU) ==========
DEVICE = torch.device("cpu") # Explicitly use CPU
print(f"Using device: {DEVICE}")
# =======================================================

# ========== HYPERPARAMETERS ============================
# Vision / Global Consts
WIDTH = 512
MORPH_REUSE_DIM = 128
MORPH_REUSE_EXPAND = 2
EPOCHS = 5
BATCH_SIZE = 128
LR_BASELINE = 1e-3
LR_MORPH_REUSE = 1e-3
# LLM Consts
PRETRAINED_NAME = "huawei-noah/TinyBERT_General_4L_312D"
LLM_SEQ_LEN = 64
MAX_SAMPLES = 1000
BATCH_SIZE_LM = 32
# =======================================================

print("========= Configuration ==========")
for name, value in globals().copy().items():
    # skip private names and built-ins
    if not name.startswith('_') and name.isupper():
        print(f"{name}: {value}")
print("=================================")

def print_sys_info():
    cpu_freq = cpu_model = ram = "Unknown"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                raw = f.read()
            cpu_model_match = re.search(r"model name\s*:\s*(.+?)\n", raw)
            if cpu_model_match:
                cpu_model = cpu_model_match.group(1).strip()
            mhz = re.findall(r"cpu MHz\s+:\s+(\d+\.\d+)", raw)
            if mhz:
                cpu_freq = f"{float(mhz[-1])/1000:.2f} GHz"
            with open("/proc/meminfo") as f:
                mem_raw = f.read()
            mem_match = re.search(r"MemTotal:\s+(\d+)", mem_raw)
            if mem_match:
                ram = f"{int(mem_match.group(1)) // 1024} MB"
        except Exception as e:
            print(f"System info error: {e}")
    print("\n========= System Information =========")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpu_model} ({os.cpu_count()}×{cpu_freq})")
    print(f"RAM: {ram}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print("=====================================\n")

print_sys_info()

DATASETS = {
    'MNIST': {
        'type': 'classification',
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    },
    'FashionMNIST': {
        'type': 'classification',
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    },
    'CIFAR10': {
        'type': 'classification',
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    },
    'sst2': {
        'type': 'llm-classification',
        'num_classes': 2,
        'input_shape': (LLM_SEQ_LEN,),
        'transform': None
    },
}

# ================== MODEL DEFINITIONS ==================
class MorphReuseCore(nn.Module):
    def __init__(self, in_dim=MORPH_REUSE_DIM, expand=MORPH_REUSE_EXPAND, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden = int(in_dim * expand)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
        # Learnable scaling factor for adapter output
        self._gain_logit = nn.Parameter(torch.tensor(0.0))
        # Initialize with smaller weights
        for layer in [0, 2]:
            nn.init.xavier_uniform_(self.net[layer].weight)
            if self.net[layer].bias is not None:
                nn.init.zeros_(self.net[layer].bias)
    @property
    def gain(self):
        return torch.sigmoid(self._gain_logit)
    def forward(self, x):
        return ((1.0-self.gain) * x) + (self.gain * self.net(x))
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

class MorphReuseAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, shared_core):
        super().__init__()
        self.core = shared_core
        bottleneck_in = shared_core.net[0].in_features
        bottleneck_out = shared_core.net[2].out_features
        self.in_proj = nn.Linear(in_dim, bottleneck_in, bias=False)
        self.out_proj = nn.Linear(bottleneck_out, out_dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    def forward(self, x):
        return self.out_proj(self.core(self.in_proj(x)))

# Simplified MorphReuse wrapper for shared weights
class MorphReuseLinearWrapperShared(nn.Module):
    """Shared bottleneck adapter with residual connection using SharedWeight"""
    def __init__(self, fin, fout):
        super().__init__()
        self.core = nn.Identity() # Placeholder for potential core logic if needed differently
        self.w = SharedWeight(fin, fout) # Use SharedWeight

    def forward(self, x):
        return self.core(self.w(x)) # Apply shared weight, then identity core

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
        #self.scale = 1

    def forward(self, x):
        return self.scale * Fn.linear(x, self.params["sweight"], self.params["sbias"])

    @classmethod
    def stats(cls):
        print(f"Shared Weights:\n {cls._usage}")
    
    @classmethod
    def reset(cls):
        cls._cache = {}
        cls._usage = {}

# ================== MEMORY STATS ==================
def get_memory_stats(model):
    """Calculate memory usage for parameters, gradients, and optimizer states"""
    stats = {
        'param_mem': 0.0,
        'grad_mem': 0.0,
        'buffer_mem': 0.0,
        'optimizer_state_mem': 0.0,
        'total_estimated_mem': 0.0
    }
    # Parameters and gradients
    for p in model.parameters():
        numel = p.numel()
        elem_size = p.element_size() # Size in bytes
        stats['param_mem'] += numel * elem_size
        if p.requires_grad:
            # Gradient memory
            stats['grad_mem'] += numel * elem_size
            # Optimizer states (Adam: 2 states per parameter)
            stats['optimizer_state_mem'] += 2 * numel * elem_size
    # Buffers
    for b in model.buffers():
        stats['buffer_mem'] += b.numel() * b.element_size()
    # Convert to MB
    for key in stats:
        if key != 'total_estimated_mem':
            stats[key] /= (1024 ** 2)
    stats['total_estimated_mem'] = (stats['param_mem'] + stats['grad_mem'] +
                                   stats['buffer_mem'] + stats['optimizer_state_mem'])
    return stats

def print_memory_stats(name, stats):
    """Print formatted memory statistics"""
    print(f"\n{name} Memory (MB)")
    print(f"  Params:   {stats['param_mem']:.2f}")
    print(f"  Grad:     {stats['grad_mem']:.2f}")
    print(f"  Buffers:  {stats['buffer_mem']:.2f}")
    print(f"  Optim:    {stats['optimizer_state_mem']:.2f}")
    print(f"  Total:    {stats['total_estimated_mem']:.2f}")

# ================== DATA LOADING ==================
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class TextDataset(Dataset):
    def __init__(self, split, seq_len=LLM_SEQ_LEN, max_samples=MAX_SAMPLES, validate_tokenization=False): # Disable validation for speed
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split=split)
        # print(ds[0]) # Debug print removed for cleaner output
        # Fixed shuffling and sampling
        texts = []
        labels = []
        indices = list(range(len(ds)))
        np.random.shuffle(indices)
        for i in indices:
            text = ds[i]["sentence"]
            if text.strip():
                texts.append(text)
                labels.append(ds[i]["label"])
                if len(texts) == max_samples:
                    break
        # Tokenize
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )
        # Validate tokenization (disabled for CPU speed)
        if validate_tokenization:
            print(f"\n=== Tokenization Check (first 5 samples) ===")
            for i in range(min(5, len(texts))):
                print(f"\nOriginal  : {texts[i]}")
                print(f"Token IDs : {enc['input_ids'][i].tolist()}")
                decoded = tokenizer.decode(enc["input_ids"][i], skip_special_tokens=True)
                print(f"Decoded   : {decoded}")
                print(f"Attention : {enc['attention_mask'][i].tolist()}")
                print(f"Length    : {len(tokenizer.tokenize(texts[i]))} tokens")
            max_len = max(len(tokenizer.tokenize(t)) for t in texts)
            print(f"\nMax tokenized input length: {max_len} (limit is {seq_len})")
        print("Labels counts:", Counter(labels)) # Debug print removed
        self.input_ids = enc["input_ids"]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.raw_texts = texts
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]

def get_text_loaders():
    train_ds = TextDataset("train")
    test_ds = TextDataset("validation")
    # Remove overlapping samples
    train_texts = set(train_ds.raw_texts)
    test_texts = set(test_ds.raw_texts)
    overlap = train_texts & test_texts
    if overlap:
        # Filter out overlapping samples from test set
        keep_indices = [i for i, text in enumerate(test_ds.raw_texts) if text not in overlap]
        test_ds.input_ids = test_ds.input_ids[keep_indices]
        test_ds.labels = test_ds.labels[keep_indices]
        test_ds.raw_texts = [text for i, text in enumerate(test_ds.raw_texts) if i in keep_indices]
        print(f"Removed {len(overlap)} leaked samples from test set")
    # Verify no overlap remains
    assert len(set(train_ds.raw_texts) & set(test_ds.raw_texts)) == 0, "Data leakage detected!"
    # Create data loaders (Reduced workers for CPU)
    pin_mem = False # Pin memory usually for GPU
    workers = 0 # Simplify for CPU
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_LM, shuffle=True,
        num_workers=workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE_LM, shuffle=False,
        num_workers=workers, pin_memory=pin_mem
    )
    return train_loader, test_loader, train_ds, test_ds

def load_dataset(name):
    ds_info = DATASETS[name]
    task_type = ds_info['type']
    if task_type == 'classification':
        # Image dataset loading
        transform = ds_info['transform']
        loader_class = getattr(tvds, name)
        train_set = loader_class(root='./data', train=True, download=True, transform=transform)
        test_set = loader_class(root='./data', train=False, download=True, transform=transform)
        print(f"\nDataset: {name}")
        print(f"Type: Classification")
        print(f"Classes: {ds_info['num_classes']}")
        print(f"Input shape: {ds_info['input_shape']}")
        print(f"Train samples: {len(train_set)}")
        print(f"Test samples: {len(test_set)}")
        print(f"Batch size: {BATCH_SIZE}")
        # Create data loaders (Reduced workers for CPU)
        pin_mem = False
        workers = 0 # Simplify for CPU
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=workers, pin_memory=pin_mem
        )
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=workers, pin_memory=pin_mem
        )
        return train_loader, test_loader, ds_info['input_shape'], ds_info['num_classes']
    elif task_type == 'llm-classification':
        # Text dataset loading
        train_loader, test_loader, train_ds, test_ds = get_text_loaders()
        print(f"\nDataset: {name}")
        print(f"Type: Text Classification")
        print(f"Classes: {ds_info['num_classes']}")
        print(f"Sequence length: {LLM_SEQ_LEN}")
        print(f"Train samples: {len(train_ds)}")
        print(f"Test samples: {len(test_ds)}")
        print(f"Batch size: {BATCH_SIZE_LM}")
        return train_loader, test_loader, (LLM_SEQ_LEN,), ds_info['num_classes']
    else:
        raise ValueError(f"Unsupported dataset type: {task_type}")

# ================== MODEL BUILDERS ==================
def build_vision_head(input_dim, output_dim, shared_core=None):
    if shared_core:
        return nn.Sequential(
            MorphReuseAdapter(input_dim, WIDTH, shared_core),
            MorphReuseAdapter(WIDTH, output_dim, shared_core)
        )
    else:
        return nn.Sequential(
            nn.Linear(input_dim, WIDTH), nn.ReLU(),
            nn.Linear(WIDTH, WIDTH), nn.ReLU(),
            nn.Linear(WIDTH, output_dim)
        )

def build_vision_backbone(channels):
    return nn.Sequential(
        nn.Conv2d(channels, 32, 3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU()
    )

def build_vision_baseline(in_shape, n_classes):
    channels, *rest = in_shape
    if len(rest) == 2 and rest == (32, 32):  # CIFAR-like
        backbone = build_vision_backbone(channels)
        flat_dim = 128 * 8 * 8
        return nn.Sequential(
            backbone, nn.Flatten(),
            build_vision_head(flat_dim, n_classes)
        ).to(DEVICE)
    else:  # MNIST-like
        input_dim = np.prod(in_shape)
        return nn.Sequential(
            nn.Flatten(),
            build_vision_head(input_dim, n_classes)
        ).to(DEVICE)

def build_vision_morph_reuse(in_shape, n_classes):
    shared_core = MorphReuseCore()
    channels, *rest = in_shape
    backbone = None
    if len(rest) == 2 and rest == (32, 32):  # CIFAR-like
        backbone = build_vision_backbone(channels)
        flat_dim = 128 * 8 * 8
        model = nn.Sequential(
            backbone, nn.Flatten(),
            build_vision_head(flat_dim, n_classes, shared_core)
        ).to(DEVICE)
    else:  # MNIST-like
        input_dim = np.prod(in_shape)
        model = nn.Sequential(
            nn.Flatten(),
            build_vision_head(input_dim, n_classes, shared_core)
        ).to(DEVICE)
    # Freeze all except core and backbone
    for param in model.parameters():
        param.requires_grad = False
    if backbone:
      for param in backbone.parameters():
          param.requires_grad = True
    shared_core.unfreeze()
    return model

def build_vision_lora(in_shape, n_classes, rank=4):
    channels, *rest = in_shape
    backbone = None
    if len(rest) == 2 and rest == (32, 32):
        backbone = build_vision_backbone(channels)
        flat_feat_dim = 128 * 8 * 8
        model = nn.Sequential(
            backbone, nn.Flatten(),
            build_vision_head(flat_feat_dim, n_classes)).to(DEVICE)
    else:
        input_dim = np.prod(in_shape)
        model = nn.Sequential(
            nn.Flatten(),
            build_vision_head(input_dim, n_classes)).to(DEVICE)

    # Create a wrapper for vision models to make them compatible with PEFT
    class VisionWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids=None, **kwargs):
            # For vision models, we only need the input tensor
            # 'input_ids' is used for text models, but we'll use the same name for consistency
            # Handle both input_ids (from PEFT) and direct tensor input (from vision)
            if input_ids is not None:
                 return self.model(input_ids)
            elif 'pixel_values' in kwargs: # Common name for vision inputs
                 return self.model(kwargs['pixel_values'])
            else:
                 # Assume first kwarg is the tensor if named differently
                 return self.model(next(iter(kwargs.values())))

    # Wrap the vision model
    wrapped_model = VisionWrapper(model)

    # Identify Linear layers for LoRA
    target_names = [n for n, m in wrapped_model.named_modules() if isinstance(m, nn.Linear)]
    # print("Target modules for LoRA:", target_names) # Debug print

    lora_config = LoraConfig(
        task_type='FEATURE_EXTRACTION', # Use FEATURE_EXTRACTION for vision tasks with PEFT
        r=rank,
        lora_alpha=rank,
        target_modules=target_names,
        lora_dropout=0.0,
        bias="none"
    )
    lora_model = get_peft_model(wrapped_model, lora_config)

    # Ensure backbone is trainable if it exists
    if backbone:
        for p in backbone.parameters():
            p.requires_grad = True
    return lora_model

def build_llm_baseline():
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_NAME,
        num_labels=2,
        torch_dtype="auto", # torch.float32 for CPU
        use_safetensors=True,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    # Freeze everything but last few layers and classifier
    model.requires_grad_(False)
    train_layers = ["classifier", "bert.encoder.layer.3", "bert.encoder.layer.2"] # Last 2 BERT layers
    for name, p in model.named_parameters():
        if any(layer in name for layer in train_layers):
            p.requires_grad_(True)
    return model.to(DEVICE)

def build_llm_morph_reuse():
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_NAME,
        num_labels=2,
        torch_dtype="auto", # torch.float32 for CPU
        use_safetensors=True,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))

    # 2. Wrap the chosen layers with TMM adapters (unchanged)
    SharedWeight.reset()
    def wrap_layer(layer):
        in_feat = layer.in_features
        out_feat = layer.out_features
        return MorphReuseLinearWrapperShared(in_feat, out_feat)

    # ---------- BERT path ----------
    for layer in model.bert.encoder.layer[-2:]:       # last two layers
        layer.attention.self.query      = wrap_layer(layer.attention.self.query)
        layer.attention.self.key        = wrap_layer(layer.attention.self.key)
        layer.attention.self.value      = wrap_layer(layer.attention.self.value)
        layer.attention.output.dense    = wrap_layer(layer.attention.output.dense)
        layer.intermediate.dense        = wrap_layer(layer.intermediate.dense)
        layer.output.dense              = wrap_layer(layer.output.dense)

    # BERT classifier stack (no pre_classifier)
    model.classifier = wrap_layer(model.classifier)
    SharedWeight.stats()

    # Freeze everything but last few layers and classifier
    model.requires_grad_(False)
    train_layers = ["classifier", "bert.encoder.layer.3", "bert.encoder.layer.2"] # Last 2 BERT layers
    for name, p in model.named_parameters():
        if any(layer in name for layer in train_layers):
            p.requires_grad_(True)
    return model.to(DEVICE)

def build_llm_lora():
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_NAME,
        num_labels=2,
        torch_dtype="auto", # torch.float32 for CPU
        use_safetensors=True,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA configuration for BERT layers
    target_modules = [
        "query", "key", "value", "output.dense",
        "intermediate.dense", "output.dense"
    ]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4, # Reduced rank for CPU
        lora_alpha=8, # Adjusted alpha
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none"
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model.to(DEVICE)

# ================== TRAINING & EVALUATION ==================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train(model, train_loader, test_loader, lr, tag):
    # Filter parameters that require gradients
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not trainable_params:
        raise Exception(f"[{tag}] Warning: No trainable parameters found!")

    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    train_acc = []
    test_acc = []
    fwd_times = []
    bwd_times = []
    upd_times = []

    # Print initial memory stats
    mem_stats = get_memory_stats(model)
    print_memory_stats(f"{tag} (Initial)", mem_stats)

    # Initial evaluation
    model.eval()
    def evaluate(loader):
        correct = total = 0
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(loader, desc="accuracy evaluation", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                if x.dtype == torch.long:               # Text (token IDs)
                    mask = (x != tokenizer.pad_token_id).long()
                    outputs = model(input_ids=x, attention_mask=mask, labels=y)
                else:                                   # Vision (Images)
                    outputs = model(x)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total

    print(f"[{tag}] Evaluating initial accuracy...")
    train_acc.append(evaluate(train_loader))
    test_acc.append(evaluate(test_loader))
    print(f"  Initial Train: {train_acc[-1]:.2f}%, Test: {test_acc[-1]:.2f}%")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_fwd = epoch_bwd = epoch_upd = 0.0
        pbar = tqdm(train_loader, desc=f"{tag} Epoch {epoch+1}/{EPOCHS}", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            start_time = time.time()
            if x.dtype == torch.long:                       # Text
                mask = (x != tokenizer.pad_token_id).long()
                outputs = model(input_ids=x, attention_mask=mask, labels=y)
                loss = outputs.loss
            else:                                           # Vision
                outputs = model(x)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, y)
            fwd_time = time.time() - start_time
            epoch_fwd += fwd_time

            # Backward pass
            start_time = time.time()
            loss.backward()
            bwd_time = time.time() - start_time
            epoch_bwd += bwd_time

            # Update
            start_time = time.time()
            optimizer.step()
            upd_time = time.time() - start_time
            epoch_upd += upd_time

            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Store timing
        fwd_times.append(epoch_fwd)
        bwd_times.append(epoch_bwd)
        upd_times.append(epoch_upd)

        # Evaluate
        train_acc.append(evaluate(train_loader))
        test_acc.append(evaluate(test_loader))
        print(f"[{tag}] Epoch {epoch+1}/{EPOCHS}: "
              f"Train: {train_acc[-1]:.2f}%, Test: {test_acc[-1]:.2f}% | "
              f"Time: Fwd {epoch_fwd:.1f}s, Bwd {epoch_bwd:.1f}s, Upd {epoch_upd:.1f}s Total {(epoch_fwd+epoch_bwd+epoch_upd):.1f}s")

    # Print final memory stats
    mem_stats = get_memory_stats(model)
    print_memory_stats(f"{tag} (Final)", mem_stats)

    return train_acc, test_acc, fwd_times, bwd_times, upd_times

# ================== EXPERIMENT RUNNER ==================
def run_experiment(name):
    print(f"\n{'='*50}")
    print(f"Running experiment: {name}")
    print(f"{'='*50}")
    # Load dataset
    train_loader, test_loader, in_shape, n_classes = load_dataset(name)

    # Build models
    if name == 'sst2':
        print("Building LLM Baseline...")
        baseline = build_llm_baseline()
        print("Building LLM MorphReuse...")
        morph_reuse_model = build_llm_morph_reuse()
        print("Building LLM LoRA...")
        lora_model = build_llm_lora()
    else: # Vision datasets
        print("Building Vision Baseline...")
        baseline = build_vision_baseline(in_shape, n_classes)
        print("Building Vision MorphReuse...")
        morph_reuse_model = build_vision_morph_reuse(in_shape, n_classes)
        print("Building Vision LoRA...")
        lora_model = build_vision_lora(in_shape, n_classes)

    # Print model stats
    def print_model_stats(model, name):
        total, trainable = count_params(model)
        print(f"{name}:")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Percentage trainable: {100*trainable/total:.2f}%")

    print("\nModel Parameters:")
    print_model_stats(baseline, "Baseline")
    print_model_stats(morph_reuse_model, "MorphReuse")
    print_model_stats(lora_model, "LoRA")

    # Train models
    print("\nTraining Baseline:")
    b_train_acc, b_test_acc, b_fwd, b_bwd, b_upd = train(
        baseline, train_loader, test_loader, LR_BASELINE, f"{name}-Baseline")

    print("\nTraining MorphReuse:")
    mr_train_acc, mr_test_acc, mr_fwd, mr_bwd, mr_upd = train(
        morph_reuse_model, train_loader, test_loader, LR_MORPH_REUSE, f"{name}-MorphReuse")

    print("\nTraining LoRA:")
    l_train_acc, l_test_acc, l_fwd, l_bwd, l_upd = train(
        lora_model, train_loader, test_loader, LR_MORPH_REUSE, f"{name}-LoRA") # Using MorphReuse LR for LoRA too

    # Get final memory stats
    mem_base = get_memory_stats(baseline)
    mem_mr = get_memory_stats(morph_reuse_model)
    mem_lora = get_memory_stats(lora_model)

    # Plotting
    epochs = list(range(EPOCHS + 1))
    plt.figure(figsize=(20, 16))
    plt.suptitle(f"{name}: Comparative Results", fontsize=16, y=0.98)

    # 1. Main metric (Accuracy)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, b_train_acc, 'o--', label='Baseline (train)', color='#1f77b4', linewidth=2.5)
    plt.plot(epochs, b_test_acc, 'o-',  label='Baseline (test)',  color='#aec7e8', linewidth=1.8)
    plt.plot(epochs, mr_train_acc,  's--', label='MorphReuse (train)',      color='#ff7f0e', linewidth=2.5)
    plt.plot(epochs, mr_test_acc,  's-',  label='MorphReuse (test)',       color='#ffbb78', linewidth=1.8)
    plt.plot(epochs, l_train_acc, 'd--', label='LoRA (train)',     color='#9467bd', linewidth=2.5)
    plt.plot(epochs, l_test_acc, 'd-',  label='LoRA (test)',      color='#c5b0d5', linewidth=1.8)
    plt.title("Accuracy Progression")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. Timing breakdown (Last Epoch)
    plt.subplot(2, 2, 2)
    models = ['Baseline', 'MorphReuse', 'LoRA']
    fwd_times = [b_fwd[-1], mr_fwd[-1], l_fwd[-1]] if b_fwd and mr_fwd and l_fwd else [0,0,0]
    bwd_times = [b_bwd[-1], mr_bwd[-1], l_bwd[-1]] if b_bwd and mr_bwd and l_bwd else [0,0,0]
    upd_times = [b_upd[-1], mr_upd[-1], l_upd[-1]] if b_upd and mr_upd and l_upd else [0,0,0]
    total_times = [sum(x) for x in zip(fwd_times, bwd_times, upd_times)]
    x = np.arange(len(models))
    w = 0.15
    plt.bar(x - 1.5*w, fwd_times, w, label='Forward', color='#56B4E9', edgecolor='k')
    plt.bar(x - 0.5*w, bwd_times, w, label='Backward',color='#E69F00', edgecolor='k')
    plt.bar(x + 0.5*w, upd_times, w, label='Update',  color='#009E73', edgecolor='k')
    plt.bar(x + 1.5*w, total_times, w, label='Total', color='#D55E00', edgecolor='k')
    plt.title("Epoch Time Breakdown (Last Epoch)")
    plt.xlabel("Models")
    plt.ylabel("Time (s)")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 3. Memory comparison
    plt.subplot(2, 2, 3)
    categories = ['Params', 'Grads', 'Buffers', 'Opt State', 'Total']
    baseline_mem = [mem_base[k] for k in ['param_mem', 'grad_mem', 'buffer_mem',
                                          'optimizer_state_mem', 'total_estimated_mem']]
    mr_mem = [mem_mr[k] for k in ['param_mem', 'grad_mem', 'buffer_mem',
                                    'optimizer_state_mem', 'total_estimated_mem']]
    lora_mem = [mem_lora[k] for k in ['param_mem', 'grad_mem', 'buffer_mem',
                                      'optimizer_state_mem', 'total_estimated_mem']]
    bar_w = 0.2
    r1 = np.arange(len(categories))
    r2 = [x + bar_w for x in r1]
    r3 = [x + 2*bar_w for x in r1]
    plt.bar(r1, baseline_mem, bar_w, label='Baseline', color='#0072B2', alpha=0.8)
    plt.bar(r2, mr_mem, bar_w, label='MorphReuse', color='#E69F00', alpha=0.8)
    plt.bar(r3, lora_mem, bar_w, label='LoRA', color='#9467bd', alpha=0.8)
    plt.title("Memory Usage (MB)")
    plt.xlabel("Memory Type")
    plt.ylabel("MB")
    plt.xticks([r + bar_w for r in r1], categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 4. Parameters comparison
    plt.subplot(2, 2, 4)
    models = ['Baseline', 'MorphReuse', 'LoRA']
    _, b_train = count_params(baseline)
    _, mr_train = count_params(morph_reuse_model)
    _, l_train = count_params(lora_model)
    _, b_total = count_params(baseline)
    _, mr_total = count_params(morph_reuse_model)
    _, l_total = count_params(lora_model)

    trainable = [b_train/1e6, mr_train/1e6, l_train/1e6]
    total = [b_total/1e6, mr_total/1e6, l_total/1e6]
    bar_w = 0.25
    x = np.arange(len(models))
    plt.bar(x - bar_w/2, trainable, bar_w, label='Trainable (M)', color='#56B4E9', edgecolor='k')
    plt.bar(x + bar_w/2, total, bar_w, label='Total (M)', color='#E69F00', edgecolor='k')
    for i, v in enumerate(trainable):
        if v is not None:
            plt.text(i - bar_w/2, v + 0.05, f"{v:.2f}", ha='center', fontsize=9)
    for i, v in enumerate(total):
        if v is not None:
            plt.text(i + bar_w/2, v + 0.05, f"{v:.2f}", ha='center', fontsize=9)
    plt.title("Parameter Comparison")
    plt.xlabel("Models")
    plt.ylabel("Parameters (Millions)")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{name}_results.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Cleanup
    del baseline, morph_reuse_model, lora_model
    gc.collect()

# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    # Run experiments on all datasets
    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'sst2'] # Enable all datasets
    for dataset in datasets:
        try:
            run_experiment(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
