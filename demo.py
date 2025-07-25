# ========================================================
# CPU Demo: Baseline vs MorphReuse vs LoRA vs BitFit
# Supports both Vision & Text
# ========================================================
import gc, os, platform, time, re, multiprocessing
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as tvds, torchvision.transforms as tvtf
from collections import Counter, defaultdict
from tqdm import tqdm
DEVICE = torch.device("cpu")

# ---------- DEMO STARTUP ----------
import platform, psutil, pkg_resources
print("\n" + "="*80)
print("  MORPHREUSE CPU DEMO")
print("="*80)
print(f"OS        : {platform.system()} {platform.release()} {platform.machine()}")
print(f"CPU       : {platform.processor()}")
print(f"CPU cores : {multiprocessing.cpu_count()} logical")
print(f"RAM       : {psutil.virtual_memory().total // (1024**3)} GB total, "
      f"{psutil.virtual_memory().available // (1024**3)} GB available")
print(f"Python    : {platform.python_version()} ({platform.python_implementation()})")
print(f"PyTorch   : {torch.__version__}")
print(f"Transformers : {pkg_resources.get_distribution('transformers').version}")
print(f"Datasets     : {pkg_resources.get_distribution('datasets').version}")
print(f"PEFT         : {pkg_resources.get_distribution('peft').version}")
print("="*80 + "\n")

# ---------- GLOBAL HYPERS ----------
EPOCHS = 3
BATCH_SIZE = 128          # image
BATCH_SIZE_LM = 32        # text
LR = 1e-3
PRETRAINED_NAME = "huawei-noah/TinyBERT_General_4L_312D"
LLM_SEQ_LEN = 64
MAX_SAMPLES = 1000

# ---------- TOKENIZER ----------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ---------- PEFT ----------
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

# ---------- MORPHREUSE ----------
class SharedWeight(nn.Module):
    _cache, _usage = {}, {}
    def __init__(self, in_f, out_f):
        super().__init__()
        key = (in_f, out_f)
        if key not in self._cache:
            w = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.xavier_uniform_(w)
            b = nn.Parameter(torch.zeros(out_f))
            self._cache[key] = nn.ParameterDict({"shared_w": w, "shared_b": b})
        self.params = self._cache[key]
        self._usage[key] = self._usage.get(key, 0) + 1
        self.scale  = nn.Parameter(torch.ones(1))
        self._usage[key] = self._usage.get(key, 0) + 1
    def forward(self, x):
        return self.scale * F.linear(x, self.params["shared_w"], self.params["shared_b"])
    @classmethod
    def stats(cls):
        print(f"Shared Weights:\n {cls._usage}")

class MorphReuseLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.shared = SharedWeight(in_f, out_f)
        self.scale = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        return self.scale * self.shared(x)

# ---------- MEMORY ----------
def mem_stats(model):
    stats = defaultdict(float)
    for p in model.parameters():
        sz = p.numel() * p.element_size()
        stats['param'] += sz
        if p.requires_grad:
            stats['grad'] += sz
            stats['optim'] += 2 * sz  # Adam
    for b in model.buffers():
        stats['buffer'] += b.numel() * b.element_size()
    for k in stats:
        stats[k] /= 1024 ** 2
    stats['total'] = sum(stats.values())
    return stats

def print_mem(tag, stats):
    print(f"\n{tag} (MB)")
    for k in ['param','grad','buffer','optim','total']:
        print(f"  {k.capitalize()}: {stats[k]:.2f}")

# ---------- DATA ----------
from datasets import load_dataset
def get_text_loaders():
    def ds(split):
        data = load_dataset("glue","sst2",split=split)[:MAX_SAMPLES]
        texts, labels = data['sentence'], data['label']
        enc = tokenizer(texts, padding="max_length", max_length=LLM_SEQ_LEN,
                        truncation=True, return_tensors="pt")
        ids, mask, y = enc["input_ids"], enc["attention_mask"], torch.tensor(labels)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(ids, mask, y),
                                           BATCH_SIZE_LM, shuffle=(split=="train"))
    train = ds("train")
    val   = ds("validation")
    return train, val

train_text, val_text = get_text_loaders()

# ---------- VISION ----------
from torchvision import transforms
TRANSFORMS = {
    'MNIST':        tvtf.Compose([tvtf.ToTensor(), tvtf.Normalize((0.5,),(0.5,))]),
    'FashionMNIST': tvtf.Compose([tvtf.ToTensor(), tvtf.Normalize((0.5,),(0.5,))]),
    'CIFAR10':      tvtf.Compose([tvtf.ToTensor(), tvtf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
}
def get_vision_loaders(name):
    train_ds = tvds.__dict__[name](root='./data', train=True,  download=True, transform=TRANSFORMS[name])
    test_ds  = tvds.__dict__[name](root='./data', train=False, download=True, transform=TRANSFORMS[name])
    train = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    test  = torch.utils.data.DataLoader(test_ds,  BATCH_SIZE, shuffle=False)
    return train, test, 10

# ---------- MODEL BUILDERS ----------
def build_llm_baseline():
    m = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_NAME,
        num_labels=2,
        torch_dtype="auto",
        use_safetensors=True,          # <-- force safetensors
        ignore_mismatched_sizes=True
    )
    m.resize_token_embeddings(len(tokenizer))
    for p in m.parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if any(k in n for k in ["classifier","layer.3","layer.2"]):
            p.requires_grad = True
    return m

def build_llm_morphreuse():
    m = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_NAME,
        num_labels=2,
        torch_dtype="auto",
        use_safetensors=True,          # <-- force safetensors
        ignore_mismatched_sizes=True
    )
    m.resize_token_embeddings(len(tokenizer))
    def wrap(layer): return MorphReuseLinear(layer.in_features, layer.out_features)
    for layer in m.bert.encoder.layer[-2:]:
        layer.attention.self.query   = wrap(layer.attention.self.query)
        layer.attention.self.key     = wrap(layer.attention.self.key)
        layer.attention.self.value   = wrap(layer.attention.self.value)
        layer.attention.output.dense = wrap(layer.attention.output.dense)
        layer.intermediate.dense     = wrap(layer.intermediate.dense)
        layer.output.dense           = wrap(layer.output.dense)
    m.classifier = wrap(m.classifier)
    SharedWeight.stats()
    for p in m.parameters(): 
      p.requires_grad = False
    for n, p in m.named_parameters():
        if any(k in n for k in ["shared","scale"]):
          p.requires_grad = True
    return m

def build_llm_lora():
    m = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_NAME,
        num_labels=2,
        torch_dtype="auto",
        use_safetensors=True,          # <-- force safetensors
        ignore_mismatched_sizes=True
    )
    m.resize_token_embeddings(len(tokenizer))
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32,
                     target_modules=["query","key","value","output.dense",
                                     "intermediate.dense","output.dense","classifier"],
                     lora_dropout=0.0, bias="none")
    return get_peft_model(m, cfg)

def build_llm_bitfit():
    m = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_NAME,
        num_labels=2,
        torch_dtype="auto",
        use_safetensors=True,          # <-- force safetensors
        ignore_mismatched_sizes=True
    )
    m.resize_token_embeddings(len(tokenizer))
    for p in m.parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if "bias" in n or "LayerNorm" in n: p.requires_grad = True
    return m

# ---------- VISION ----------
class VisionMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def build_vision_baseline(in_dim, n_class):
    return VisionMLP(in_dim, 512, n_class)

def build_vision_morphreuse(in_dim, n_class):
    m = VisionMLP(in_dim, 512, n_class)
    def wrap(layer):
        if isinstance(layer, nn.Linear):
            return MorphReuseLinear(layer.in_features, layer.out_features)
        return layer
    for name, module in m.named_children():
        if name == "net":
            for i, child in enumerate(module):
                if isinstance(child, nn.Linear):
                    module[i] = wrap(child)
    for p in m.parameters(): p.requires_grad = False
    for n,p in m.named_parameters():
        if any(k in n for k in ["shared","scale"]): p.requires_grad = True
    return m

def build_vision_lora(in_dim, n_class):
    m = VisionMLP(in_dim, 512, n_class)
    target = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)]
    cfg = LoraConfig(task_type='FEATURE_EXTRACTION', r=8, lora_alpha=32,
                     target_modules=target, lora_dropout=0.0, bias="none")
    return get_peft_model(m, cfg)

def build_vision_bitfit(in_dim, n_class):
    m = VisionMLP(in_dim, 512, n_class)
    for p in m.parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if "bias" in n: p.requires_grad = True
    return m

# ---------- TRAIN ----------
def train_and_eval(model, train_loader, val_loader, tag):
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    stats = mem_stats(model)
    print_mem(tag + " (Initial)", stats)

    def acc(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for *X, y in tqdm(loader, desc="eval"):
                X = [x.to(DEVICE) for x in X]
                y = y.to(DEVICE)
                out = model(*X) if len(X) == 3 else model(X[0])
                logits = out.logits if hasattr(out, 'logits') else out
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
        return 100 * correct / total

    train_acc, test_acc = [acc(train_loader)], [acc(val_loader)]
    for epoch in range(EPOCHS):
        model.train()
        for *X, y in tqdm(train_loader, desc=f"{tag} epoch {epoch+1}"):
            X = [x.to(DEVICE) for x in X]; y = y.to(DEVICE)
            opt.zero_grad()
            out = model(*X) if len(X) == 3 else model(X[0])
            logits = out.logits if hasattr(out, 'logits') else out
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        train_acc.append(acc(train_loader))
        test_acc.append(acc(val_loader))
        print(f"{tag} epoch {epoch+1}: Train {train_acc[-1]:.2f}%  Val {test_acc[-1]:.2f}%")

    stats = mem_stats(model)
    print_mem(tag + " (Final)", stats)
    return train_acc, test_acc

# ---------- MAIN ----------
if __name__ == "__main__":
    datasets = ['sst2', 'MNIST', 'FashionMNIST', 'CIFAR10']
    for ds in datasets:
        print(f"\n{'='*60}\n{ds}\n{'='*60}")
        if ds == 'sst2':
            train_loader = train_text
            val_loader   = val_text
            builders = {
                "Baseline": build_llm_baseline,
                "MorphReuse": build_llm_morphreuse,
                "LoRA": build_llm_lora,
                "BitFit": build_llm_bitfit,
            }
        else:
            train_loader, val_loader, n_class = get_vision_loaders(ds)
            in_dim = 28*28 if ds in ['MNIST','FashionMNIST'] else 3*32*32
            builders = {
                "Baseline": lambda: build_vision_baseline(in_dim, n_class),
                "MorphReuse": lambda: build_vision_morphreuse(in_dim, n_class),
                "LoRA": lambda: build_vision_lora(in_dim, n_class),
                "BitFit": lambda: build_vision_bitfit(in_dim, n_class),
            }
        for name, builder in builders.items():
            model = builder()
            total, train = sum(p.numel() for p in model.parameters()), \
                           sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name}: total={total:,}  trainable={train:,}  ({100*train/total:.2f}%)")
            train_and_eval(model, train_loader, val_loader, name)
            del model
            gc.collect()
