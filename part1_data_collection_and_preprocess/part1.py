import subprocess
subprocess.run([
    'pip', 'install', '-q',
    'datasets', 'lxml', 'cairosvg',
    'tokenizers', 'sentencepiece', 'tqdm', 'numpy'
], check=True)
print("Packages installed.")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, re, json
from lxml import etree
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = '/content/drive/MyDrive/svg-lm-scaling'

DIRS = [
    'data/clean',
    'data/tokenized',
    'tokenizer',
    'checkpoints',
    'logs',
    'samples',
    'configs',
]

for d in DIRS:
    os.makedirs(f'{BASE_DIR}/{d}', exist_ok=True)

print(f"\nAll directories ready under: {BASE_DIR}")


BASE_DIR = '/content/drive/MyDrive/svg-lm-scaling'
CLEAN_DIR = f'{BASE_DIR}/data/clean'
OUTPUT_FILE = f'{CLEAN_DIR}/all_svgs.jsonl'

MIN_CHARS = 50
MAX_CHARS = 8192   # ~1024-2048 tokens depending on tokenizer

# https://platform.openai.com/tokenizer#:~:text=translates%20to%20roughly-,%C2%BE%20of%20a%20word,-(so%20100%20tokens
# Token estimate: OpenAI reports ~4 chars/token for English (= 0.25 tokens/char).
# SVG has more varied numbers and less repetitive structure than English so we can use a bit higher ratio

TOKEN_ESTIMATE_RATIO = 0.35
TARGET_TOKENS = 102_000_000   # 100M training + 2M val/test


def clean_svg(svg_str: str):
    """Return cleaned SVG string, or None if invalid/out-of-range."""
    if not svg_str:
        return None
    svg_str = svg_str.strip()
    if len(svg_str) < MIN_CHARS or len(svg_str) > MAX_CHARS:
        return None

    try:
        parser = etree.XMLParser(remove_comments=True, remove_pis=True, recover=False)
        root = etree.fromstring(svg_str.encode('utf-8'), parser)
    except etree.XMLSyntaxError:
        return None

    # Remove noise elements
    SVG_NS = 'http://www.w3.org/2000/svg'
    for tag in ['metadata', 'title', 'desc', 'defs']:
        for elem in root.findall(f'.//{{{SVG_NS}}}{tag}'):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)

    etree.cleanup_namespaces(root)
    svg_out = etree.tostring(root, encoding='unicode')

    # Round floats to 1 decimal place  (reduces vocab size significantly)
    def _round(m):
        try:
            v = float(m.group())
            r = round(v, 1)
            return str(int(r)) if r == int(r) else f'{r:.1f}'
        except Exception:
            return m.group()

    svg_out = re.sub(r'-?\d+\.\d{2,}', _round, svg_out)

    # Collapse whitespace
    svg_out = re.sub(r'\s+', ' ', svg_out).strip()

    return svg_out


def _find_svg_field(item: dict) -> str:
    """Try common field names to find the SVG string."""
    for key in ('Svg', 'text', 'content', 'code'):
        val = item.get(key, '')
        if isinstance(val, str) and val.strip().startswith('<'):
            return val
    return ''


def load_and_clean(dataset_name: str, split: str = 'train',
                   max_items: int = None, streaming: bool = False) -> list:
    print(f"\nLoading  {dataset_name}  (split={split}, streaming={streaming})")
    ds = load_dataset(dataset_name, split=split, streaming=streaming)

    cleaned, skipped = [], 0
    iterator = tqdm(ds, total=max_items, desc='Cleaning')

    for item in iterator:
        if max_items and len(cleaned) + skipped >= max_items:
            break
        svg = _find_svg_field(item)
        result = clean_svg(svg)
        if result:
            cleaned.append(result)
        else:
            skipped += 1

    print(f"Kept {len(cleaned):,} Skipped {skipped:,}")
    return cleaned


all_svgs = []

# Primary dataset svg-icons-simple
all_svgs += load_and_clean('starvector/svg-icons-simple')

def _estimated_tokens(svgs):
    return int(sum(len(s) for s in svgs) * TOKEN_ESTIMATE_RATIO)

print(f"\nEstimated tokens so far: {_estimated_tokens(all_svgs):,}")

# Emoji dataset (~14.5 MB), we proceed only after checking if it is lower than the target
if _estimated_tokens(all_svgs) < TARGET_TOKENS:
    all_svgs += load_and_clean('starvector/svg-emoji-simple')
    print(f"Estimated tokens so far: {_estimated_tokens(all_svgs):,}")

# Stack dataset (large) and stream only what we need
if _estimated_tokens(all_svgs) < TARGET_TOKENS:
    needed_chars = (TARGET_TOKENS - _estimated_tokens(all_svgs)) / TOKEN_ESTIMATE_RATIO
    # Assume avg ~500 chars per SVG after cleaning
    max_needed = int(needed_chars / 500) + 10_000
    print(f"\nNeed ~{max_needed:,} more SVGs from svg-stack-simpl (streaming)…")
    all_svgs += load_and_clean(
        'starvector/svg-stack-simple',
        max_items=max_needed,
        streaming=True,
    )
    print(f"Estimated tokens so far: {_estimated_tokens(all_svgs):,}")
    print(f"Need ~{max_needed:,} more SVGs from svg-fonts-simple (streaming)…")

    # Font dataset
    all_svgs += load_and_clean(
        'starvector/svg-fonts-simple',
        max_items=max_needed,
        streaming=True,
    )
    print(f"Estimated tokens so far: {_estimated_tokens(all_svgs):,}")

print(f"\nSaving {len(all_svgs):,} SVGs to {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for svg in all_svgs:
        f.write(json.dumps({'svg': svg}) + '\n')

total_chars = sum(len(s) for s in all_svgs)
stats = {
    'total_svgs': len(all_svgs),
    'total_chars': total_chars,
    'estimated_tokens':  _estimated_tokens(all_svgs),
    'avg_chars_per_svg': total_chars // max(len(all_svgs), 1),
    'min_chars': MIN_CHARS,
    'max_chars': MAX_CHARS,
}
with open(f'{CLEAN_DIR}/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\nDataset stats:")
for k, v in stats.items():
    print(f"{k}: {v:,}")


# finding the field names
ds = load_dataset('starvector/svg-icons-simple', split='train')
item = ds[0]

print("Field names:", list(item.keys()))
print()
for k, v in item.items():
    val_str = str(v)
    print(f"  '{k}': {val_str[:120]}")


import os, json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing


BASE_DIR  = '/content/drive/MyDrive/svg-lm-scaling'
CLEAN_DIR = f'{BASE_DIR}/data/clean'
TOK_DIR   = f'{BASE_DIR}/tokenizer'
os.makedirs(TOK_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)


VOCAB_SIZE = 4096  # 1K–8K are all reasonable but 4096 is a good balance
INPUT_FILE = f'{CLEAN_DIR}/all_svgs.jsonl'

# building a plain-text iterator from the jsonl

def svg_iterator(path, limit=None):
    """Yield raw SVG strings one at a time (memory-efficient)."""
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                yield json.loads(line)['svg']
            except Exception:
                continue

# Write SVGs to a temp txt file and the tokenizers trainer reads files
TMP_CORPUS = '/tmp/svg_corpus.txt'
print(f"Writing corpus to {TMP_CORPUS} …")
count = 0
with open(TMP_CORPUS, 'w', encoding='utf-8') as out:
    for svg in svg_iterator(INPUT_FILE):
        out.write(svg + '\n')
        count += 1
print(f"Corpus: {count:,} lines written.")


