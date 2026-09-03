"""Tokenize a text corpus into train.bin / val.bin (uint16 GPT-2 BPE), the format
common.py's Data loader expects.

Our runs use the Dolma corpus (AI2, ODC-BY). Dolma is large and distributed as its own
shards; we do not redistribute it. Point --input at a directory of .txt / .jsonl(.gz)
shards (one document per line for .jsonl, key --text_key), or at a plain text file.

    python prepare_data.py --input /path/to/dolma_shards --out_dir data/dolma

The result is train.bin / val.bin of uint16 GPT-2 token ids (vocab padded to 50304 in
the model; the tokenizer itself emits ids < 50257). Set TDA_DATA_DIR=data/dolma (or
pass --data_dir to the training script) to use it.
"""
import argparse, os, glob, gzip, json
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")


def iter_texts(path, text_key):
    paths = [path] if os.path.isfile(path) else sorted(
        glob.glob(os.path.join(path, "**", "*"), recursive=True))
    for p in paths:
        if os.path.isdir(p):
            continue
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8", errors="ignore") as f:
            if p.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)[text_key]
                        except Exception:
                            continue
            else:
                yield f.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="text file or directory of shards")
    p.add_argument("--out_dir", default="data/dolma")
    p.add_argument("--text_key", default="text", help="jsonl field holding the document text")
    p.add_argument("--val_frac", type=float, default=0.0005)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids = []
    for i, text in enumerate(iter_texts(args.input, args.text_key)):
        ids.extend(enc.encode_ordinary(text))
        ids.append(enc.eot_token)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1} docs, {len(ids)/1e6:.1f}M tokens")
    ids = np.asarray(ids, dtype=np.uint16)
    n_val = int(len(ids) * args.val_frac)
    ids[n_val:].tofile(os.path.join(args.out_dir, "train.bin"))
    ids[:n_val].tofile(os.path.join(args.out_dir, "val.bin"))
    print(f"wrote {len(ids)-n_val} train / {n_val} val tokens to {args.out_dir}")


if __name__ == "__main__":
    main()
