import os, sys, traceback
from functools import partial

deleted = []

def hook(event, args):
    if event in ("os.remove", "os.unlink", "os.rmdir", "shutil.rmtree"):
        try:
            p = args[0]
            if isinstance(p, (bytes, bytearray)):
                p = p.decode("utf-8", "replace")
            deleted.append((event, str(p)))
        except Exception:
            pass

sys.addaudithook(hook)

sys.path.insert(0, r"F:\zhuomian\transformers_tasks-main\text_classification")
os.chdir(r"F:\zhuomian\transformers_tasks-main\text_classification")

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import convert_example

tok = AutoTokenizer.from_pretrained("bert-base-chinese")
ds = load_dataset("text", data_files={"train": "data/comment_classify/train.txt",
                                      "dev": "data/comment_classify/dev.txt"})
cf = partial(convert_example, tokenizer=tok, max_seq_len=128)
ds = ds.map(cf, batched=True)

print("PREP OK: train=%d dev=%d" % (ds["train"].num_rows, ds["dev"].num_rows))
print("DELETIONS: %d" % len(deleted))
from collections import Counter
c = Counter(os.path.dirname(p) for _, p in deleted)
for d, n in c.most_common(10):
    print("  %4d  %s" % (n, d))
