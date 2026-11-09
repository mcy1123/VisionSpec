#!/usr/bin/env python3
"""Probe target logit/hidden sensitivity to the video content at high token counts."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from shared import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--device-map", default="balanced")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    target, proc = common.load_target(device=dev, dtype=torch.float32,
                                      device_map=a.device_map)
    lm = target.lm_head
    class LM(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x):
            return self.inner(x.to(self.inner.weight.device)).to(x.device)
        @property
        def weight(self): return self.inner.weight
    target.lm_head = LM(lm)
    model = target.model
    class M(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, *args, **kwargs):
            out = self.inner(*args, **kwargs)
            if getattr(out, "last_hidden_state", None) is not None:
                out.last_hidden_state = out.last_hidden_state.to(dev)
            if getattr(out, "logits", None) is not None:
                out.logits = out.logits.to(dev)
            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                out.hidden_states = tuple(h.to(dev) for h in hs)
            return out
        def __getattr__(self, name):
            try: return super().__getattr__(name)
            except AttributeError: return getattr(self.inner, name)
    target.model = M(model)
    target.eval().requires_grad_(False)

    def encode(row):
        text = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        return {k: v.to(dev) for k, v in ins.items() if hasattr(v, "to")}

    real = pd.read_parquet(a.real).head(a.n).reset_index(drop=True)
    other = pd.read_parquet(a.other).head(a.n).reset_index(drop=True)
    rows = []
    for i in range(len(real)):
        outs = []
        for df in (real, other):
            ins = encode(df.iloc[i])
            with torch.no_grad():
                out = target.model(**ins, use_cache=False, output_hidden_states=True)
            logits = target.lm_head(out.last_hidden_state[:, -1:])[0, 0].float()
            outs.append((logits, out.last_hidden_state[0, -1].float()))
        (lr, hr), (lo, ho) = outs
        pr = torch.softmax(lr, -1); po = torch.softmax(lo, -1)
        kl = float((pr * (pr.clamp_min(1e-30).log() - po.clamp_min(1e-30).log())).sum())
        rows.append(dict(video=str(real.iloc[i]["video_name"]),
                         top1_real=int(lr.argmax()), top1_other=int(lo.argmax()),
                         top1_agree=bool(int(lr.argmax()) == int(lo.argmax())),
                         kl=kl, cos=float(torch.nn.functional.cosine_similarity(hr, ho, dim=0))))
        print(json.dumps(rows[-1]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
