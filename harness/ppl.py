#!/usr/bin/env python3
"""Perplexity sweep — the most sensitive KV-precision quality probe.

Single-needle NIAH saturates at 100% and is blind to the slow precision rot that
KV quantization causes. Token-level NLL on natural text is not: every token is a
measurement, and the degradation shows up as a few extra nats long before any
retrieval task fails. This is the headline metric of the 8-pattern KV-quality study.

Method (vLLM /v1/completions, teacher-forced, no generation):
  - Build ONE fixed raw prompt per context length {2k,8k,32k,128k} by concatenating
    Pile-10K documents to a precise token count (via the server's /tokenize). The
    prompts are PERSISTED (kvq_prompts_ppl.json) so every one of the 8 patterns scores
    the byte-identical input.
  - POST {prompt, max_tokens:0, echo:true, prompt_logprobs:0, temperature:0} and read
    back choices[0].prompt_logprobs -> the selected (actual) prompt-token logprob at
    every position.
  - Report mean NLL (nats/token) and PPL=exp(NLL) with the first `--warmup` tokens
    excluded, plus position-binned NLL (where the rot grows). The comparator turns
    these into Δnats vs the bf16-off gold row.

Build mode needs `datasets` (or pyarrow) to read Pile-10K offline; the scoring path is
pure stdlib. Run with a python that has datasets (e.g. ~/eigenself/.venv/bin/python) the
first time so the prompts file gets built; later runs just load it.
"""
import argparse, json, math, os, sys, urllib.request

PROMPTS_DEFAULT = "/home/hikari/bench/results/kvq_prompts_ppl.json"
PILE_DEFAULT = os.path.expanduser(
    "~/.cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/"
    "127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
)


# --------------------------------------------------------------------------- io
def _post(base, path, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ntok(base, model, text, timeout=120):
    d = _post(base, "/tokenize", {"model": model, "prompt": text}, timeout)
    return int(d["count"])


# ----------------------------------------------------------------- pile loading
def iter_pile_texts(pile_path):
    """Yield document text strings from the offline Pile-10K arrow cache."""
    try:
        from datasets import Dataset  # type: ignore
        ds = Dataset.from_file(pile_path)
        for row in ds:
            t = row.get("text")
            if t:
                yield t
        return
    except Exception as e:
        sys.stderr.write(f"[ppl] datasets load failed ({e}); trying pyarrow\n")
    import pyarrow.ipc as ipc  # type: ignore
    with ipc.open_file(pile_path) as rdr:
        tbl = rdr.read_all()
    for t in tbl.column("text").to_pylist():
        if t:
            yield t


# ------------------------------------------------------------ prompt building
def build_prompt(base, model, target_tokens, pile_path, tol=0.01):
    """Concatenate Pile docs to ~target_tokens, then binary-search a char cutoff
    so the tokenized length lands within tol of target."""
    buf, approx_chars = [], 0
    # ~3.8 chars/token is a safe lower bound; overshoot then trim down.
    need_chars = int(target_tokens * 5)
    for t in iter_pile_texts(pile_path):
        buf.append(t.strip())
        approx_chars += len(t) + 2
        if approx_chars >= need_chars:
            break
    text = "\n\n".join(buf)
    # ensure we actually overshoot in tokens; if not, keep appending
    while ntok(base, model, text) < target_tokens:
        more = []
        for i, t in enumerate(iter_pile_texts(pile_path)):
            more.append(t.strip())
            if i > 200:
                break
        text = text + "\n\n" + "\n\n".join(more)
    # binary search on character length for an exact-ish token count
    lo, hi = 0, len(text)
    best = text
    for _ in range(24):
        mid = (lo + hi) // 2
        cand = text[:mid]
        n = ntok(base, model, cand)
        if abs(n - target_tokens) <= max(8, int(target_tokens * tol)):
            best = cand
            break
        if n < target_tokens:
            lo = mid
        else:
            hi = mid
            best = cand
    return best


def load_or_build_prompts(base, model, ctxs, prompts_file, pile_path):
    prompts = {}
    if os.path.exists(prompts_file):
        with open(prompts_file) as f:
            prompts = json.load(f)
    changed = False
    for c in ctxs:
        key = str(c)
        if key not in prompts:
            print(f"[ppl] building Pile prompt for ctx~{c} tokens ...", flush=True)
            prompts[key] = build_prompt(base, model, c, pile_path)
            changed = True
    if changed:
        os.makedirs(os.path.dirname(prompts_file) or ".", exist_ok=True)
        with open(prompts_file, "w") as f:
            json.dump(prompts, f)
        print(f"[ppl] persisted prompts -> {prompts_file}", flush=True)
    return prompts


# ------------------------------------------------------------------ scoring
def extract_token_nlls(resp):
    """Pull per-position NLL (-logprob of the actual prompt token) from a vLLM
    completions response with echo+prompt_logprobs. Robust to the dict shape:
    prompt_logprobs is a list (len = #prompt tokens); element 0 is None; each other
    element is {token_id_str: {"logprob": float, ...}} (with prompt_logprobs=0 the
    single entry IS the actual token)."""
    ch = resp["choices"][0]
    pl = ch.get("prompt_logprobs")
    nlls = []
    if pl:
        for entry in pl:
            if not entry:
                continue
            # entry: {tok_id: {"logprob":..}} or {tok_id: float}
            vals = list(entry.values())
            v = vals[0]
            lp = v["logprob"] if isinstance(v, dict) else float(v)
            if lp is not None and math.isfinite(lp):
                nlls.append(-lp)
        return nlls
    # fallback: OpenAI legacy logprobs.token_logprobs
    lg = ch.get("logprobs") or {}
    for lp in (lg.get("token_logprobs") or []):
        if lp is not None and math.isfinite(lp):
            nlls.append(-lp)
    return nlls


def score_ctx(base, model, prompt, warmup, n_bins, timeout):
    payload = {"model": model, "prompt": prompt, "max_tokens": 0,
               "echo": True, "prompt_logprobs": 0, "temperature": 0}
    resp = _post(base, "/v1/completions", payload, timeout)
    nlls = extract_token_nlls(resp)
    if len(nlls) <= warmup + 4:
        return {"n_tokens": len(nlls), "mean_nll": None, "ppl": None, "bins": [],
                "err": f"too few token logprobs ({len(nlls)})"}
    body = nlls[warmup:]
    mean_nll = sum(body) / len(body)
    # position bins over the post-warmup range
    bins = []
    step = max(1, len(body) // n_bins)
    for b in range(0, len(body), step):
        seg = body[b:b + step]
        if seg:
            bins.append(round(sum(seg) / len(seg), 5))
    return {"n_tokens": len(nlls), "scored": len(body),
            "mean_nll": round(mean_nll, 5), "ppl": round(math.exp(mean_nll), 4),
            "bins": bins[:n_bins]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="step3p7")
    ap.add_argument("--tag", required=True, help="e.g. nvfp4_mtp0")
    ap.add_argument("--ctx", default="2000,8000,32000,128000")
    ap.add_argument("--warmup", type=int, default=256)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--prompts-file", default=PROMPTS_DEFAULT)
    ap.add_argument("--pile", default=PILE_DEFAULT)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/hikari/bench/results/kvq_ppl_{args.tag}.json"
    base = f"http://127.0.0.1:{args.port}"
    ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]

    prompts = load_or_build_prompts(base, args.model, ctxs, args.prompts_file, args.pile)

    per_ctx = {}
    for c in ctxs:
        print(f"[ppl] scoring ctx~{c} (tag={args.tag}) ...", flush=True)
        try:
            r = score_ctx(base, args.model, prompts[str(c)], args.warmup, args.bins, args.timeout)
        except Exception as e:
            r = {"mean_nll": None, "ppl": None, "bins": [], "err": str(e)[:160]}
        per_ctx[str(c)] = r
        print(f"    ctx~{c}: nll={r.get('mean_nll')} ppl={r.get('ppl')} "
              f"ntok={r.get('n_tokens')} {r.get('err','')}", flush=True)

    out = {"tag": args.tag, "port": args.port, "model": args.model,
           "ctx": ctxs, "warmup": args.warmup, "per_ctx": per_ctx}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ppl] saved -> {args.out}")


if __name__ == "__main__":
    main()
