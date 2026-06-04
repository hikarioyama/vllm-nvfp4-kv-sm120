#!/usr/bin/env python3
"""NIAH (needle-in-a-haystack) retrieval oracle for the NVFP4-KV Pareto study.

The *ruler* for every (K-dtype, V-dtype) capacity/precision trade-off: it measures
whether the model can still retrieve a buried fact as KV precision drops. Pure stdlib
HTTP client (no torch / no model load) so it runs on the host against a live vLLM
OpenAI server (:8001 = nvfp4 v0.1, :8002 = dev, or an fp8 server in a maint window).

Design (deterministic, exact-match):
  - Haystack = neutral digit-free prose, repeated to a precise token length (via the
    server's /tokenize endpoint, ±tol).
  - Needle   = "The secret passcode is <5-digit code>." inserted at a fractional depth.
    The code is seeded per (ctx, depth, trial) so runs are reproducible and unique.
  - Query    = "What is the secret passcode?" -> exact substring match of the code in
    the answer content (reasoning content is also scanned as a fallback).
  - Score    = accuracy per (ctx_len, depth) cell + overall, saved to JSON.

Gentle by default (concurrency=1, few trials) because it hammers PRODUCTION :8001 which
also serves the swarm. Crank --trials / --ctx / --concurrency only in a maintenance window.

Examples:
  # light live baseline against nvfp4 v0.1 on :8001 (safe to co-run with swarm)
  python3 niah.py --port 8001 --tag nvfp4_v01_live --ctx 1000,2000,4000,8000,16000 --trials 3
  # full sweep (maintenance window)
  python3 niah.py --port 8002 --tag dev_b2 --ctx 1000,4000,16000,32000,64000,128000 --trials 5
"""
import argparse, json, os, random, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# Neutral, digit-free haystack sentences (no numerals -> no false-positive code matches).
_HAY = [
    "The morning fog drifted slowly across the quiet valley below the ridge.",
    "A gardener trimmed the hedges while sparrows argued in the old oak tree.",
    "Rain tapped against the window as the kettle began to whistle softly.",
    "The librarian reshelved the worn volumes with a patient, practiced hand.",
    "Sunlight spilled over the meadow and warmed the dew on the tall grass.",
    "An old fisherman mended his nets by the harbor while gulls wheeled overhead.",
    "The baker dusted flour across the counter and shaped the soft dough.",
    "Children chased one another through the orchard beneath the apple boughs.",
    "A painter studied the changing light and mixed a quieter shade of blue.",
    "The carpenter sanded the table until the grain felt smooth as river stone.",
    "Wind moved through the pines and carried the scent of distant woodsmoke.",
    "The cartographer traced a coastline she had never had the chance to walk.",
    "A cellist practiced scales while the afternoon dimmed into a gentle dusk.",
    "The shepherd counted the flock as they wandered back across the hillside.",
    "Steam rose from the cobblestones after the brief and sudden summer shower.",
    "The watchmaker leaned close, coaxing a stubborn gear back into its place.",
    "A potter centered the clay and let the wheel hum beneath her steady palms.",
    "The traveler paused at the crossroads, unsure which lane led toward the sea.",
    "Lanterns swayed along the pier as the evening tide came quietly inward.",
    "The beekeeper lifted a frame, heavy and golden, and smiled at the harvest.",
]


def _post(base, path, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _ntok(base, model, text, timeout=30):
    try:
        d = _post(base, "/tokenize", {"model": model, "prompt": text}, timeout)
        return int(d["count"])
    except Exception:
        # fallback: rough char/4 estimate if /tokenize is unavailable
        return max(1, len(text) // 4)


def build_haystack(base, model, target_tokens, tol=0.02):
    """Repeat _HAY sentences until the token count is within tol of target_tokens."""
    # estimate tokens-per-sentence from a one-shot measurement
    block = " ".join(_HAY)
    btok = _ntok(base, model, block)
    per = btok / len(_HAY)
    n = max(1, int(target_tokens / per))
    sents = [_HAY[i % len(_HAY)] for i in range(n)]
    # fine-tune by adding/removing whole sentences
    for _ in range(64):
        text = " ".join(sents)
        t = _ntok(base, model, text)
        if abs(t - target_tokens) <= max(8, target_tokens * tol):
            break
        if t < target_tokens:
            sents.append(_HAY[len(sents) % len(_HAY)])
        else:
            if len(sents) <= 1:
                break
            sents.pop()
    return sents


def make_prompt(sents, depth, code):
    needle = f"The secret passcode is {code}."
    k = round(depth * len(sents))
    k = min(max(k, 0), len(sents))
    body = " ".join(sents[:k] + [needle] + sents[k:])
    return (
        "Read the following text carefully, then answer the question at the end.\n\n"
        + body
        + "\n\nQuestion: What is the secret passcode? Reply with only the number."
    )


def query(base, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    d = _post(base, "/v1/chat/completions", payload, timeout)
    msg = d["choices"][0]["message"]
    txt = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    usage = d.get("usage", {})
    return txt, usage.get("prompt_tokens"), usage.get("completion_tokens")


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    cells = []  # (ctx, depth)
    for c in ctxs:
        for dp in depths:
            cells.append((c, dp))

    # pre-build haystacks once per ctx (token measurement is the slow part)
    print(f"[niah] building haystacks for ctx={ctxs} via {base}/tokenize ...", flush=True)
    hay = {}
    for c in ctxs:
        hay[c] = build_haystack(base, args.model, c)
        approx = _ntok(base, args.model, " ".join(hay[c]))
        print(f"  ctx~{c}: {len(hay[c])} sentences -> {approx} tokens", flush=True)

    trials = []  # one dict per query
    for (c, dp) in cells:
        for t in range(args.trials):
            rng = random.Random(hash((c, round(dp, 4), t)) & 0x7FFFFFFF)
            code = f"{rng.randint(10000, 99999)}"
            prompt = make_prompt(hay[c], dp, code)
            trials.append({"ctx": c, "depth": dp, "trial": t, "code": code, "prompt": prompt})

    print(f"[niah] {len(trials)} queries (ctx×depth×trials), conc={args.concurrency}", flush=True)
    results = []

    def work(item):
        t0 = time.time()
        try:
            txt, ptok, ctok = query(base, args.model, item["prompt"], args.max_tokens, args.timeout)
            hit = item["code"] in txt
            return {**{k: item[k] for k in ("ctx", "depth", "trial", "code")},
                    "hit": hit, "ptok": ptok, "ctok": ctok, "dt": round(time.time() - t0, 2),
                    "ans": txt.strip()[:120]}
        except Exception as e:
            return {**{k: item[k] for k in ("ctx", "depth", "trial", "code")},
                    "hit": False, "err": str(e)[:160], "dt": round(time.time() - t0, 2)}

    done = 0
    if args.concurrency <= 1:
        for it in trials:
            results.append(work(it)); done += 1
            r = results[-1]
            print(f"  [{done}/{len(trials)}] ctx={r['ctx']} d={r['depth']:.2f} "
                  f"{'HIT ' if r['hit'] else 'MISS'} {r.get('err','')}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(work, it): it for it in trials}
            for f in as_completed(futs):
                results.append(f.result()); done += 1
                r = results[-1]
                print(f"  [{done}/{len(trials)}] ctx={r['ctx']} d={r['depth']:.2f} "
                      f"{'HIT ' if r['hit'] else 'MISS'} {r.get('err','')}", flush=True)

    # aggregate
    grid = {}
    for r in results:
        key = f"{r['ctx']}|{r['depth']}"
        g = grid.setdefault(key, {"hit": 0, "n": 0})
        g["n"] += 1
        g["hit"] += 1 if r["hit"] else 0
    per_ctx = {}
    for c in ctxs:
        hs = sum(1 for r in results if r["ctx"] == c and r["hit"])
        ns = sum(1 for r in results if r["ctx"] == c)
        per_ctx[c] = {"acc": round(hs / ns, 4) if ns else None, "n": ns}
    overall = round(sum(1 for r in results if r["hit"]) / len(results), 4) if results else None

    out = {
        "tag": args.tag, "port": args.port, "model": args.model,
        "ctx": ctxs, "depths": depths, "trials": args.trials,
        "overall_acc": overall,
        "per_ctx": per_ctx,
        "grid": {k: {"acc": round(v["hit"] / v["n"], 4), "n": v["n"]} for k, v in grid.items()},
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[niah] OVERALL acc={overall}  (n={len(results)})")
    print("[niah] per-ctx acc:")
    for c in ctxs:
        print(f"    ctx~{c:>7}: acc={per_ctx[c]['acc']}  (n={per_ctx[c]['n']})")
    print(f"[niah] saved -> {args.out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="step3p7")
    ap.add_argument("--tag", default="niah")
    ap.add_argument("--ctx", default="1000,2000,4000,8000,16000",
                    help="comma list of target context token lengths")
    ap.add_argument("--depths", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="reasoning model needs room to think AND answer; <=256 truncates -> empty answers")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="keep at 1 against production :8001; raise only in a maint window")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/hikari/bench/results/niah_{args.tag}.json"
    run(args)


if __name__ == "__main__":
    main()
