#!/usr/bin/env python3
"""Greedy divergence — attributes quality loss DIRECTLY to the KV cache.

PPL measures distribution drift; this measures whether that drift actually changes
what the model *says*. Same fixed prompts, greedy (temp=0, conc=1), max 512 tokens,
recorded as token-id sequences. The comparator diffs every pattern against the
bf16-off gold run: first-divergence index, token agreement rate, exact-match rate.
Because MTP runs the SAME KV dtype for the draft, the on-vs-off divergence is the
empirical test of the "MTP is lossless" assumption.

This script is a pure GENERATOR: it records (token_ids, text) per prompt for ONE
pattern. compare_kv_quality.py does the baseline diff (servers are already down by then).

Prompt set (~60, persisted so all 8 patterns see identical inputs):
  - short: math / code / reasoning / QA (deterministic, hardcoded)
  - long : 16k & 64k haystack with a buried needle + question (seeded, niah-style)
"""
import argparse, hashlib, json, os, time, urllib.request
import niah  # reuse the haystack builder + _HAY sentences

SHORT = [
    ("math", "Compute 27 * 34 + 19. Show your steps, then give the final number."),
    ("math", "What is the remainder when 2^20 is divided by 1000? Explain briefly."),
    ("math", "A train travels 60 km in 45 minutes. What is its speed in km/h?"),
    ("math", "Solve for x: 3x + 7 = 2x + 19. Give the value of x."),
    ("math", "What is the sum of the first 15 positive odd integers?"),
    ("code", "Write a Python function is_palindrome(s) that ignores case and spaces."),
    ("code", "In Python, reverse a linked list iteratively. Give the function only."),
    ("code", "Write a SQL query selecting the 2nd-highest salary from a table Employee(salary)."),
    ("code", "Explain what this does: lambda xs: [x for x in xs if x % 2 == 0]."),
    ("code", "Write a regex that matches a valid IPv4 octet (0-255)."),
    ("reason", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Why?"),
    ("reason", "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?"),
    ("reason", "Five people shake hands once with everyone else. How many handshakes total?"),
    ("reason", "You have 8 balls, one heavier. Find it in 2 weighings with a balance. How?"),
    ("reason", "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?"),
    ("qa", "Who wrote the play 'Hamlet', and in roughly what era?"),
    ("qa", "What is the chemical symbol for gold, and what group is it in?"),
    ("qa", "Name the largest moon of Saturn and one notable fact about it."),
    ("qa", "What causes the seasons on Earth? Answer in two sentences."),
    ("qa", "What is the difference between TCP and UDP? Be concise."),
    ("qa", "Explain what a hash table is and its average lookup complexity."),
    ("reason", "Arrange the words 'time', 'flies', 'arrow' into a well-known saying and explain it."),
    ("code", "Write a Python one-liner to flatten a list of lists."),
    ("math", "Differentiate f(x) = x^3 - 2x with respect to x."),
]


def _post(base, path, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tok_ids(base, model, text, timeout=120):
    """Return the token-id list for `text` (vLLM /tokenize returns 'tokens')."""
    try:
        d = _post(base, "/tokenize", {"model": model, "prompt": text,
                                      "add_special_tokens": False}, timeout)
        t = d.get("tokens")
        if t is not None:
            return t
        return list(range(int(d["count"])))  # degenerate fallback (length only)
    except Exception:
        return []


def build_prompts(base, model, prompts_file):
    if os.path.exists(prompts_file):
        with open(prompts_file) as f:
            return json.load(f)
    items = [{"id": f"short_{i:02d}", "kind": k, "prompt": p} for i, (k, p) in enumerate(SHORT)]
    # long-context needles (niah-style), 16k and 64k, a few depths each
    for c in (16000, 64000):
        sents = niah.build_haystack(base, model, c)
        for depth in (0.25, 0.5, 0.75):
            rng = int(hashlib.sha1(f"{c}_{depth}".encode()).hexdigest(), 16) % 90000 + 10000
            prompt = niah.make_prompt(sents, depth, str(rng))
            items.append({"id": f"long_{c}_{int(depth*100)}", "kind": "longctx", "prompt": prompt})
    os.makedirs(os.path.dirname(prompts_file) or ".", exist_ok=True)
    with open(prompts_file, "w") as f:
        json.dump(items, f)
    return items


def generate(base, model, prompt, max_tokens, timeout):
    """Raw /v1/completions greedy continuation. We deliberately do NOT use the chat
    endpoint: the step3p5 reasoning parser drops ALL tokens when generation is
    truncated mid-reasoning (content==reasoning_content==None at finish=length),
    which destroys the token stream we need to diff. /v1/completions returns the raw
    text plus logprobs.tokens — the exact generated token sequence, no re-tokenize."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "logprobs": 1, "stream": False}
    d = _post(base, "/v1/completions", payload, timeout)
    ch = d["choices"][0]
    text = ch.get("text") or ""
    toks = ((ch.get("logprobs") or {}).get("tokens")) or []
    usage = d.get("usage", {})
    return text, toks, usage.get("completion_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="step3p7")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompts-file", default="/home/hikari/bench/results/kvq_prompts_divergence.json")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/hikari/bench/results/kvq_divergence_{args.tag}.json"
    base = f"http://127.0.0.1:{args.port}"

    items = build_prompts(base, args.model, args.prompts_file)
    print(f"[div] {len(items)} prompts, tag={args.tag}", flush=True)
    results = []
    for i, it in enumerate(items):
        t0 = time.time()
        try:
            text, toks, ctok = generate(base, args.model, it["prompt"], args.max_tokens, args.timeout)
            results.append({"id": it["id"], "kind": it["kind"], "n_gen": ctok,
                            "n_ids": len(toks), "token_ids": toks,
                            "sha": hashlib.sha1(text.encode()).hexdigest()[:16],
                            "text_head": text[:200]})
        except Exception as e:
            results.append({"id": it["id"], "kind": it["kind"], "err": str(e)[:160]})
        r = results[-1]
        print(f"  [{i+1}/{len(items)}] {it['id']:<16} n_ids={r.get('n_ids')} "
              f"sha={r.get('sha','-')} {r.get('err','')} ({time.time()-t0:.1f}s)", flush=True)

    out = {"tag": args.tag, "port": args.port, "model": args.model,
           "max_tokens": args.max_tokens, "n_prompts": len(items), "results": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[div] saved -> {args.out}")


if __name__ == "__main__":
    main()
