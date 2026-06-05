#!/usr/bin/env python3
"""RULER-hard — long-context tasks that single-needle NIAH cannot see fail.

NIAH with one needle saturates at 100% even on aggressively quantized KV, so it
tells us nothing about where K vs V precision actually matters. These three harder
constructions force the model to use MORE of the buried context, which is exactly
where low-precision V (or K) starts dropping retrieval:

  - multikey   : 4 distractor passcodes, ask for ONE specific key (selectivity).
  - multivalue : N values logged for one key, must enumerate ALL (recall breadth).
  - vartrack   : a chained-assignment trail (X1=val; X2=X1; ...), ask the final var
                 (follow a dependency chain through distractors).

Built on niah.py's haystack / /tokenize / exact-match machinery. Deterministic and
seeded; accuracy per (task, ctx, depth) cell saved to JSON. Gentle defaults; raise
--trials / --ctx only with a server you own (eval :8002).
"""
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import niah

WORDS = ["harbor", "lantern", "meadow", "cobalt", "thistle", "marble", "cinder",
         "willow", "quartz", "ember", "saffron", "drift", "pewter", "almond"]


def _spread_depths(n):
    if n == 1:
        return [0.5]
    return [round(i / (n - 1), 3) for i in range(n)]


def insert_lines(sents, lines_at_depth):
    """Insert (depth, line) pairs into the sentence list at fractional depths."""
    out = list(sents)
    # insert from deepest to shallowest so indices stay valid
    for depth, line in sorted(lines_at_depth, key=lambda x: -x[0]):
        k = min(max(round(depth * len(out)), 0), len(out))
        out.insert(k, line)
    return out


def make_multikey(sents, rng):
    keys = rng.sample(WORDS, 4)
    codes = [f"{rng.randint(10000, 99999)}" for _ in keys]
    depths = _spread_depths(4)
    lines = [(d, f"The secret passcode for {k} is {c}.") for d, k, c in zip(depths, keys, codes)]
    ti = rng.randrange(4)
    body = " ".join(insert_lines(sents, lines))
    prompt = ("Read the following text carefully, then answer the question.\n\n" + body +
              f"\n\nQuestion: What is the secret passcode for {keys[ti]}? Reply with only the number.")
    return prompt, codes[ti], {"target": keys[ti], "distractors": codes}


def make_multivalue(sents, rng):
    key = rng.choice(WORDS)
    m = 4
    codes = [f"{rng.randint(10000, 99999)}" for _ in range(m)]
    depths = _spread_depths(m)
    lines = [(d, f"Record for {key}: {c}.") for d, c in zip(depths, codes)]
    body = " ".join(insert_lines(sents, lines))
    prompt = ("Read the following text carefully, then answer the question.\n\n" + body +
              f"\n\nQuestion: List ALL numbers recorded for {key}, separated by commas.")
    return prompt, codes, {"key": key}  # expected = list; hit = all present


def make_vartrack(sents, rng):
    seed = f"{rng.randint(10000, 99999)}"
    chain = 5
    names = [f"VAR_{rng.choice(WORDS).upper()}{i}" for i in range(chain)]
    lines = [f"{names[0]} = {seed}."]
    for i in range(1, chain):
        lines.append(f"{names[i]} = {names[i-1]}.")
    # add a couple of decoy assignments
    decoys = [f"VAR_{rng.choice(WORDS).upper()}D = {rng.randint(10000,99999)}." for _ in range(2)]
    all_lines = lines + decoys
    rng.shuffle(all_lines)
    depths = _spread_depths(len(all_lines))
    placed = [(d, ln) for d, ln in zip(depths, all_lines)]
    body = " ".join(insert_lines(sents, placed))
    prompt = ("Read the following text carefully. Variables are assigned in the text "
              "(some refer to earlier variables). Then answer.\n\n" + body +
              f"\n\nQuestion: What is the numeric value of {names[-1]}? Reply with only the number.")
    return prompt, seed, {"final_var": names[-1]}


TASKS = {"multikey": make_multikey, "multivalue": make_multivalue, "vartrack": make_vartrack}


def score(task, expected, answer):
    if task == "multivalue":
        return all(c in answer for c in expected)
    return expected in answer


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    depths = [float(x) for x in args.depths.split(",") if x.strip()]

    print(f"[ruler] building haystacks ctx={ctxs} via {base}/tokenize ...", flush=True)
    hay = {c: niah.build_haystack(base, args.model, c) for c in ctxs}
    for c in ctxs:
        print(f"  ctx~{c}: {len(hay[c])} sentences", flush=True)

    # build the full trial list (deterministic), then execute (optionally concurrent)
    specs = []
    for task in tasks:
        for c in ctxs:
            for dp in depths:
                for tr in range(args.trials):
                    rng = random.Random(hash((task, c, round(dp, 3), tr)) & 0x7FFFFFFF)
                    prompt, expected, meta = TASKS[task](hay[c], rng)
                    specs.append({"task": task, "ctx": c, "depth": dp, "trial": tr,
                                  "prompt": prompt, "expected": expected})

    def work(s):
        t0 = time.time()
        try:
            txt, ptok, ctok = niah.query(base, args.model, s["prompt"], args.max_tokens, args.timeout)
            hit = score(s["task"], s["expected"], txt)
            return {"task": s["task"], "ctx": s["ctx"], "depth": s["depth"], "trial": s["trial"],
                    "hit": hit, "ptok": ptok, "dt": round(time.time()-t0, 1)}
        except Exception as e:
            return {"task": s["task"], "ctx": s["ctx"], "depth": s["depth"], "trial": s["trial"],
                    "hit": False, "err": str(e)[:140], "dt": round(time.time()-t0, 1)}

    results = []
    done = 0
    if args.concurrency <= 1:
        for s in specs:
            r = work(s); results.append(r); done += 1
            print(f"  [{done}/{len(specs)}] {r['task']:<11} ctx={r['ctx']:>6} d={r['depth']:.2f} "
                  f"{'HIT ' if r['hit'] else 'MISS'} {r.get('err','')}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(work, s) for s in specs]
            for f in as_completed(futs):
                r = f.result(); results.append(r); done += 1
                print(f"  [{done}/{len(specs)}] {r['task']:<11} ctx={r['ctx']:>6} d={r['depth']:.2f} "
                      f"{'HIT ' if r['hit'] else 'MISS'} {r.get('err','')}", flush=True)

    # aggregate per (task, ctx)
    agg = {}
    for r in results:
        key = f"{r['task']}|{r['ctx']}"
        a = agg.setdefault(key, {"hit": 0, "n": 0})
        a["n"] += 1
        a["hit"] += 1 if r["hit"] else 0
    per = {k: {"acc": round(v["hit"]/v["n"], 4), "n": v["n"]} for k, v in agg.items()}
    per_task = {}
    for task in tasks:
        hs = sum(1 for r in results if r["task"] == task and r["hit"])
        ns = sum(1 for r in results if r["task"] == task)
        per_task[task] = {"acc": round(hs/ns, 4) if ns else None, "n": ns}

    out = {"tag": args.tag, "port": args.port, "model": args.model,
           "ctx": ctxs, "tasks": tasks, "depths": depths, "trials": args.trials,
           "per_task": per_task, "per_task_ctx": per, "results": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[ruler] per-task acc:")
    for task in tasks:
        print(f"    {task:<11}: acc={per_task[task]['acc']} (n={per_task[task]['n']})")
    print(f"[ruler] saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="step3p7")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", default="4000,16000,64000,128000")
    ap.add_argument("--tasks", default="multikey,multivalue,vartrack")
    ap.add_argument("--depths", default="0.25,0.5,0.75")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="eval server is ours -> raise to ~8; accuracy is conc-invariant")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"/home/hikari/bench/results/kvq_ruler_{args.tag}.json"
    run(args)


if __name__ == "__main__":
    main()
