#!/usr/bin/env python3
"""Aggregate the 8-pattern KV-quality study into a bf16-off-referenced Δ matrix.

Reads kvq_{ppl,divergence,ruler}_{kv}_mtp{0,1}.json for every available pattern and
produces:
  - PPL    : Δnats/token vs bf16-off per context length + the position-bin curve.
  - DIVERGE: vs bf16-off — median first-divergence token index, token agreement,
             exact-match rate. Plus the MTP on-vs-off diff per KV config.
  - RULER  : per-task accuracy + Δ vs bf16-off.
Then a verdict paragraph: does K-protected mixed beat symmetric nvfp4? Where does the
rot concentrate? Does MTP amplify or hide it? Missing patterns are simply skipped, so
this runs on the 6 ready configs and again once mixed lands.

Pure stdlib; runs after the servers are down. Writes kv_quality_compare.md + .json.
"""
import argparse, glob, json, math, os, statistics

RESULTS = "/home/hikari/bench/results"
# (display, kv_key) — launcher KVDTYPE: bf16->auto
KVS = [("BF16", "bf16"), ("FP8", "fp8"), ("NVFP4", "nvfp4"), ("MIXED", "mixed")]
MTPS = [0, 1]
BASE = "bf16_mtp0"


def load(component, tag):
    p = f"{RESULTS}/kvq_{component}_{tag}.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def tags_available():
    out = []
    for _, kv in KVS:
        for m in MTPS:
            t = f"{kv}_mtp{m}"
            if any(load(c, t) for c in ("ppl", "divergence", "ruler")):
                out.append(t)
    return out


# ----------------------------------------------------------------------- PPL
def ppl_matrix():
    base = load("ppl", BASE)
    rows = {}
    ctxs = base["ctx"] if base else []
    for _, kv in KVS:
        for m in MTPS:
            t = f"{kv}_mtp{m}"
            d = load("ppl", t)
            if not d:
                continue
            row = {}
            for c in d["ctx"]:
                cur = d["per_ctx"].get(str(c), {})
                nll = cur.get("mean_nll")
                row[str(c)] = {"nll": nll, "ppl": cur.get("ppl")}
                if base and nll is not None:
                    bnll = base["per_ctx"].get(str(c), {}).get("mean_nll")
                    if bnll is not None:
                        row[str(c)]["dnats"] = round(nll - bnll, 5)
            rows[t] = row
    return ctxs, rows


# ----------------------------------------------------------------- DIVERGENCE
def _seq(d):
    return {r["id"]: r.get("token_ids") for r in d.get("results", []) if r.get("token_ids")}


def diverge(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return None
    fd = n
    for i in range(n):
        if a[i] != b[i]:
            fd = i
            break
    agree = sum(1 for i in range(n) if a[i] == b[i]) / n
    return fd, agree, (a == b)


def div_vs(base_seqs, cand_seqs):
    fds, agrees, exacts = [], [], 0
    common = set(base_seqs) & set(cand_seqs)
    for k in common:
        r = diverge(base_seqs[k], cand_seqs[k])
        if r:
            fds.append(r[0])
            agrees.append(r[1])
            exacts += 1 if r[2] else 0
    if not fds:
        return None
    return {"n": len(fds), "median_first_div": int(statistics.median(fds)),
            "mean_agree": round(sum(agrees) / len(agrees), 4),
            "exact_match_rate": round(exacts / len(fds), 4)}


def div_matrix():
    base = load("divergence", BASE)
    base_seqs = _seq(base) if base else {}
    vs_base, mtp_onoff = {}, {}
    for _, kv in KVS:
        seqs = {m: (_seq(load("divergence", f"{kv}_mtp{m}") or {})) for m in MTPS}
        for m in MTPS:
            t = f"{kv}_mtp{m}"
            if seqs[m] and base_seqs:
                r = div_vs(base_seqs, seqs[m])
                if r:
                    vs_base[t] = r
        # MTP on vs off for the same KV (lossless-MTP test)
        if seqs[0] and seqs[1]:
            r = div_vs(seqs[0], seqs[1])
            if r:
                mtp_onoff[kv] = r
    return vs_base, mtp_onoff


# --------------------------------------------------------------------- RULER
def ruler_matrix():
    base = load("ruler", BASE)
    btask = base.get("per_task", {}) if base else {}
    rows = {}
    for _, kv in KVS:
        for m in MTPS:
            t = f"{kv}_mtp{m}"
            d = load("ruler", t)
            if not d:
                continue
            row = {}
            for task, v in d.get("per_task", {}).items():
                acc = v.get("acc")
                cell = {"acc": acc, "n": v.get("n")}
                bacc = btask.get(task, {}).get("acc")
                if acc is not None and bacc is not None:
                    cell["dacc"] = round(acc - bacc, 4)
                row[task] = cell
            rows[t] = row
    return rows


# -------------------------------------------------------------------- report
def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def control_block():
    """bf16ctrl_mtp0 = a 2nd independent bf16-off load. Its Δ vs the gold bf16_mtp0
    is the pure load-to-load NOISE FLOOR — the bar every KV-precision Δ must clear."""
    bp, cp = load("ppl", BASE), load("ppl", "bf16ctrl_mtp0")
    bd, cd = load("divergence", BASE), load("divergence", "bf16ctrl_mtp0")
    L = ["\n## ⓪ ノイズ床 control (bf16-off 2回ロード, bf16ctrl vs bf16)\n"]
    if cp and bp:
        cells = []
        for c in bp["ctx"]:
            bn = bp["per_ctx"].get(str(c), {}).get("mean_nll")
            cn = cp["per_ctx"].get(str(c), {}).get("mean_nll")
            cells.append("—" if (bn is None or cn is None) else f"{cn-bn:+.4f}")
        L.append("| | " + " | ".join(str(c) for c in bp["ctx"]) + " |")
        L.append("|" + "---|" * (len(bp["ctx"]) + 1))
        L.append("| PPL Δnats 床 | " + " | ".join(cells) + " |")
    if cd and bd:
        r = div_vs(_seq(bd), _seq(cd))
        if r:
            L.append(f"\n- 分岐 床: median first-div=**{r['median_first_div']}** tok, "
                     f"token 一致率=**{r['mean_agree']:.3f}**, 完全一致率={r['exact_match_rate']:.3f}")
    L.append("\n*KV 精度の Δ がこの床を超えて初めて「実劣化」。床未満なら非決定性ノイズ。*")
    return "\n".join(L)


def build_md(ctxs, ppl, vs_base, mtp_onoff, ruler):
    L = []
    L.append("# KV-cache 量子化 品質劣化 — 8 パターン比較 (bf16-off 基準)\n")
    avail = tags_available()
    L.append(f"対象パターン: {', '.join(avail)}\n")
    L.append(f"gold 基準: `{BASE}`  |  KV 構成 = {{bf16, fp8, nvfp4, mixed(K=fp8/V=nvfp4)}} × MTP{{0,1}}\n")

    # PPL
    L.append("\n## ① Perplexity — Δnats/token vs bf16-off\n")
    if ctxs:
        head = "| pattern | " + " | ".join(f"{c} (Δnats)" for c in ctxs) + " |"
        L.append(head)
        L.append("|" + "---|" * (len(ctxs) + 1))
        for t in avail:
            if t not in ppl:
                continue
            cells = []
            for c in ctxs:
                e = ppl[t].get(str(c), {})
                d = e.get("dnats")
                ppv = e.get("ppl")
                cells.append("—" if d is None else f"{d:+.4f} (ppl {fmt(ppv,3)})")
            L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("\n*Δnats>0 は劣化。bf16-off 行は定義上 0。単調性 bf16≤fp8≤nvfp4 を確認。*\n")

    # noise-floor control (bf16ctrl = a 2nd independent bf16-off load)
    L.append(control_block())

    # divergence
    L.append("\n## ② Greedy 分岐 vs bf16-off\n")
    L.append("| pattern | n | median first-div (tok) | token 一致率 | 完全一致率 |")
    L.append("|---|---|---|---|---|")
    for t in avail:
        r = vs_base.get(t)
        if not r:
            continue
        L.append(f"| {t} | {r['n']} | {r['median_first_div']} | "
                 f"{fmt(r['mean_agree'])} | {fmt(r['exact_match_rate'])} |")
    L.append("\n### MTP on vs off (同一 KV、lossless-MTP 仮説の実測)\n")
    L.append("| KV | n | median first-div (tok) | token 一致率 | 完全一致率 |")
    L.append("|---|---|---|---|---|")
    for _, kv in KVS:
        r = mtp_onoff.get(kv)
        if not r:
            continue
        L.append(f"| {kv} | {r['n']} | {r['median_first_div']} | "
                 f"{fmt(r['mean_agree'])} | {fmt(r['exact_match_rate'])} |")

    # ruler
    L.append("\n## ③ RULER-hard — task accuracy (Δacc vs bf16-off)\n")
    tasks = ["multikey", "multivalue", "vartrack"]
    L.append("| pattern | " + " | ".join(tasks) + " |")
    L.append("|" + "---|" * (len(tasks) + 1))
    for t in avail:
        if t not in ruler:
            continue
        cells = []
        for task in tasks:
            e = ruler[t].get(task, {})
            acc = e.get("acc")
            dacc = e.get("dacc")
            cells.append("—" if acc is None else f"{fmt(acc,3)}" +
                         ("" if dacc is None else f" (Δ{dacc:+.3f})"))
        L.append(f"| {t} | " + " | ".join(cells) + " |")

    # verdict (data-driven hints)
    L.append("\n## Verdict\n")
    L.append(_verdict(ctxs, ppl, vs_base, mtp_onoff, ruler))
    return "\n".join(L)


def _verdict(ctxs, ppl, vs_base, mtp_onoff, ruler):
    out = []
    # mixed vs nvfp4 on PPL (mtp0)
    def last_dnats(t):
        if t not in ppl or not ctxs:
            return None
        return ppl[t].get(str(ctxs[-1]), {}).get("dnats")
    nv, mx = last_dnats("nvfp4_mtp0"), last_dnats("mixed_mtp0")
    if nv is not None and mx is not None:
        if mx < nv:
            out.append(f"- **混合 < 対称nvfp4**: 最長文脈で Δnats mixed={mx:+.4f} < nvfp4={nv:+.4f} "
                       f"→ K 保護 (K=fp8) が retrieval 品質に効いている。hybrid は意味あり。")
        else:
            out.append(f"- **混合 ≈/> 対称nvfp4**: Δnats mixed={mx:+.4f} vs nvfp4={nv:+.4f} "
                       f"→ K 保護の効果薄。V をさらに潰す (3-bit) 方向に全振り検討。")
    # control sanity: bf16-off self
    out.append("- ノイズ床 control・単調性は ① 表で確認 (bf16-off 自己 ΔPPL≈0, bf16≤fp8≤nvfp4)。")
    # MTP effect
    for _, kv in KVS:
        r = mtp_onoff.get(kv)
        if r:
            out.append(f"- MTP on/off ({kv}): token 一致率 {r['mean_agree']:.3f}, "
                       f"完全一致率 {r['exact_match_rate']:.3f} "
                       f"→ {'ほぼ lossless' if r['mean_agree']>0.98 else 'MTP が出力を動かす'}。")
    return "\n".join(out) if out else "- (データ不足: 採取済みパターンが少ない)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default=f"{RESULTS}/kv_quality_compare.md")
    ap.add_argument("--out-json", default=f"{RESULTS}/kv_quality_compare.json")
    args = ap.parse_args()
    ctxs, ppl = ppl_matrix()
    vs_base, mtp_onoff = div_matrix()
    ruler = ruler_matrix()
    md = build_md(ctxs, ppl, vs_base, mtp_onoff, ruler)
    with open(args.out_md, "w") as f:
        f.write(md)
    with open(args.out_json, "w") as f:
        json.dump({"ctxs": ctxs, "ppl": ppl, "divergence_vs_base": vs_base,
                   "mtp_onoff": mtp_onoff, "ruler": ruler}, f, indent=2)
    print(md)
    print(f"\n[compare] saved -> {args.out_md}")


if __name__ == "__main__":
    main()
