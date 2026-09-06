# NVFP4 KV cache for vLLM on SM120

This branch is a surgical port of
[`hikarioyama/vllm-nvfp4-kv-sm120`](https://github.com/hikarioyama/vllm-nvfp4-kv-sm120)
to the following installed package pair:

| Component | Exact supported version |
|---|---|
| vLLM | `0.27.1` |
| FlashInfer | `0.6.16.post3` |
| GPU | SM120 consumer Blackwell, including RTX 5090 |

It enables `--kv-cache-dtype nvfp4` through FlashInfer FA2 without downgrading
vLLM, replacing the wheel, or rebuilding vLLM. The installer modifies four
files only, validates their exact hashes, stages and checks the patch before
installation, keeps recoverable backups, and is safe to rerun.

## Why this is still needed after vLLM 0.28.0

There are three separate FP4 capabilities:

1. NVFP4 **model weights/GEMM** on Blackwell.
2. NVFP4 **KV cache** support in vLLM generally.
3. NVFP4 **KV cache on SM120/SM121**.

The first two exist upstream. The third does not ship in either vLLM 0.27.1
or the tagged vLLM 0.28.0 FlashInfer backend: that path still accepts NVFP4 KV
only on SM100 with `trtllm-gen`. v0.28.0 adds SM12x XQA and several other
NVFP4 improvements, but those are not the missing SM120 NVFP4-KV routing.
Accordingly, a model card saying that FP4 works on Blackwell normally refers to
its quantized weights, not to the independent `--kv-cache-dtype nvfp4` option.

Relevant upstream trail:

- [vLLM 0.27.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.27.0)
- [vLLM 0.27.1 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.27.1)
- [vLLM 0.28.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)
- [Tagged v0.28.0 FlashInfer backend](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py)
- [SM120 NVFP4-KV tracking issue #49011](https://github.com/vllm-project/vllm/issues/49011)
- [Open native FA2 implementation PR #46329](https://github.com/vllm-project/vllm/pull/46329)

## What the port changes

The patch is
[`patches/vllm-0.27.1-flashinfer-0.6.16.post3/sm120-fa2-nvfp4-kv.patch`](patches/vllm-0.27.1-flashinfer-0.6.16.post3/sm120-fa2-nvfp4-kv.patch).

- `vllm/v1/attention/backends/flashinfer.py`
  - accepts NVFP4 KV on capability family 120;
  - selects native FA2 for prefill and decode instead of the SM100-only
    `trtllm-gen` path;
  - keeps Q and output in model dtype rather than the TRTLLM FP8 contract;
  - uses an explicit five-dimensional K/V cache layout so each side remains a
    contiguous `[packed data | scale factors]` view.
- `vllm/v1/worker/gpu/attn_utils.py` and
  `vllm/v1/worker/gpu_model_runner.py`
  - pass the cache dtype to the backend while preserving compatibility with
    backends that do not implement that optional argument.
- `flashinfer/data/include/flashinfer/attention/prefill.cuh`
  - reads vLLM's four-token-swizzled V scale factors in their existing layout
    and reverses the swizzle in the JIT kernel;
  - adds no side cache and therefore does not hide extra KV memory from vLLM's
    profiler.

FlashInfer 0.6.16.post3 already carries the explicit scale-factor stride
plumbing that the original patch had to add to older FlashInfer releases. The
installed vLLM wheel's KV writer is compiled, so this port deliberately adapts
the JIT FlashInfer reader instead of pretending a Python-only change can alter
the compiled writer.

FA2 is used for both phases on purpose. It preserves the multi-token MTP path
used by `{"method":"mtp","num_speculative_tokens":3}` and avoids relying on
the still-evolving SM12x XQA path.

## Apply it yourself

The script defaults to read-only inspection. Point it at the Python executable
inside the environment you intend to patch:

```bash
cd ~/vllm-nvfp4-kv-sm120

PYTHON=~/Odysseus/Diogenes/.venv/bin/python ./apply_patches.sh --check
PYTHON=~/Odysseus/Diogenes/.venv/bin/python ./apply_patches.sh --apply
```

Expected pre-apply result:

```text
state:       READY
All four files match the supported originals; --apply is safe to run.
```

After applying, rerunning either `--check` or `--apply` reports `APPLIED` and
does not touch the files again. Unknown versions, missing paths, changed source
hashes, and partial installs fail explicitly.

The first patched serve must compile the changed FlashInfer header. Before it,
locate the old JIT entries under:

```text
${FLASHINFER_WORKSPACE_BASE:-<your home>}/.cache/flashinfer/0.6.16/<architecture>/cached_ops
```

Move the matching `cached_ops` directory aside (rather than deleting it), then
start vLLM. FlashInfer will rebuild the required kernel and the moved directory
remains available for recovery.

Add one argument to the otherwise working command:

```bash
--kv-cache-dtype nvfp4
```

For example, the relevant tail remains:

```bash
--reasoning-parser qwen3 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
--kv-cache-dtype nvfp4
```

## Restore

```bash
PYTHON=~/Odysseus/Diogenes/.venv/bin/python ./apply_patches.sh --restore
```

Restore succeeds only when all four backups exist and still hash to the exact
0.27.1/0.6.16.post3 originals. Backups are retained after restoration.

## Scope and validation

This path targets standard MHA/GQA attention. It does not add NVFP4 KV support
to MLA, Mamba/SSM, attention sinks, or distributed-context-parallel attention.
The vLLM route is runtime-gated to SM120. Because the patched FlashInfer header
is shared JIT source, reserve this environment for the patched vLLM path rather
than unrelated direct FlashInfer FP4 callers.

The port has been statically checked by applying its unified diff to clean
copies of the exact installed files and compiling the three resulting Python
modules. No inference or GPU benchmark is claimed for this 0.27.1 port. The
capacity, quality, and throughput reports in [`docs/`](docs/) are the original
project's evidence on its older pinned stack; they explain the design but must
not be presented as fresh validation of this rebase.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The patch modifies
Apache-2.0 code from vLLM and FlashInfer.
