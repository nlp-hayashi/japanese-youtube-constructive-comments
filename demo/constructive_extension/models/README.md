# Model weights (not in Git)

We do **not** publish the full trained checkpoint in this repository. Download
`model.safetensors` separately (paper supplementary or other distribution channel)
and place files locally:

```
models/best/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors   # not included in Git
```

Or set `CONSTRUCTIVE_MODEL_DIR` when running the API (see extension README).

## Base model and license

The demo uses
[sbintuitions/modernbert-ja-130m](https://huggingface.co/sbintuitions/modernbert-ja-130m)
as the base architecture. Any tokenizer or config files you place under this
directory, and weights obtained from our release, are derived from or compatible
with that model.

**License for artifacts under `models/`:** see [LICENSE](./LICENSE) (MIT, SB
Intuitions). When you redistribute those files, keep that copyright and license
notice.
