# olive/models

OpenVLA checkpoints are **not** kept here permanently. Each `openvla-7b`
checkpoint is ~15 GB, so we fetch one on demand, use it, then release the disk.

Manage everything through `olive-model.sh`.

## Suites

| alias            | HF repo                                         | notes |
|------------------|-------------------------------------------------|-------|
| `base`           | `openvla/openvla-7b`                            | pretrained only, no LIBERO SFT |
| `spatial`        | `openvla/openvla-7b-finetuned-libero-spatial`   | LIBERO-Spatial SFT |
| `object`         | `openvla/openvla-7b-finetuned-libero-object`    | LIBERO-Object SFT |
| `goal`           | `openvla/openvla-7b-finetuned-libero-goal`      | LIBERO-Goal SFT |
| `long` (or `10`) | `openvla/openvla-7b-finetuned-libero-10`        | LIBERO-10 / LIBERO-Long SFT |

OLIVE uses the **frozen** LIBERO-SFT checkpoints (`spatial`/`object`/`goal`/`long`)
as the backbone; `base` is only for reference.

## Commands

```bash
# download a checkpoint into ./openvla-7b-finetuned-libero-10/
./olive-model.sh fetch long

# print its local path (downloads first if missing) — handy in scripts
MODEL=$(./olive-model.sh path long)

# delete the local folder and reclaim ~15 GB
./olive-model.sh release long

# what is currently on disk
./olive-model.sh list

# remove a stale copy from the shared HF hub cache (~/.cache/huggingface/hub)
./olive-model.sh purge-cache            # all suites
./olive-model.sh purge-cache spatial    # one suite

# fetch -> run a command with $OPENVLA_MODEL_PATH set -> auto-release afterwards
./olive-model.sh run long -- python eval_libero.py --suite libero_10
./olive-model.sh run --keep long -- python eval_libero.py   # keep the folder
```

`run` only auto-releases a checkpoint it downloaded itself; if the folder was
already present it is left alone. Pass `--keep` to always keep it.

## Environment

Use the `openvla` conda env:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate openvla
```

Loading these checkpoints needs `trust_remote_code=True` (custom
`modeling_prismatic.py` / `processing_prismatic.py` ship in each repo).
