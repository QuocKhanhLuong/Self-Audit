# PG materialized-manifest relocation

## Contract

`open_pg_data` now supports one opt-in real-data relocation contract. Keep the
historical `data.source_k8_sha256` value at
`60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433`; it names
the unavailable original lock and is not the hash of a reconstructed lock.
The preparation receipt pins the new derived K8 lock separately.

The PG base config must add exactly this optional field under `data` after
preparation, alongside all three new `data.manifests` references:

```json
"materialization": {
  "receipt": "/root/dataset/prepared/ACDC/pg-k8-20261116/PREPARATION_RECEIPT.json",
  "sha256": "<SHA256 of the exact receipt bytes>"
}
```

Each `data.manifests.{train,val,test}` remains exactly `{ "path": ..., "sha256": ... }`.
The validator binds each pair to the READY receipt, checks the receipt's split
against committed FUSE/K8 evidence, checks exact L8 and derived-lock identities,
and verifies raw image/mask source paths against the verified download and
Drive inventories, including original mask SHA identities. Raw CINE identities
live under `relocation_lineage.source_images`, which preserves FUSE's exact
record key contracts. The validator also checks each manifest's per-patient
geometry audit. There is no hash bypass field or generic relocation mode;
synthetic configs cannot use this path.

The original K8 JSON was unavailable on this fresh server. The code does not
claim to have it or fabricate its bytes. W1 must write a new derived lock from
the byte-verified raw data and the exact committed L8 IDs, with its own SHA in
the receipt and the TRAIN `labeled_contract`.

Split membership is checked against the committed TRAIN split file
`splits/acdc_patient_split_seed42.json` (SHA256
`bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a`), the
FUSE VAL receipt (SHA256
`70f1d9b2ca1c6d93e2ab42ff69344b9721cd4fe89ed4170cf6f72474aa3a2816`), the FUSE
TEST transaction (SHA256
`0f196a32e8d21bc6541907b954c44a4fc59c2f0395a5b87dd7f32ce1f7b36359`), the
K8 L8 receipt (SHA256
`7ba19db0a896ce1169680e62dad1548a970af37fdf5c374cfb23ea40a4087db5`), and the
duplicate exclusion report (SHA256
`5ff62527c51541dc514cdeb5ecc00738c8112bd4fcdf8b419aeaebaa1e304017`).

## Coordinator sequence

After W1 reports the READY output, pin its exact bytes in the config:

```sh
sha256sum \
  /root/dataset/prepared/ACDC/pg-k8-20261116/train.json \
  /root/dataset/prepared/ACDC/pg-k8-20261116/val.json \
  /root/dataset/prepared/ACDC/pg-k8-20261116/test.json \
  /root/dataset/prepared/ACDC/pg-k8-20261116/PREPARATION_RECEIPT.json
```

Set the three manifest references to those absolute paths and printed hashes,
then set `data.materialization` to the receipt path and its printed hash. Keep
the existing split lock and historical K8 SHA fields unchanged. The coordinator
can materialize those fields after W1 completes with:

```sh
python - <<'PY'
import hashlib, json
from pathlib import Path

out = Path("/root/dataset/prepared/ACDC/pg-k8-20261116")
config_path = Path("configs/candidate_pg/pg_k8_base.json")
config = json.loads(config_path.read_text())
def pin(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
config["data"]["manifests"] = {
    "train": pin(out / "train.json"),
    "val": pin(out / "val.json"),
    "test": pin(out / "test.json"),
}
receipt = pin(out / "PREPARATION_RECEIPT.json")
config["data"]["materialization"] = {"receipt": receipt["path"], "sha256": receipt["sha256"]}
config_path.write_text(json.dumps(config, indent=2) + "\n")
PY
```

Run the focused software check with:

```sh
python -m pytest -q tests/test_candidate_pg_manifest_relocation.py tests/test_candidate_pg_config.py tests/test_candidate_pg_data.py
```

## Verification and limits

The focused local software checks pass (16 tests). They cover a valid pinned
relocation, changed split and L8 membership, changed lineage, strict config
fields, and raw inventory/hash checks. No real preparation receipt or derived
manifest is available in this workstation checkout, so server artifact
admission, patient data loading, training, and GPU behavior remain unverified.
