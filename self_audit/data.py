"""Separate labeled/unlabeled access. U masks are not listed or opened."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from .model import digest


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def label_subset(patient_ids, count, seed):
    if len(set(patient_ids)) != len(patient_ids) or not 2 <= count < len(patient_ids):
        raise ValueError("need unique TRAIN IDs and 2 <= label budget < TRAIN patients")
    return sorted(sorted(patient_ids, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())[:count])


def crop_native(x, labels, rng, size):
    """Shared image/label random crop; geometry-only padding validity for U.

    Labels are optional and never determine crop coordinates. No resizing.
    """
    h, w = x.shape[-2:]
    valid = torch.ones((len(x), h, w), dtype=torch.bool, device=x.device)
    if not size:
        return x, labels, valid, None
    ph, pw = max(0, size-h), max(0, size-w)
    padding = (pw//2, pw-pw//2, ph//2, ph-ph//2)
    if ph or pw:
        x = F.pad(x, padding, mode='replicate')
        valid = F.pad(valid, padding, value=False)
        if labels is not None:
            labels = F.pad(labels, padding, value=255)
    top = int(rng.integers(x.shape[-2]-size+1))
    left = int(rng.integers(x.shape[-1]-size+1))
    spatial = (..., slice(top, top+size), slice(left, left+size))
    info = {'original_hw': [h,w], 'pad_lrtb': list(padding), 'top': top, 'left': left, 'size': size}
    return x[spatial], labels[spatial] if labels is not None else None, valid[spatial], info


# Known duplicates from origin/main reports/automatic_pilot_pro4000/dataset_MnM_verified_preparation.json
KNOWN_QUARANTINED_MNM_IDS = frozenset({
    "A7G0P5", "A8C9H8", "C5Q2Y5", "C8J7L5", "C8O0P2", "E9L4N2", "K3R0Y7", "Q0Q1Y4",
})
KNOWN_EXCLUDED_ACDC_IDS = frozenset({
    "patient055", "patient076", "055", "076",
})


def _is_quarantined_patient(patient_id: str) -> bool:
    raw = patient_id.split(":", 1)[1] if ":" in patient_id else patient_id
    if raw.startswith("acdc_") or raw.startswith("acdc-"):
        raw = raw[5:]
    elif raw.startswith("mnm_") or raw.startswith("mnm-") or raw.startswith("MnM_"):
        raw = raw[4:]
    num_suffix = raw.replace("patient", "")
    return raw in KNOWN_QUARANTINED_MNM_IDS or raw in KNOWN_EXCLUDED_ACDC_IDS or num_suffix in KNOWN_EXCLUDED_ACDC_IDS


def _validate_patient_id(patient_id: str) -> None:
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError(f"invalid patient ID: {patient_id}")
    pid = patient_id.strip()
    for forbidden in ("..", "/", "\\", "\0", "\n", "\r", "\t"):
        if forbidden in pid:
            raise ValueError(f"potential path traversal or invalid characters in patient ID: '{patient_id}'")


def _validate_relative_path(path_str: str) -> None:
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError(f"invalid path string: {path_str}")
    p = Path(path_str)
    for part in p.parts:
        if part == "..":
            raise ValueError(f"path traversal detected in path: '{path_str}'")


class StepIndices(tuple):
    """Step index tuple with named/attribute access for multi-domain coordination."""
    def __new__(cls, li, ai, ui, domain_s=None, domain_u=None):
        return super().__new__(cls, (li, ai, ui))

    def __init__(self, li, ai, ui, domain_s=None, domain_u=None):
        self.supervised = li
        self.audit = ai
        self.unlabeled = ui
        self.domain_supervised = domain_s
        self.domain_unlabeled = domain_u

    def __getitem__(self, item):
        if isinstance(item, str):
            mapping = {
                "supervised": self[0], "audit": self[1], "unlabeled": self[2],
                "li": self[0], "ai": self[1], "ui": self[2],
                "domain_supervised": self.domain_supervised,
                "domain_unlabeled": self.domain_unlabeled,
                "domain_s": self.domain_supervised,
                "domain_u": self.domain_unlabeled,
            }
            if item in mapping:
                return mapping[item]
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


class TrainingData:
    def __init__(self, manifest, expected_sha256, *, synthetic=False):
        self.path = Path(manifest).resolve()
        if digest(self.path) != expected_sha256:
            raise ValueError("training manifest hash mismatch")
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m.get("schema") != "self_audit.data.v2" or m.get("role") != "training":
            raise ValueError("not a v2 training manifest")
        if bool(m.get("synthetic", False)) != synthetic:
            raise ValueError("synthetic/patient identity mismatch")
        self.labeled, self.unlabeled = m["labeled"], m["unlabeled"]
        self.lids, self.uids = [r["id"] for r in self.labeled], [r["id"] for r in self.unlabeled]
        for pid in self.lids + self.uids + m.get("evaluation_ids", []):
            _validate_patient_id(pid)
        if len(self.lids) < 2 or not self.uids or len(set(self.lids + self.uids)) != len(self.lids + self.uids):
            raise ValueError("nonempty disjoint patient sets and at least two labeled patients required")
        if set(self.lids + self.uids) & set(m.get("evaluation_ids", [])):
            raise ValueError("TRAIN/evaluation patient overlap")
        if m.get("label_budget") != len(self.lids):
            raise ValueError("audit labels must be counted in the label budget")
        for r in self.unlabeled:
            _validate_relative_path(r.get("image", ""))
            if set(r) != {"id", "image", "image_sha256"}:
                raise ValueError("unlabeled record must contain image only, never reference paths")
        for r in self.labeled:
            _validate_relative_path(r.get("image", ""))
            if set(r) != {"id", "image", "image_sha256", "masks"} or not r["masks"]:
                raise ValueError("invalid labeled record")
            for minfo in r["masks"].values():
                _validate_relative_path(minfo.get("path", ""))

        # Strict shared metadata validation BEFORE array / mask hash IO
        has_shared_metadata = any(k in m for k in ("shared", "domains", "domain_budgets", "domain_patients", "domain_map"))
        if has_shared_metadata:
            if m.get("shared") is not True:
                raise ValueError("shared manifest must declare 'shared': true")
            declared_domains = m.get("domains")
            if declared_domains != ["acdc", "mnm"]:
                raise ValueError("manifest domains must be exactly ['acdc', 'mnm'] with canonical lowercase names and no aliases")

            # Budget checks
            budget = m.get("label_budget")
            if type(budget) is not int or isinstance(budget, bool):
                raise ValueError(f"label_budget must be int, got {type(budget).__name__}")
            db = m.get("domain_budgets")
            if not isinstance(db, dict) or set(db.keys()) != {"acdc", "mnm"}:
                raise ValueError("domain_budgets must be dict with exact keys {'acdc', 'mnm'}")
            for k in ("acdc", "mnm"):
                if type(db[k]) is not int or isinstance(db[k], bool):
                    raise ValueError(f"domain_budgets[{k}] must be int, got {type(db[k]).__name__}")
            if not synthetic:
                if budget != 8 or db["acdc"] != 4 or db["mnm"] != 4:
                    raise ValueError(f"real patient shared budget must be exactly 4+4=8, reject any other budget (got {db}, total={budget})")
            else:
                if sum(db.values()) != budget:
                    raise ValueError("sum of domain budgets must equal label_budget")

            # Domain patients coverage
            dp = m.get("domain_patients")
            if not isinstance(dp, dict) or set(dp.keys()) != {"acdc", "mnm"}:
                raise ValueError("manifest must declare 'domain_patients' with exact keys {'acdc', 'mnm'}")
            dp_acdc = set(dp["acdc"])
            dp_mnm = set(dp["mnm"])
            if dp_acdc & dp_mnm:
                raise ValueError("domain_patients sets must be disjoint")
            if dp_acdc | dp_mnm != set(self.lids + self.uids):
                raise ValueError("domain_patients must cover all labeled and unlabeled IDs completely")

            # Domain map completeness & consistency
            dmap = m.get("domain_map")
            if not isinstance(dmap, dict) or set(dmap.keys()) != set(self.lids + self.uids):
                raise ValueError("manifest must declare 'domain_map' covering every train patient ID")
            for pid in self.lids + self.uids:
                dom = dmap[pid]
                if dom not in ("acdc", "mnm"):
                    raise ValueError(f"domain_map[{pid}]='{dom}' is not a canonical domain")
                exp_dom = "acdc" if pid in dp_acdc else "mnm"
                if dom != exp_dom:
                    raise ValueError(f"domain_map[{pid}]='{dom}' conflicts with domain_patients '{exp_dom}'")

            # Namespace consistency
            for pid in self.lids + self.uids:
                if ":" not in pid:
                    raise ValueError(f"shared patient ID '{pid}' lacks canonical 'domain:id' namespace")
                p_dom, _ = pid.split(":", 1)
                if p_dom not in ("acdc", "mnm"):
                    raise ValueError(f"shared patient ID '{pid}' has invalid domain prefix '{p_dom}'")
                if p_dom != dmap[pid]:
                    raise ValueError(f"patient ID '{pid}' namespace '{p_dom}' conflicts with domain_map '{dmap[pid]}'")

            # Quarantines
            for pid in self.lids + self.uids + m.get("evaluation_ids", []):
                if _is_quarantined_patient(pid):
                    raise ValueError(f"quarantined patient '{pid}' found in manifest")

            # Evaluation IDs must be namespaced
            for eid in m.get("evaluation_ids", []):
                if ":" not in eid or eid.split(":", 1)[0] not in ("acdc", "mnm"):
                    raise ValueError(f"evaluation ID '{eid}' must be namespaced with 'acdc:' or 'mnm:'")

            # Per-domain labeled counts match budgets
            acdc_l_count = sum(1 for pid in self.lids if dmap[pid] == "acdc")
            mnm_l_count = sum(1 for pid in self.lids if dmap[pid] == "mnm")
            if acdc_l_count != db["acdc"] or mnm_l_count != db["mnm"]:
                raise ValueError(f"domain labeled counts ({acdc_l_count}, {mnm_l_count}) != declared budgets ({db['acdc']}, {db['mnm']})")

            # Shared metadata: preprocessing, candidate_pool, mapping_provenance, ssl_initialization_requirement
            from .shared_data import CANONICAL_PREPROCESSING, SSL_INITIALIZATION_REQUIREMENT
            prep = m.get("preprocessing")
            if not isinstance(prep, dict) or prep != CANONICAL_PREPROCESSING:
                raise ValueError("manifest missing or invalid canonical preprocessing metadata")

            cp = m.get("candidate_pool")
            if not isinstance(cp, dict) or "mode" not in cp or "restricted_candidate_pool" not in cp:
                raise ValueError("manifest must declare 'candidate_pool' with 'mode' and 'restricted_candidate_pool'")
            if cp.get("mode") == "restricted_manifest_labeled" and not cp.get("restricted_candidate_pool"):
                raise ValueError("restricted manifest candidate pool must declare restricted_candidate_pool=True")

            mp = m.get("mapping_provenance")
            if not isinstance(mp, dict) or "acdc" not in mp or "mnm" not in mp:
                raise ValueError("manifest must declare 'mapping_provenance' containing 'acdc' and 'mnm'")
            for dom in ("acdc", "mnm"):
                desc = str(mp[dom]).strip().lower()
                for forbidden in ("double_remap", "unknown", "not canonical", "non-canonical", "not_canonical", "uncanonical"):
                    if forbidden in desc:
                        raise ValueError(f"rejected invalid mapping provenance for {dom}: {desc}")
                is_canonical = "canonical" in desc or "identity" in desc
                is_raw_mnm = "raw_mnm" in desc
                if dom == "acdc" and not is_canonical:
                    raise ValueError(f"ACDC mapping provenance must declare canonical, got: {desc}")
                if dom == "mnm" and not (is_canonical or is_raw_mnm):
                    raise ValueError(f"MnM mapping provenance must declare canonical or raw_mnm, got: {desc}")

            if m.get("ssl_initialization_requirement") != SSL_INITIALIZATION_REQUIREMENT:
                raise ValueError("manifest missing or invalid ssl_initialization_requirement")

            self.is_shared = True
            self.domains = ["acdc", "mnm"]
            self._domain_map = dict(dmap)
            self.domain_indices = {
                "labeled": {d: [i for i, r in enumerate(self.labeled) if dmap[r["id"]] == d] for d in self.domains},
                "unlabeled": {d: [i for i, r in enumerate(self.unlabeled) if dmap[r["id"]] == d] for d in self.domains},
            }
        else:
            self.is_shared = False
            self.domains = ["default"]
            self._domain_map = {}
            self.domain_indices = {
                "labeled": {"default": list(range(len(self.labeled)))},
                "unlabeled": {"default": list(range(len(self.unlabeled)))},
            }

        self.cache, self.label_reads = {}, set()
        hashes = [r["image_sha256"] for r in self.labeled + self.unlabeled]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate TRAIN image content across patient IDs")
        for r in self.labeled + self.unlabeled:
            a = self.array(r["image"], r["image_sha256"])
            if a.ndim != 4 or min(a.shape) < 1 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError("image must be finite normalized TZYX")
            if "masks" in r:
                for t, mask in r["masks"].items():
                    if not 0 <= int(t) < a.shape[0] or set(mask) != {"path", "sha256"}:
                        raise ValueError("invalid labeled frame")
                    # Hash access is allowed only for the budgeted labeled pool.
                    if digest(self.resolve(mask["path"])) != mask["sha256"]:
                        raise ValueError("labeled mask hash mismatch")

    def resolve(self, path):
        p = Path(path)
        return p if p.is_absolute() else self.path.parent / p

    def array(self, path, sha):
        resolved = self.resolve(path)
        key = str(resolved)
        if key not in self.cache:
            if digest(resolved) != sha:
                raise ValueError("data payload hash mismatch")
            self.cache[key] = np.load(resolved, mmap_mode="r", allow_pickle=False)
        return self.cache[key]

    @staticmethod
    def context(cine, t, z):
        inds = [max(0, z - 1), z, min(cine.shape[1] - 1, z + 1)]
        return torch.from_numpy(np.array(cine[t, inds], dtype=np.float32, copy=True))[None]

    def sample_labeled(self, index, rng, *, foreground_probability=.5):
        r = self.labeled[index]
        cine = self.array(r["image"], r["image_sha256"])
        frame = sorted(r["masks"], key=int)[int(rng.integers(len(r["masks"])))]
        rec = r["masks"][frame]
        mask = self.array(rec["path"], rec["sha256"])
        if not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == bool):
            raise ValueError("mask must have integer dtype")
        if mask.shape != cine.shape[1:] or not np.isin(mask, [0, 1, 2, 3, 255]).all():
            raise ValueError("mask shape/ontology mismatch")
        self.label_reads.add(r["id"])
        foreground = np.flatnonzero(((mask > 0) & (mask < 4)).reshape(len(mask), -1).any(1))
        z = int(rng.choice(foreground)) if len(foreground) and rng.random() < foreground_probability else int(rng.integers(len(mask)))
        x = self.context(cine, int(frame), z)
        y = torch.from_numpy(np.array(mask[z], dtype=np.int64, copy=True))[None]
        return x, y, {"patient": r["id"], "frame": int(frame), "slice": z}

    def sample_unlabeled(self, index, rng):
        r = self.unlabeled[index]
        cine = self.array(r["image"], r["image_sha256"])
        t, z = int(rng.integers(len(cine))), int(rng.integers(cine.shape[1]))
        return self.context(cine, t, z), {"patient": r["id"], "frame": t, "slice": z}

    def sample_batch(self, index, rng, batch_size, *, labeled, crop_size=0, return_valid=False):
        """Slices from one patient: optional fixed crop; no resizing or U mask access.

        A transaction uses different supervised/audit patients. For batch>1 the
        audit measure gives equal mass to sampled slices of that audit patient.
        This is a declared batch protocol, not the single-slice experiment.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("positive integer batch_size required")
        sampler = self.sample_labeled if labeled else self.sample_unlabeled
        samples, validity = [], []
        for _ in range(batch_size):
            item = sampler(index, rng)
            x, y, valid, crop = crop_native(item[0], item[1] if labeled else None, rng, crop_size)
            meta = dict(item[-1])
            if crop is not None:
                meta['crop'] = crop
            samples.append((x,y,meta) if labeled else (x,meta))
            validity.append(valid)
        images = torch.cat([s[0] for s in samples])
        metadata = dict(samples[0][-1])
        if batch_size > 1:
            metadata["samples"] = [s[-1] for s in samples]
        if labeled:
            result = (images, torch.cat([s[1] for s in samples]), metadata)
            # The mask comes only from image geometry, never anatomy labels.
            return (*result, torch.cat(validity)) if return_valid else result
        if return_valid:
            return images, metadata, torch.cat(validity)
        return images, metadata

    def domain_of(self, patient_id):
        if self._domain_map and patient_id in self._domain_map:
            return self._domain_map[patient_id]
        if self.is_shared and ":" in patient_id:
            dom = patient_id.split(":", 1)[0].lower()
            if dom in self.domains:
                return dom
        return "default"

    def choose_patient(self, step, rng=None, *, labeled=True, domain=None):
        """Select patient index with 50/50 domain balancing then round-robin within domain.

        If domain is None and dataset is shared, alternates domain per step (step % len(domains)),
        and indexes round-robin within that domain (step // len(domains)).
        If not shared, preserves exact baseline round-robin (step % len(pool)).
        """
        pool = self.labeled if labeled else self.unlabeled
        role = "labeled" if labeled else "unlabeled"
        if not self.is_shared or domain == "all":
            return step % len(pool)

        if domain is None:
            target_domain = self.domains[step % len(self.domains)]
            domain_step = step // len(self.domains)
        else:
            if domain not in self.domains:
                raise ValueError(f"unknown domain: {domain}; available: {self.domains}")
            target_domain = domain
            domain_step = step

        indices = self.domain_indices[role][target_domain]
        if not indices:
            raise ValueError(f"no {role} patients in domain {target_domain}")
        return indices[domain_step % len(indices)]

    def sample_indices(self, step, rng):
        """Return plain tuple (li, ai, ui) adhering to 50/50 domain balancing.

        When not shared:
            li = step % len(self.lids)
            ui = step % len(self.uids)
            ai = (li + 1 + int(rng.integers(len(self.lids) - 1))) % len(self.lids)
        When shared:
            li and ui alternate domains 50/50 per step and round-robin within domain.
            ai is chosen from labeled pool excluding li, guaranteeing disjoint patient.
        """
        if not self.is_shared:
            li = step % len(self.lids)
            ui = step % len(self.uids)
            ai = (li + 1 + int(rng.integers(len(self.lids) - 1))) % len(self.lids)
            return (li, ai, ui)

        li = self.choose_patient(step, rng, labeled=True)
        ui = self.choose_patient(step, rng, labeled=False)

        sup_patient = self.labeled[li]["id"]
        sup_domain = self.domain_of(sup_patient)
        domain_cand = [idx for idx in self.domain_indices["labeled"][sup_domain] if self.labeled[idx]["id"] != sup_patient]
        if domain_cand:
            ai = domain_cand[int(rng.integers(len(domain_cand)))]
        else:
            all_cand = [idx for idx in range(len(self.labeled)) if self.labeled[idx]["id"] != sup_patient]
            ai = all_cand[int(rng.integers(len(all_cand)))]

        if self.labeled[li]["id"] == self.labeled[ai]["id"]:
            raise ValueError("audit patient must be disjoint from supervised patient")
        return (li, ai, ui)

    def domain_coverage(self, seen_patient_ids):
        """Return per-domain telemetry of observed patients."""
        seen = set(seen_patient_ids)
        telemetry = {}
        for dom in self.domains:
            l_pids = {self.labeled[i]["id"] for i in self.domain_indices["labeled"][dom]}
            u_pids = {self.unlabeled[i]["id"] for i in self.domain_indices["unlabeled"][dom]}
            all_pids = l_pids | u_pids
            seen_dom_l = l_pids & seen
            seen_dom_u = u_pids & seen
            seen_dom = all_pids & seen
            telemetry[dom] = {
                "labeled_total": len(l_pids),
                "labeled_seen": len(seen_dom_l),
                "labeled_coverage": len(seen_dom_l) / max(len(l_pids), 1),
                "unlabeled_total": len(u_pids),
                "unlabeled_seen": len(seen_dom_u),
                "unlabeled_coverage": len(seen_dom_u) / max(len(u_pids), 1),
                "total_patients": len(all_pids),
                "total_seen": len(seen_dom),
                "total_coverage": len(seen_dom) / max(len(all_pids), 1),
            }
        return telemetry


def synthetic_fixture(folder, seed=11, n_labeled=3, n_unlabeled=3, size=32):
    """Manufactured objects only. Never evidence of cardiac accuracy."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    labeled, unlabeled = [], []
    for j in range(n_labeled + n_unlabeled):
        cy, cx = size * (.5 + rng.uniform(-.08, .08)), size * (.5 + rng.uniform(-.08, .08))
        radius = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        y = np.zeros((size, size), np.uint8)
        y[radius < size * .27] = 2
        y[radius < size * .18] = 3
        y[(yy - cy)**2 + (xx - (cx - size * .29))**2 < (size * .12)**2] = 1
        image = np.clip(np.choose(y, [.1, .8, .45, .9]) + rng.normal(0, .04, y.shape), 0, 1).astype(np.float32)[None, None]
        pid = f"synthetic_{j:03d}"; ip = folder / f"{pid}.npy"; np.save(ip, image)
        r = {"id": pid, "image": ip.name, "image_sha256": digest(ip)}
        if j < n_labeled:
            mp = folder / f"{pid}_mask.npy"; np.save(mp, y[None])
            r["masks"] = {"0": {"path": mp.name, "sha256": digest(mp)}}; labeled.append(r)
        else:
            unlabeled.append(r)
    manifest = folder / "train.json"
    write_json(manifest, {"schema": "self_audit.data.v2", "role": "training", "synthetic": True,
                         "label_budget": n_labeled, "labeled": labeled, "unlabeled": unlabeled, "evaluation_ids": []})
    return manifest, digest(manifest)
