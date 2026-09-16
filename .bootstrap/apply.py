"""One-shot guarded source edit; removed from branch after CI materialization."""
from pathlib import Path
import ast
import hashlib

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'src/self_audit_maskfree/candidate_execution.py': '011d7ff70c01c18b3503ced49b669c346421bdc9',
    'src/self_audit_maskfree/config.py': '84b909022a53fb1184e87e0b910096aa9aab5896',
    'src/self_audit_maskfree/trainer.py': '8220b6eabf07e424be683083c799c692fe2077d2',
    'scripts/profile_maskfree.py': '5ab513c84db336940c0e8141401ee23eee7e4e68',
    'scripts/benchmark_maskfree_rental.py': '9ce2c32765b98f95f5354cfdfb722b3e055aff73',
    'scripts/run_maskfree_full.sh': 'a95e9cdf601b7da6b66000694627cdfb87ba557d',
    'scripts/train_maskfree.py': '92dfa492fa97e7941153abdcf4517a17d7338bc3',
}
texts = {}
for name, expected in EXPECTED.items():
    data = (ROOT / name).read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Unreviewed source changed: {name}: {actual} != {expected}')
    texts[name] = data.decode()


def replace(name, old, new, count=1):
    s = texts[name]
    if s.count(old) != count:
        raise RuntimeError(f'{name}: expected {count} exact matches, got {s.count(old)} for {old!r}')
    texts[name] = s.replace(old, new)


cfg = 'src/self_audit_maskfree/config.py'
replace(cfg, '"prefetch_batches", "prefetch_max_bytes", "candidate_workers", "candidate_worker_threads",',
        '"prefetch_batches", "prefetch_max_bytes", "candidate_workers", "candidate_worker_threads",\n    "candidate_chunk_size",')
replace(cfg, '    candidate_worker_threads: int = 1\n', '    candidate_worker_threads: int = 1\n    candidate_chunk_size: int = 8\n')
replace(cfg, '"prefetch_max_bytes", "candidate_workers", "candidate_worker_threads"):',
        '"prefetch_max_bytes", "candidate_workers", "candidate_worker_threads", "candidate_chunk_size"):')
replace(cfg, 'if self.candidate_workers not in (0, 2, 4):', 'if self.candidate_workers not in (0, 2, 4, 8):')
replace(cfg, '            raise ConfigError("candidate_workers must be 0, 2 or 4")',
        '            raise ConfigError("candidate_workers must be 0, 2, 4 or 8")\n        if not 1 <= self.candidate_chunk_size <= 8:\n            raise ConfigError("candidate_chunk_size must be between 1 and 8")')

trainer = 'src/self_audit_maskfree/trainer.py'
replace(trainer, 'if not self._bulk_audit_enabled() or config.batch_size > 8:', 'if not self._bulk_audit_enabled():')
replace(trainer, 'candidate workers require canonical audit components and batch <= 8',
        'candidate workers require canonical audit components')
replace(trainer, 'workers=config.candidate_workers, worker_threads=config.candidate_worker_threads)',
        'workers=config.candidate_workers, worker_threads=config.candidate_worker_threads,\n                chunk_size=config.candidate_chunk_size)')
replace(trainer, '        self._candidate_executor = None\n\n    def _close_candidate_executor',
        '        self._candidate_executor = None\n        self._candidate_stats: dict[str, Any] = {}\n\n    def _close_candidate_executor')
replace(trainer, '        if self._candidate_executor is not None:\n            self._candidate_executor.close()',
        '        if self._candidate_executor is not None:\n            self._candidate_stats = self._candidate_executor.stats()\n            self._candidate_executor.close()')
replace(trainer, '            "prefetch": dict(self._prefetch_stats),',
        '            "prefetch": dict(self._prefetch_stats),\n            "candidate_execution": (self._candidate_executor.stats() if self._candidate_executor is not None\n                                    else dict(self._candidate_stats)),')

shell = 'scripts/run_maskfree_full.sh'
replace(shell, 'CANDIDATE_WORKERS=0  0, 2 or 4;', 'CANDIDATE_WORKERS=0  0, 2, 4 or 8;')
replace(shell, '  CANDIDATE_WORKER_THREADS=1\n',
        '  CANDIDATE_WORKER_THREADS=1\n  CANDIDATE_CHUNK_SIZE=8  1..8 in-flight units, independent of physical batch.\n')
train = 'scripts/train_maskfree.py'
replace(train, 'f"cache_bytes={config.data_cache_bytes}',
        'f"candidate_chunk={config.candidate_chunk_size} cache_bytes={config.data_cache_bytes}')

profile = 'scripts/profile_maskfree.py'
replace(profile, 'IMAGE_SIZE_CHOICES = (128, 224)', 'IMAGE_SIZE_CHOICES = (128, 224)\nBATCH_SIZE_CHOICES = (8, 16, 32)')
replace(profile, 'def _scientific_requirements(image_size: int) -> dict[str, Any]:',
        'def _scientific_requirements(image_size: int, batch_size: int = 8) -> dict[str, Any]:')
replace(profile, '    requirements["image_size"] = int(image_size)\n    return requirements',
        '    if isinstance(batch_size, bool) or batch_size not in BATCH_SIZE_CHOICES:\n        raise ValueError(f"batch-size must be one of {BATCH_SIZE_CHOICES}")\n    requirements["image_size"] = int(image_size)\n    requirements["batch_size"] = int(batch_size)\n    return requirements')
replace(profile, '    requested_image_size = _selected_image_size(args)\n',
        '    requested_image_size = _selected_image_size(args)\n    requested_batch_size = getattr(args, "batch_size", 8)\n')
replace(profile, '            batch_size=8,\n', '            batch_size=requested_batch_size,\n')
replace(profile, '    scientific_requirements = _scientific_requirements(requested_image_size)',
        '    scientific_requirements = _scientific_requirements(requested_image_size, requested_batch_size)')
replace(profile, '"candidate_workers", "candidate_worker_threads"):',
        '"candidate_workers", "candidate_worker_threads", "candidate_chunk_size"):', count=2)
replace(profile, '    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)',
        '    parser.add_argument("--batch-size", type=int, choices=BATCH_SIZE_CHOICES, default=8,\n                        help="explicit profile contract; must match the supplied YAML (default 8)")\n    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)')
replace(profile, 'if any(item != 32 for item in fit_many_items):',
        'if any(item != config.batch_size * PROFILE_BUDGET_REQUIREMENTS["bank_size"] for item in fit_many_items):')
replace(profile, '"canonical bulk fit_many item count mismatch: expected 32 per measured batch"',
        'f"canonical bulk fit_many item count mismatch: expected {config.batch_size * PROFILE_BUDGET_REQUIREMENTS[\'bank_size\']} per measured batch"')
replace(profile, 'if args.warmup_batches + args.measured_batches >= PARTIAL_EPOCH_MAX_BATCHES:',
        'if args.warmup_batches + args.measured_batches >= min(PARTIAL_EPOCH_MAX_BATCHES,\n            SYNTHETIC_DEPTH // getattr(args, "batch_size", 8) if args.synthetic else PARTIAL_EPOCH_MAX_BATCHES):', count=3)
replace(profile, '            "image_size_contract": preparation.get("image_size_contract"),',
        '            "image_size_contract": preparation.get("image_size_contract"),\n            "physical_batch": config.batch_size,\n            "effective_batch": config.effective_batch,')

bench = 'scripts/benchmark_maskfree_rental.py'
replace(bench, '    p.add_argument("--candidate-workers", type=int, choices=(0, 2, 4), default=0)',
        '    p.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=8)\n    p.add_argument("--candidate-workers", type=int, choices=(0, 2, 4, 8), default=0)\n    p.add_argument("--candidate-chunk-size", type=int, choices=range(1, 9), default=8)\n    p.add_argument("--prefetch-max-bytes", type=int, default=32 * 1024 * 1024)')
replace(bench, 'run_id=f"rental-{dataset}-gate", image_size=224, batch_size=8,',
        'run_id=f"rental-{dataset}-gate", image_size=224, batch_size=args.batch_size,')
replace(bench, '            candidate_workers=args.candidate_workers)',
        '            candidate_workers=args.candidate_workers, candidate_chunk_size=args.candidate_chunk_size,\n            prefetch_max_bytes=args.prefetch_max_bytes)')
replace(bench, '"--audit-device", "cpu", "--image-size", "224", "--warmup-batches",',
        '"--audit-device", "cpu", "--image-size", "224", "--batch-size", str(args.batch_size), "--warmup-batches",')
replace(bench, 'if str(app.get("gpu_uuid", "")) == selected_uuid and app.get("pid") != os.getpid()',
        'if str(app.get("gpu_uuid", "")).removeprefix("GPU-").lower() == selected_uuid.removeprefix("GPU-").lower() and app.get("pid") != os.getpid()')

texts['src/self_audit_maskfree/candidate_execution.py'] = (ROOT / '.bootstrap/candidate_execution.py').read_text()
for name, content in texts.items():
    if name.endswith('.py'):
        ast.parse(content, filename=name)
for name, content in texts.items():
    (ROOT / name).write_text(content)
    print('updated', name)
