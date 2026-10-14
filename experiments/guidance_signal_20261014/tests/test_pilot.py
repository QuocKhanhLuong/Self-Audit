"""Manufactured CPU contracts only; never original patient/4070 evidence."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments.guidance_signal_20261014 import pilot
from experiments.guidance_signal_20261014.runtime import load_runtime


def paired_tensors():
    q = torch.tensor([.15, .55, .15, .15]).view(1, 4, 1, 1).repeat(1, 1, 2, 2)
    z = torch.tensor([.05, .85, .05, .05]).view(1, 4, 1, 1).repeat(1, 1, 2, 2)
    weights = torch.tensor([[[.1, .2], [.3, 0.]]])
    valid = torch.tensor([[[True, True], [True, False]]])
    z[:, :, 1, 1] = float("nan")
    return q, z, weights, valid


def test_ce_forward_kl_gradients_and_detached_targets():
    q, z, weights, valid = paired_tensors()
    q.requires_grad_(); z.requires_grad_(); weights.requires_grad_()
    logits = torch.randn(1, 4, 2, 2, requires_grad=True)
    terms = pilot.semantic_terms(logits, q, z, weights, valid)
    for arm in ("Q", "Z", "full_Q"):
        ce, kl, entropy = [terms[arm][key] for key in ("ce", "kl", "entropy")]
        assert torch.allclose(ce - entropy, kl, atol=3e-7, rtol=0)
        ce_grad = torch.autograd.grad(ce, logits, retain_graph=True)[0]
        kl_grad = torch.autograd.grad(kl, logits, retain_graph=True)[0]
        assert torch.allclose(ce_grad, kl_grad, atol=1e-7, rtol=0)
        if arm != "full_Q":
            assert torch.equal(ce_grad[:, :, 1, 1], torch.zeros(1, 4))
    terms["Z"]["ce"].backward()
    assert q.grad is None and z.grad is None and weights.grad is None
    assert torch.isfinite(logits.grad).all()


def test_original_activation_and_full_denominator_even_when_z_changes_argmax():
    q, z, weights, valid = paired_tensors()
    logits = torch.zeros_like(q, requires_grad=True)
    z[valid[:, None].expand_as(z)] = .25
    weights[0, 1, 1] = 1e-8  # numerically negligible exclusion remains in denominator
    terms = pilot.semantic_terms(logits, q, z, weights, valid)
    assert terms["active"] is True  # Z is BG by tie, but original Q activates.
    assert torch.equal(terms["denominator"], weights.sum())
    assert terms["excluded_weight_mass"] > 0
    assert terms["full_Q"]["per_class_target_mass"].sum() > terms["Q"]["per_class_target_mass"].sum() - 1e-7
    q = torch.full_like(q, .1); q[:, 0] = .7
    z = torch.full_like(z, .05); z[:, 1] = .85
    inactive = pilot.semantic_terms(logits, q, z, weights, valid)
    assert inactive["active"] is False
    assert inactive["Z"]["ce"].item() == 0
    assert inactive["Q"]["ce"].item() == 0


def test_invalid_support_is_never_replaced_with_zero_or_background():
    q, z, weights, valid = paired_tensors()
    logits = torch.zeros_like(q, requires_grad=True)
    first = pilot.semantic_terms(logits, q, z, weights, valid)
    z[:, :, 1, 1] = float("inf")
    second = pilot.semantic_terms(logits, q, z, weights, valid)
    assert torch.equal(first["Z"]["ce"], second["Z"]["ce"])
    assert first["support_pixels"] == 3
    weights[0, 1, 1] = .01
    with pytest.raises(ValueError, match="substantial original weight"):
        pilot.semantic_terms(logits, q, z, weights, valid)
    z[:, :, 0, 0] = float("nan")
    weights[0, 1, 1] = 0
    with pytest.raises(ValueError, match="Z must be finite"):
        pilot.semantic_terms(logits, q, z, weights, valid)


def test_wrong_named_vote_amplification_is_not_accuracy_evidence():
    # Synthetic latent truth is LV; wrongly named evidence points at RV.
    # No latent truth is an input to the pilot's optimization or permissions.
    q = torch.tensor([.2, .4, .2, .2]).view(1, 4, 1, 1)
    z = torch.tensor([0., 1., 0., 0.]).view(1, 4, 1, 1)
    logits = torch.zeros_like(q, requires_grad=True)
    terms = pilot.semantic_terms(logits, q, z, torch.ones(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool))
    q_gradient = torch.autograd.grad(terms["Q"]["ce"], logits, retain_graph=True)[0]
    z_gradient = torch.autograd.grad(terms["Z"]["ce"], logits)[0]
    assert z_gradient[0, 1, 0, 0] < q_gradient[0, 1, 0, 0] < 0
    assert z_gradient[0, 3, 0, 0] > q_gradient[0, 3, 0, 0] > 0
    assert terms["Z"]["positive_RV_contrast_mass"] > terms["Q"]["positive_RV_contrast_mass"]


@pytest.fixture
def artifact_fixture(tmp_path, monkeypatch, request):
    """Actual executed student/io, with an explicitly synthetic signal loader.

    prepare's reconstruction and original-run checks have their own tests. This
    fixture substitutes only that dependency so a manufactured Q/Z bank can
    exercise all pilot steps without requiring raw data or teacher weights.
    """
    rt = load_runtime()
    inactive = getattr(request, "param", False)
    run_root = tmp_path / "original_run"
    run_root.mkdir()
    bank, signals, student = run_root / "bank", tmp_path / "signals", run_root / "student"
    for folder in (bank, signals, student):
        folder.mkdir()
    cfg = dict(feature_dim=4, min_probability=.7, min_margin=.1)
    splits = dict(train=["train_a", "train_b"], val=["val_c"], test=["test_d"])
    generator = np.random.default_rng(9)
    for pid in splits["train"]:
        image = generator.normal(size=(2, 1, 8, 8)).astype(np.float32)
        features = generator.normal(size=(2, 1, 2, 2, 4)).astype(np.float16)
        q = np.broadcast_to(np.array([.15, .55, .15, .15], np.float32), (2, 1, 2, 2, 4)).copy()
        z = np.broadcast_to(np.array([.05, .85, .05, .05], np.float32), q.shape).copy()
        valid = np.ones(q.shape[:-1], bool); valid[..., 1, 1] = False
        z[~valid] = np.nan
        q[~valid] = .25
        rel = np.full(valid.shape, .2, np.float32); rel[~valid] = 0
        labels = np.where(valid, 1, 255).astype(np.uint8)
        if inactive:
            labels[:] = 255
        np.savez(bank / (pid + ".npz"), image=image, features=features, q=q, r0=q,
                 reliability=rel, labels=labels)
        h = valid.astype(np.float32) * .5
        np.savez(signals / (pid + ".npz"), M=np.nan_to_num(z)*h[..., None], h=h, Z=z, valid=valid)
    manifest = dict(role="guidance_bank", split="train", patients=splits["train"], locked_splits=splits,
                    source_sha256=rt.source_sha256, config=cfg, teacher={"kind": "synthetic_fixture_only"},
                    patient_reports={pid: dict(class_pixels=[0, 0 if inactive else 6, 0, 0]) for pid in splits["train"]})
    rt.io.freeze(bank, manifest)
    bank_sha = rt.io.digest(bank / "FROZEN.json")
    torch.manual_seed(5)
    model = rt.student.GuidanceStudent(4, 8)
    ck = dict(model=model.state_dict(), feature_dim=4, width=8,
              status="REPRESENTATION_ONLY" if inactive else "PARTIAL_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED", semantic_updates=1,
              bank_manifest_sha256=bank_sha, teacher=manifest["teacher"], config=cfg,
              train_patients=splits["train"], locked_splits=splits,
              class_patients=[0, 0 if inactive else 2, 0, 0], allowed_classes=[False, not inactive, False, False])
    torch.save(ck, student / "last.pt")
    rt.io.freeze(student, dict(role="student", status=ck["status"], bank_manifest_sha256=bank_sha,
                              source_sha256=rt.source_sha256))
    recipe = dict(config=cfg, width=8, feature_dim=4, lr=3e-4, threads=1, batch_size=1,
                  source_sha256=rt.source_sha256)
    rt.io.write_json(run_root / "RECIPE.json", recipe)
    rt.io.freeze(signals, dict(role="guidance_signal_bank_v1", split="train", patients=splits["train"],
                              original_bank_manifest_sha256=bank_sha, synthetic_reproduction_pass=True,
                              recipe_sha256=rt.io.digest(run_root/"RECIPE.json"), source_sha256=rt.source_sha256,
                              config=cfg, teacher=manifest["teacher"]))
    calls = []

    def load_signal_bank(path, original):
        calls.append((str(path), str(original)))
        m = rt.io.verify(path, "guidance_signal_bank_v1")
        if (m["original_bank_manifest_sha256"] != rt.io.digest(original / "FROZEN.json")
                or m["synthetic_reproduction_pass"] is not True):
            raise ValueError("synthetic reproduction contract failed")
        return m

    def load_signal_patient(path, pid):
        with np.load(path / (pid + ".npz"), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    monkeypatch.setattr(pilot, "_dependencies", lambda: (rt, load_signal_bank, load_signal_patient))
    expected_sha = rt.io.digest(student / "last.pt")
    plan = pilot.create_plan(bank, signals, student / "last.pt", expected_sha, splits["train"], steps=2, seed=7)
    return dict(root=tmp_path, rt=rt, bank=bank, signals=signals, checkpoint=student/"last.pt",
                sha=expected_sha, plan=plan, calls=calls)


def test_two_step_cpu_paired_pilot_invariants_and_frozen_outputs(artifact_fixture):
    f = artifact_fixture
    output = f["root"] / "paired"
    original_hash = f["rt"].io.digest(f["checkpoint"])
    report = pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], f["plan"], output,
                             expected_plan_sha=pilot.json_hash(f["plan"]))
    assert report["steps_per_arm"] == 2
    assert not report["anatomical_claim"] and not report["export_or_evaluation_run"]
    assert report["determinism"]["torch_deterministic_algorithms"] is True
    assert report["determinism"]["deterministic_warn_only"] is False
    assert report["determinism"]["cudnn_deterministic"] is True
    assert report["determinism"]["cudnn_benchmark"] is False
    assert report["determinism"]["cublas_workspace_config"] == ":4096:8"
    assert f["rt"].io.digest(f["checkpoint"]) == original_hash
    assert len(f["calls"]) == 3  # plan validation, before run, after run
    traces = [f["rt"].io.read_json(output / arm / "TRACE.json") for arm in ("Q", "Z")]
    a, b = traces
    assert a["initial_model_state_sha256"] == b["initial_model_state_sha256"] == f["plan"]["model_state_sha256"]
    assert a["ordered_patient_tz_schedule_sha256"] == b["ordered_patient_tz_schedule_sha256"]
    assert a["fresh_optimizer_initial_state_entries"] == b["fresh_optimizer_initial_state_entries"] == 0
    assert a["final_model_state_sha256"] != b["final_model_state_sha256"]
    for qa, zb in zip(a["steps"], b["steps"]):
        for key in report["paired_invariants_checked"]:
            assert qa[key] == zb[key]
        for row in (qa, zb):
            assert "full_Q" in row["semantic"]
            assert "original_full_Q_total_loss" in row
            assert row["semantic"]["excluded_weight_mass"] == 0
    for arm, trace in zip(("Q", "Z"), traces):
        f["rt"].io.verify(output/arm, "private_technical_pilot")
        for moment in ("before", "after"):
            assert trace["fixed_train_batch_no_updates"][moment]["gradient_norms"]["weights_updated"] is False
        ck = torch.load(output/arm/"private_last.pt", weights_only=True)
        assert ck["status"] == "PRIVATE_TECHNICAL_PILOT_NOT_EXPORTABLE_NOT_ANATOMY_VALIDATED"
    f["rt"].io.verify(output, "paired_technical_pilot")
    with pytest.raises(FileExistsError, match="FRESH"):
        pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], f["plan"], output,
                        expected_plan_sha=pilot.json_hash(f["plan"]))


@pytest.mark.parametrize("steps", [0, 129, -1, True])
def test_step_bound_rejected_before_optimizer_or_output(artifact_fixture, monkeypatch, steps):
    f = artifact_fixture
    plan = copy.deepcopy(f["plan"]); plan["steps"] = steps
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer created"))
    output = f["root"] / "rejected"
    with pytest.raises(ValueError, match="integer|1..128"):
        pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], plan, output,
                        expected_plan_sha=pilot.json_hash(plan))
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["test_patient", "val_patient", "wrong_sha", "wrong_model", "policy", "epochs"])
def test_safety_rejects_nontrain_wrong_identity_or_policy(artifact_fixture, monkeypatch, mutation):
    f = artifact_fixture
    plan = copy.deepcopy(f["plan"])
    if mutation == "test_patient": plan["patient_ids"] = ["test_d"]
    if mutation == "val_patient": plan["patient_ids"] = ["val_c"]
    if mutation == "wrong_sha": plan["checkpoint_sha256"] = "0" * 64
    if mutation == "wrong_model": plan["model_state_sha256"] = "0" * 64
    if mutation == "policy": plan["policy"]["allowed_classes"] = [True] * 4
    if mutation == "epochs": plan["epochs"] = 1
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer created"))
    monkeypatch.setattr(f["rt"].io, "load_bank_patient", lambda *a: pytest.fail("patient arrays opened"))
    output = f["root"] / "rejected"
    with pytest.raises(ValueError):
        pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], plan, output,
                        expected_plan_sha=pilot.json_hash(plan))
    assert not output.exists()


def test_signal_validation_failure_stops_before_optimizer(artifact_fixture, monkeypatch):
    f = artifact_fixture
    rt, _, loader = pilot._dependencies()
    def fail(*args):
        raise ValueError("FAIL_REPRODUCTION")
    monkeypatch.setattr(pilot, "_dependencies", lambda: (rt, fail, loader))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer created"))
    with pytest.raises(ValueError, match="FAIL_REPRODUCTION"):
        pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], f["plan"], f["root"] / "reject",
                        expected_plan_sha=pilot.json_hash(f["plan"]))


def test_cli_plan_creation_has_no_optimizer_and_no_overwrite(artifact_fixture, monkeypatch):
    f = artifact_fixture
    target = f["root"] / "PLAN.json"
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer created"))
    argv = ["--bank", str(f["bank"]), "--signals", str(f["signals"]), "--checkpoint", str(f["checkpoint"]),
            "--expected-checkpoint-sha", f["sha"], "--plan", str(target), "--write-plan-only",
            "--patient-ids", "train_a", "train_b", "--steps", "2", "--seed", "7"]
    assert pilot.main(argv) == 0
    assert json.loads(target.read_text()) == f["plan"]
    with pytest.raises(FileExistsError, match="FRESH"):
        pilot.main(argv)
    with pytest.raises(SystemExit):
        pilot.main(argv + ["--epochs", "1"])


def test_schedule_is_reproducible_and_state_hash_is_order_independent():
    shapes = {"a": (3, 2), "b": (2, 1)}
    first = pilot.make_schedule(["a", "b"], shapes, 2, 2, 4)
    assert first == pilot.make_schedule(["a", "b"], shapes, 2, 2, 4)
    assert len(first) == 2
    assert pilot.state_hash({"a": torch.ones(1), "b": torch.zeros(2)}) == pilot.state_hash({"b": torch.zeros(2), "a": torch.ones(1)})


@pytest.mark.parametrize("total_weight", [1e-8, 1e-9, 1e-12])
def test_tiny_fully_excluded_objective_is_rejected(total_weight):
    with pytest.raises(ValueError, match="substantial original weight"):
        pilot.check_excluded_mass(torch.tensor([total_weight]), torch.tensor([False]))


@pytest.mark.parametrize("key,new_value", [("steps", 1), ("seed", 999), ("patient_ids", ["train_a"])])
def test_mutated_plan_rejected_against_independent_plan_hash_before_any_load(artifact_fixture, monkeypatch, key, new_value):
    f = artifact_fixture
    original_plan_sha = pilot.json_hash(f["plan"])
    changed = copy.deepcopy(f["plan"]); changed[key] = new_value
    monkeypatch.setattr(pilot, "_dependencies", lambda: pytest.fail("runtime loaded before plan identity check"))
    with pytest.raises(ValueError, match="plan SHA mismatch"):
        pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], changed, f["root"] / "reject",
                        expected_plan_sha=original_plan_sha)


def test_late_batch_excluded_mass_fails_before_first_model_or_update(artifact_fixture, monkeypatch):
    f = artifact_fixture
    original_loader = f["rt"].io.load_bank_patient
    def manufactured_late_exclusion(*args):
        data = original_loader(*args)
        data["q"][:] = [.15, .55, .15, .15]
        data["reliability"][:] = 0
        data["reliability"][0, 0, 0, 0] = 1
        data["reliability"][1, 0, 1, 1] = 1e-8
        return data
    monkeypatch.setattr(f["rt"].io, "load_bank_patient", manufactured_late_exclusion)
    monkeypatch.setattr(f["rt"].student, "GuidanceStudent", lambda *a: pytest.fail("model created before all batch guards"))
    output = f["root"] / "reject"
    with pytest.raises(ValueError, match="substantial original weight"):
        pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], f["plan"], output,
                        expected_plan_sha=pilot.json_hash(f["plan"]))
    assert not output.exists()


@pytest.mark.parametrize("artifact_fixture", [True], indirect=True)
def test_original_inactive_batch_preserves_decoder_even_under_adamw(artifact_fixture):
    f = artifact_fixture
    output = f["root"] / "inactive_pair"
    pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], f["plan"], output,
                    expected_plan_sha=pilot.json_hash(f["plan"]))
    initial = torch.load(f["checkpoint"], weights_only=True)["model"]
    final = []
    for arm in ("Q", "Z"):
        trace = f["rt"].io.read_json(output/arm/"TRACE.json")
        assert trace["active_fraction"] == 0
        state = torch.load(output/arm/"private_last.pt", weights_only=True)["model"]
        for key in initial:
            if key.startswith(("decode.", "head.")):
                assert torch.equal(initial[key], state[key])
        assert any(not torch.equal(initial[key], state[key]) for key in initial if key.startswith("enc"))
        final.append(pilot.state_hash(state))
    assert final[0] == final[1]


def test_actual_prepare_loader_to_pilot_integration_on_manufactured_bank(tmp_path, monkeypatch):
    """Real preparation/loader paths, but wholly manufactured CPU artifacts.

    The one-patient fixture earns no class permissions; active semantic paired
    updates are covered separately above with the controlled two-patient bank.
    """
    from experiments.guidance_signal_20261014.prepare import prepare, ORIGINAL_RECIPE_BY_BANK
    from experiments.guidance_signal_20261014.tests.test_evidence import original
    rt = load_runtime()
    bank, recipe_path, _, manifest = original.__wrapped__(tmp_path, monkeypatch)
    # Complete the manufactured trainer receipts before sealing signal inputs.
    data = rt.io.load_bank_patient(bank, "p1")
    pixels = [int((data["labels"] == cls).sum()) for cls in range(4)]
    manifest["patient_reports"] = {"p1": dict(class_pixels=pixels)}
    rt.io.write_json(bank/"FROZEN.json", manifest)
    bank_sha = rt.io.digest(bank/"FROZEN.json")
    recipe = rt.io.read_json(recipe_path)
    recipe.update(width=8, batch_size=1, lr=3e-4, threads=1)
    rt.io.write_json(recipe_path, recipe)
    monkeypatch.setitem(ORIGINAL_RECIPE_BY_BANK, bank_sha, rt.io.digest(recipe_path))
    student = recipe_path.parent/"student"; student.mkdir()
    model = rt.student.GuidanceStudent(manifest["config"]["feature_dim"], 8)
    ck = dict(model=model.state_dict(), width=8, feature_dim=manifest["config"]["feature_dim"],
              status="REPRESENTATION_ONLY", semantic_updates=0,
              bank_manifest_sha256=bank_sha, config=manifest["config"],
              train_patients=manifest["patients"], locked_splits=manifest["locked_splits"],
              class_patients=[int(n > 0) for n in pixels], allowed_classes=[False]*4)
    torch.save(ck, student/"last.pt")
    rt.io.freeze(student, dict(role="student", status=ck["status"], bank_manifest_sha256=bank_sha,
                              source_sha256=rt.source_sha256))
    signals = tmp_path/"signals"
    prepared = prepare(bank, recipe_path, signals, expected_bank_sha=bank_sha)
    assert prepared["status"] == "PASS_REPRODUCTION"
    checkpoint_sha = rt.io.digest(student/"last.pt")
    plan = pilot.create_plan(bank, signals, student/"last.pt", checkpoint_sha, ["p1"], steps=2, seed=7)
    report = pilot.run_pilot(bank, signals, student/"last.pt", checkpoint_sha, plan, tmp_path/"paired",
                             expected_plan_sha=pilot.json_hash(plan))
    assert report["steps_per_arm"] == 2
    assert report["status"] == "TECHNICAL_PAIRED_TRAIN_PILOT_ONLY"


@pytest.mark.parametrize("destination_kind", ["bank", "signals", "original_run", "frozen_ancestor", "symlink_to_input"])
def test_output_and_plan_paths_reject_input_overlap_and_frozen_ancestors(artifact_fixture, monkeypatch, destination_kind):
    f = artifact_fixture
    if destination_kind in ("bank", "signals"):
        parent = f[destination_kind]
    elif destination_kind == "original_run":
        parent = f["checkpoint"].parent.parent
    elif destination_kind == "frozen_ancestor":
        parent = f["root"] / "unrelated_frozen"
        parent.mkdir(); (parent / "FROZEN.json").write_text("{}")
    else:
        parent = f["root"] / "link"
        parent.symlink_to(f["signals"], target_is_directory=True)
    monkeypatch.setattr(pilot, "_dependencies", lambda: pytest.fail("runtime loaded before destination guard"))
    with pytest.raises(ValueError, match="overlap|frozen/locked"):
        pilot.run_pilot(f["bank"], f["signals"], f["checkpoint"], f["sha"], f["plan"], parent/"output",
                        expected_plan_sha=pilot.json_hash(f["plan"]))
    argv = ["--bank", str(f["bank"]), "--signals", str(f["signals"]), "--checkpoint", str(f["checkpoint"]),
            "--expected-checkpoint-sha", f["sha"], "--plan", str(parent/"PLAN.json"), "--write-plan-only",
            "--patient-ids", "train_a", "train_b", "--steps", "2"]
    with pytest.raises(ValueError, match="overlap|frozen/locked"):
        pilot.main(argv)
    assert not (parent/"output").exists() and not (parent/"PLAN.json").exists()


def test_recipe_changed_after_preparation_cannot_self_pin_new_learning_rate(artifact_fixture):
    f = artifact_fixture
    recipe_path = f["checkpoint"].parent.parent/"RECIPE.json"
    recipe = f["rt"].io.read_json(recipe_path)
    recipe["lr"] *= 10
    f["rt"].io.write_json(recipe_path, recipe)
    with pytest.raises(ValueError, match="signal recipe/source/config/teacher binding"):
        pilot.create_plan(f["bank"], f["signals"], f["checkpoint"], f["sha"], ["train_a"], steps=2)


def test_rv_above_uniform_does_not_imply_positive_winning_contrast():
    q = torch.tensor([.1, .3, .5, .1]).view(1, 4, 1, 1)
    terms = pilot.semantic_terms(q.log(), q, q, torch.ones(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool))
    assert terms["Q"]["RV_above_uniform_positive_mass"] > 0
    assert terms["Q"]["positive_RV_contrast_mass"] == 0
    rv = terms["Q"]["strata_by_original_Q_argmax"]["RV"]
    assert rv["status"] == "NO_WEIGHTED_Q_ARGMAX_SUPPORT" and rv["kl"] is None
    myo = terms["Q"]["strata_by_original_Q_argmax"]["MYO"]
    assert myo["weight_sum"] == 1 and torch.allclose(myo["kl"], torch.tensor(0.), atol=2e-7)


def test_class_argmax_strata_are_pixel_losses_not_target_channel_contributions():
    q = torch.tensor([[[[.1, .1]], [[.3, .7]], [[.5, .1]], [[.1, .1]]]])
    logits = torch.zeros_like(q, requires_grad=True)
    weights = torch.tensor([[[.2, .8]]])
    terms = pilot.semantic_terms(logits, q, q, weights, torch.ones(1, 1, 2, dtype=torch.bool))["Q"]
    strata = terms["strata_by_original_Q_argmax"]
    assert strata["RV"]["weight_sum"] == .8 and strata["MYO"]["weight_sum"] == .2
    for key in ("ce", "entropy", "kl"):
        total = sum(row["contribution_original_denominator"][key] for row in strata.values())
        assert torch.allclose(total, terms[key], atol=2e-7)
    assert not torch.allclose(strata["RV"]["contribution_original_denominator"]["ce"], terms["per_class_ce"][1])


def test_conflicting_cublas_deterministic_policy_fails(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        pilot.configure_determinism(42)
