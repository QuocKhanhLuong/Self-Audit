"""Comprehensive unit tests for CompactStudent model and architecture validation."""
import pytest
import torch
from torch import nn

from self_audit.compact_model import (
    ALLOWED_ARCHITECTURE_KEYS,
    CompactDecoder,
    CompactEncoder,
    CompactStudent,
    ResidualBlock,
    build_compact_model,
    validate_architecture,
)


def test_compact_student_default_construction():
    model = CompactStudent()
    assert model.widths == (16, 32, 64, 128)
    assert model.in_channels == 3
    assert model.num_classes == 4
    assert model.architecture["name"] == "CompactStudent"
    assert model.architecture["model_class"] == "CompactStudent"
    assert model.architecture["init"] == "kaiming_normal"
    assert model.architecture["init_lineage"] == "random_kaiming_distinct_from_dino"
    assert model.architecture["pretrained"] is False
    assert model.architecture["dino_required"] is False
    assert model.architecture["norm"] == "group_norm"


def test_build_compact_model_requires_explicit_mapping():
    with pytest.raises(ValueError, match="explicit architecture Mapping"):
        build_compact_model(None)

    with pytest.raises(ValueError, match="explicit architecture Mapping"):
        build_compact_model([16, 32, 64, 128])


def test_architecture_strict_validation_no_float_or_bool_coercion():
    # Float in widths
    with pytest.raises(ValueError, match="positive integer without coercion"):
        validate_architecture({"widths": [16.0, 32, 64, 128]})

    # Bool in widths
    with pytest.raises(ValueError, match="positive integer without coercion"):
        validate_architecture({"widths": [True, 32, 64, 128]})

    # Negative or zero width
    with pytest.raises(ValueError, match="positive integer without coercion"):
        validate_architecture({"widths": [0, 32, 64, 128]})

    # Float in in_channels
    with pytest.raises(ValueError, match="in_channels must be exactly integer 3"):
        validate_architecture({"in_channels": 3.0})

    # Bool in in_channels
    with pytest.raises(ValueError, match="in_channels must be exactly integer 3"):
        validate_architecture({"in_channels": True})

    # Wrong in_channels
    with pytest.raises(ValueError, match="in_channels must be exactly integer 3"):
        validate_architecture({"in_channels": 1})

    # Float in num_classes
    with pytest.raises(ValueError, match="num_classes must be exactly integer 4"):
        validate_architecture({"num_classes": 4.0})

    # Bool in num_classes
    with pytest.raises(ValueError, match="num_classes must be exactly integer 4"):
        validate_architecture({"num_classes": False})

    # Wrong num_classes
    with pytest.raises(ValueError, match="num_classes must be exactly integer 4"):
        validate_architecture({"num_classes": 2})


def test_architecture_unknown_keys_rejected():
    with pytest.raises(ValueError, match="unknown architecture fields"):
        validate_architecture({"widths": [16, 32, 64, 128], "unknown_hyperparam": True})

    with pytest.raises(ValueError, match="unknown architecture fields"):
        validate_architecture({"widths": [16, 32, 64, 128], "backbone": "resnet50"})


def test_architecture_name_and_norm_validation():
    with pytest.raises(ValueError, match="invalid architecture name"):
        validate_architecture({"name": "WrongStudent", "widths": [16, 32, 64, 128]})

    with pytest.raises(ValueError, match="invalid architecture model_class"):
        validate_architecture({"model_class": "NativeStudent", "widths": [16, 32, 64, 128]})

    with pytest.raises(ValueError, match="architecture norm must be 'group_norm'"):
        validate_architecture({"norm": "batch_norm", "widths": [16, 32, 64, 128]})

    with pytest.raises(ValueError, match="architecture init must be 'kaiming_normal'"):
        validate_architecture({"init": "xavier_uniform", "widths": [16, 32, 64, 128]})

    with pytest.raises(ValueError, match="architecture pretrained must be exact False"):
        validate_architecture({"pretrained": True, "widths": [16, 32, 64, 128]})

    with pytest.raises(ValueError, match="architecture dino_required must be exact False"):
        validate_architecture({"dino_required": True, "widths": [16, 32, 64, 128]})


def test_groupnorm_used_throughout_no_batchnorm():
    model = CompactStudent()
    group_norms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert len(group_norms) > 0
    batch_norms = [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))]
    assert len(batch_norms) == 0


def test_kaiming_initialization_finite():
    model = CompactStudent()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            assert m.weight is not None
            assert torch.isfinite(m.weight).all()
            if m.bias is not None:
                assert torch.all(m.bias == 0)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                assert torch.all(m.weight == 1)
            if m.bias is not None:
                assert torch.all(m.bias == 0)


def test_module_parameters_partitioning():
    model = CompactStudent(widths=[16, 32, 64, 128])
    groups = model.module_parameters()

    assert set(groups.keys()) == {"encoder", "decoder", "head"}
    enc = groups["encoder"]
    dec = groups["decoder"]
    head = groups["head"]

    assert len(enc) > 0
    assert len(dec) > 0
    assert len(head) > 0

    enc_ids = {id(p) for p in enc}
    dec_ids = {id(p) for p in dec}
    head_ids = {id(p) for p in head}

    # Disjointness
    assert len(enc_ids & dec_ids) == 0
    assert len(enc_ids & head_ids) == 0
    assert len(dec_ids & head_ids) == 0

    # Completeness
    all_param_ids = {id(p) for p in model.parameters()}
    assert enc_ids | dec_ids | head_ids == all_param_ids
    assert len(enc) + len(dec) + len(head) == len(list(model.parameters()))


def test_module_named_parameters():
    model = CompactStudent(widths=[16, 32, 64, 128])
    named_groups = model.module_named_parameters()
    assert set(named_groups.keys()) == {"encoder", "decoder", "head"}
    assert len(named_groups["encoder"]) > 0
    assert len(named_groups["decoder"]) > 0
    assert len(named_groups["head"]) > 0


def test_input_range_and_dtype_validation():
    model = CompactStudent()

    # Integer tensor
    with pytest.raises(ValueError, match="expected floating point"):
        model(torch.zeros(2, 3, 32, 32, dtype=torch.int64))

    # Out of range negative
    with pytest.raises(ValueError, match="expected normalized input values in"):
        model(torch.full((1, 3, 32, 32), -0.5))

    # Out of range positive
    with pytest.raises(ValueError, match="expected normalized input values in"):
        model(torch.full((1, 3, 32, 32), 1.5))

    # Wrong channels
    with pytest.raises(ValueError, match="expected 4D tensor with 3 channels"):
        model(torch.rand(2, 1, 32, 32))

    with pytest.raises(ValueError, match="expected 4D tensor with 3 channels"):
        model(torch.rand(2, 4, 32, 32))

    # Non-finite values
    x_nan = torch.rand(1, 3, 32, 32)
    x_nan[0, 0, 5, 5] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        model(x_nan)

    x_inf = torch.rand(1, 3, 32, 32)
    x_inf[0, 0, 5, 5] = float("inf")
    with pytest.raises(ValueError, match="non-finite values"):
        model(x_inf)


def test_valid_mask_validation():
    model = CompactStudent()
    x = torch.rand(2, 3, 32, 32)

    # Valid mask
    mask = torch.ones(2, 32, 32, dtype=torch.bool)
    mask[0, :5, :] = False
    out = model(x, valid_mask=mask)
    assert out.shape == (2, 4, 32, 32)

    # Wrong dtype
    with pytest.raises(ValueError, match="valid_mask must have boolean dtype"):
        model(x, valid_mask=torch.ones(2, 32, 32, dtype=torch.float32))

    # Wrong shape
    with pytest.raises(ValueError, match="valid_mask shape"):
        model(x, valid_mask=torch.ones(2, 16, 16, dtype=torch.bool))

    # Non-tensor
    with pytest.raises(ValueError, match="valid_mask must be a torch.Tensor"):
        model(x, valid_mask=[[True]])


def test_tiny_geometry_groupnorm():
    model = CompactStudent(widths=[16, 32, 64, 128])
    model.eval()

    # Extremely small geometries
    tiny_shapes = [(1, 1), (2, 3), (3, 5), (7, 7), (8, 8)]
    for h, w in tiny_shapes:
        x = torch.rand(1, 3, h, w)
        logits = model(x)
        assert logits.shape == (1, 4, h, w)
        assert torch.isfinite(logits).all()


def test_mode_changes_eval_train():
    model = CompactStudent()
    x = torch.rand(2, 3, 32, 32)

    model.train()
    out_train = model(x)
    assert out_train.shape == (2, 4, 32, 32)

    model.eval()
    out_eval = model(x)
    assert out_eval.shape == (2, 4, 32, 32)


def test_build_compact_model_return_metadata():
    arch = {"widths": [16, 32, 64, 128], "in_channels": 3, "num_classes": 4}
    model, metadata = build_compact_model(arch, return_metadata=True)
    assert isinstance(model, CompactStudent)
    assert metadata["schema"] == "self_audit.compact.model.v1"
    assert metadata["dino_required"] is False
    assert metadata["pretrained"] is False
    assert metadata["weights_only_bytes"] == sum(p.numel() * p.element_size() for p in model.parameters())
    assert metadata["parameters_total"] == sum(p.numel() for p in model.parameters())
