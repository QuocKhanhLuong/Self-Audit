from __future__ import annotations

import torch
import pytest

from self_audit.candidate_pg.views import StatefulViewGenerator, center_crop_pad, cutmix_box, mix_packet


torch.set_num_threads(1)


def _volume(shape=(1, 1, 4, 64, 64)):
    image = torch.linspace(0, 1, int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
    valid = torch.ones((shape[0], shape[2], shape[3], shape[4]), dtype=torch.bool)
    geometry = {"shape": tuple(shape[2:]), "spacing_zyx": (5.0, 1.0, 1.0)}
    return image, valid, geometry


def test_cutmix_box_is_aligned_full_depth_and_within_intersection():
    valid = torch.ones((1, 4, 64, 64), dtype=torch.bool)
    valid[:, :, :16, :] = False
    box = cutmix_box(valid, torch.Generator().manual_seed(42), area_fraction=(.25, .5), align_px=16)
    y0, y1, x0, x1 = box
    assert all(v % 16 == 0 for v in box)
    assert y0 >= 16 and y1 <= 64 and x0 >= 0 and x1 <= 64
    assert .25 <= ((y1 - y0) * (x1 - x0)) / (64 * 64) <= .5


def test_cutmix_box_intersects_anchor_and_partner_valid_support():
    anchor = torch.ones((1, 2, 64, 64), dtype=torch.bool)
    partner = torch.ones_like(anchor)
    partner[:, :, :16] = False
    box = cutmix_box(anchor, torch.Generator().manual_seed(2), partner_valid=partner)
    y0, y1, x0, x1 = box
    assert y0 >= 16 and bool(anchor[:, :, y0:y1, x0:x1].all()) and bool(partner[:, :, y0:y1, x0:x1].all())


def test_cutmix_never_falls_back_to_unaligned_box():
    valid = torch.zeros((1, 2, 32, 32), dtype=torch.bool)
    valid[..., 3:29, 3:29] = True
    with pytest.raises(ValueError, match="no aligned box"):
        cutmix_box(valid, torch.Generator().manual_seed(42), align_px=16)


def test_full_depth_mix_keeps_partner_z_padding_invalid():
    valid = torch.ones((1, 4, 64, 64), dtype=torch.bool)
    partner_valid = center_crop_pad(torch.ones((1, 2, 64, 64), dtype=torch.bool), (4, 64, 64), pad_value=0)
    box = cutmix_box(valid, torch.Generator().manual_seed(42), partner_valid=partner_valid)
    packet = mix_packet(torch.zeros(1, 4, 4, 64, 64), torch.zeros_like(valid, dtype=torch.long), valid.float(), valid, box,
                        partner_p_W=torch.ones(1, 4, 4, 64, 64), partner_y=torch.ones_like(valid, dtype=torch.long),
                        partner_w=partner_valid.float(), partner_valid=partner_valid)
    y0, y1, x0, x1 = box
    assert all(v % 16 == 0 for v in box)
    assert packet["y"][..., y0:y1, x0:x1].eq(1).all()
    assert not packet["valid"][:, (0, 3), y0:y1, x0:x1].any()
    assert packet["w"][:, (0, 3), y0:y1, x0:x1].eq(0).all()


def test_mix_packet_uses_one_box_for_every_field_and_partner_valid():
    shape = (1, 1, 2, 32, 32)
    p = torch.zeros(shape)
    q = torch.ones(shape)
    y = torch.zeros((1, 2, 32, 32), dtype=torch.long)
    y_partner = torch.full_like(y, 3)
    w = torch.ones_like(y, dtype=torch.float32)
    valid = torch.ones_like(y, dtype=torch.bool)
    partner_valid = torch.ones_like(valid)
    packet = mix_packet(
        p, y, w, valid, (0, 32, 0, 32), partner_p_W=q, partner_y=y_partner,
        partner_w=w * 2, partner_valid=partner_valid,
    )
    assert torch.equal(packet["p_W"], q)
    assert torch.equal(packet["y"], y_partner)
    assert torch.equal(packet["w"], w * 2)
    assert torch.equal(packet["valid"], partner_valid)
    assert not packet["p_W"].requires_grad


def test_mix_packet_center_crops_or_pads_partner_before_one_box():
    anchor_p = torch.zeros((1, 1, 4, 32, 32))
    anchor_y = torch.zeros((1, 4, 32, 32), dtype=torch.long)
    anchor_w = torch.ones_like(anchor_y, dtype=torch.float32)
    anchor_v = torch.ones_like(anchor_y, dtype=torch.bool)
    partner_p = torch.ones((1, 1, 4, 16, 16))
    partner_y = torch.full((1, 4, 16, 16), 3, dtype=torch.long)
    partner_w = torch.ones_like(partner_y, dtype=torch.float32)
    partner_v = torch.ones_like(partner_y, dtype=torch.bool)
    packet = mix_packet(
        anchor_p, anchor_y, anchor_w, anchor_v, (16, 24, 16, 24),
        partner_p_W=partner_p, partner_y=partner_y, partner_w=partner_w,
        partner_valid=partner_v,
    )
    assert packet["p_W"].shape == anchor_p.shape and packet["valid"].shape == anchor_v.shape
    assert packet["valid"][:, :, 16:24, 16:24].all()


def test_view_generator_state_replays_spatial_and_intensity_rng():
    image, valid, geometry = _volume()
    generator = StatefulViewGenerator(42)
    state = generator.state_dict()
    first = generator.joint_views(image, valid, geometry)
    generator.load_state_dict(state)
    second = generator.joint_views(image, valid, geometry)
    for left, right in zip(first[:3], second[:3]):
        assert torch.equal(left, right)
    assert first[0].shape == image.shape and first[1].min() >= 0 and first[1].max() <= 1
    assert first[2].dtype == torch.bool and first[2].shape == valid.shape


def test_labeled_target_replays_exact_same_spatial_draw_as_valid():
    image, valid, geometry = _volume()
    target = torch.zeros_like(valid, dtype=torch.long)
    target[:, :, 12:24, 16:32] = 3
    generator = StatefulViewGenerator(9)
    state = generator.state_dict()
    out_image, out_target, out_valid, _ = generator.weak(image, valid, geometry, target)
    control = StatefulViewGenerator(9)
    control.weak(image, valid, geometry)
    assert torch.equal(generator.state_dict()["augmentation"]["generator_state"], control.state_dict()["augmentation"]["generator_state"])
    generator.load_state_dict(state)
    out_image2, out_target2, out_valid2, _ = generator.weak(image, valid, geometry, target)
    assert torch.equal(out_image, out_image2)
    assert torch.equal(out_target, out_target2) and torch.equal(out_valid, out_valid2)


def test_center_crop_pad_marks_padded_cells_when_used_for_partner_valid():
    source = torch.ones((1, 2, 8, 8), dtype=torch.float32)
    result = center_crop_pad(source, (4, 12, 12), pad_value=0)
    assert result.shape == (1, 4, 12, 12)
    assert torch.equal(result[:, 1:3, 2:10, 2:10], torch.ones((1, 2, 8, 8)))
    assert result[:, :, :1].eq(0).all() and result[:, :, :, :2].eq(0).all()
