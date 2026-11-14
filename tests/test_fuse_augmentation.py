import torch
import pytest

from self_audit.candidate_fuse.augmentation import StrongInplaneAugment


def sample():
    image = torch.zeros((1, 1, 3, 32, 40), dtype=torch.float32)
    image[0, 0, :, 10:22, 13:27] = 1
    target = torch.zeros((1, 3, 32, 40), dtype=torch.long)
    target[0, :, 10:22, 13:27] = 3
    return image, target, {"shape": (3, 32, 40), "spacing_zyx": (5., 1., 1.), "affine_zyx": torch.eye(4).tolist(), "context": "full_volume"}


def test_all_probabilities_zero_is_bitwise_identity_and_rng_is_private():
    image, target, geometry = sample()
    aug = StrongInplaneAugment(7, probabilities={k: 0 for k in StrongInplaneAugment(7).probabilities})
    rng = torch.random.get_rng_state().clone()
    xi, yi, gi = aug.spatial(image, target, geometry)
    xt, yt, gt = aug.teacher_view(image, target, geometry)
    assert torch.equal(xi, image) and torch.equal(yi, target) and gi is geometry
    assert torch.equal(xt, image) and torch.equal(yt, target) and gt is geometry
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert xi.data_ptr() == image.data_ptr()


def test_spatial_transforms_align_mask_and_preserve_z_and_metadata():
    image, target, geometry = sample()
    probs = {k: 0 for k in StrongInplaneAugment(1).probabilities}
    probs.update(rotate=1, scale=1, elastic=1, flip_y=1, flip_x=1, rot90=1)
    aug = StrongInplaneAugment(3, probabilities=probs)
    out, mask, out_geometry = aug.spatial(image, target, geometry)
    assert out.shape == image.shape and mask.shape == target.shape
    assert out_geometry is geometry
    assert set(torch.unique(mask).tolist()) <= {0, 3}
    # Every slice has the same transformed footprint; no z interpolation/mixing.
    for z in range(1, 3):
        assert torch.equal(mask[0, z], mask[0, 0])
        assert torch.equal(out[0, 0, z], out[0, 0, 0])
    assert (out[0, 0, 0] > .5).any()
    image_fg, mask_fg = out[0, 0, 0] > .5, mask[0, 0] == 3
    # Compare foreground overlap, so abundant matching background cannot hide
    # a shifted mask. Bilinear/nearest boundaries need not match pixel for pixel.
    assert 2 * (image_fg & mask_fg).sum() / (image_fg.sum() + mask_fg.sum()) > .9


@pytest.mark.parametrize("axis,dim", [("flip_y", -2), ("flip_x", -1)])
def test_axis_flips_match_torch_flip_exactly(axis, dim):
    image, target, geometry = sample()
    probs = {k: 0 for k in StrongInplaneAugment(1).probabilities}
    probs[axis] = 1
    out, mask, _ = StrongInplaneAugment(12, probabilities=probs).spatial(image, target, geometry)
    assert torch.equal(out, torch.flip(image, dims=(dim,)))
    assert torch.equal(mask, torch.flip(target, dims=(dim,)))


def test_rectangular_rot90_keeps_canvas_and_skips_unsafe_spacing():
    image, target, geometry = sample()
    probs = {k: 0 for k in StrongInplaneAugment(1).probabilities}
    probs["rot90"] = 1
    aug = StrongInplaneAugment(22, probabilities=probs)
    out, mask, _ = aug.spatial(image, target, geometry)
    assert out.shape == image.shape and mask.shape == target.shape
    unsafe = dict(geometry, spacing_zyx=(5., 1., 1.01))
    out_unsafe, mask_unsafe, _ = aug.spatial(image, target, unsafe)
    assert torch.equal(out_unsafe, image) and torch.equal(mask_unsafe, target)


def test_elastic_is_applied_after_rotation_and_scale():
    image, target, geometry = sample()
    probs = {k: 0 for k in StrongInplaneAugment(1).probabilities}
    probs.update(rotate=1, scale=1, elastic=1, flip_y=1, flip_x=1, rot90=1)
    aug = StrongInplaneAugment(3, probabilities=probs)
    calls = []
    original = aug._warp

    def recording_warp(im, lab, matrix, displacement=None):
        calls.append(displacement is not None)
        return original(im, lab, matrix, displacement)

    aug._warp = recording_warp
    out, mask, _ = aug.spatial(image, target, geometry)
    # Rotation, scale, elastic field, then spacing-safe quarter turn.
    assert calls == [False, False, True, False]
    assert out.shape == image.shape and mask.shape == target.shape
    assert set(torch.unique(mask).tolist()) <= {0, 3}


def test_intensity_student_view_only_changes_intensity_and_rng_resume_is_exact():
    image, target, geometry = sample()
    probs = {k: 0 for k in StrongInplaneAugment(1).probabilities}
    probs.update(gamma=1, brightness=1, contrast=1, noise=1, blur=1, low_resolution=1)
    aug = StrongInplaneAugment(99, probabilities=probs)
    state = aug.state_dict()
    teacher, student, out_target, out_geometry = aug.joint_views(image, target, geometry)
    expected_state = aug.state_dict()
    aug.load_state_dict(state)
    teacher2, student2, target2, geometry2 = aug.joint_views(image, target, geometry)
    assert torch.equal(teacher, teacher2) and torch.equal(student, student2)
    assert torch.equal(out_target, target2) and out_geometry is geometry2
    assert torch.equal(teacher, image)  # no spatial operations were enabled
    assert torch.equal(aug.state_dict()["generator_state"], expected_state["generator_state"])
    # Intensities are the only difference between the two J views.
    assert not torch.equal(student, teacher)
    assert torch.equal(out_target, target2)
    assert torch.isfinite(student).all() and student.min() >= 0 and student.max() <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_inputs_keep_device_and_mask_alignment():
    image, target, geometry = sample()
    probs = {k: 0 for k in StrongInplaneAugment(1).probabilities}
    probs.update(rotate=1, scale=1, flip_y=1, flip_x=1, rot90=1)
    out, mask, _ = StrongInplaneAugment(4, probabilities=probs).spatial(image.cuda(), target.cuda(), geometry)
    assert out.is_cuda and mask.is_cuda
    assert out.shape == image.shape and mask.shape == target.shape
    assert set(torch.unique(mask).tolist()) <= {0, 3}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_autocast_does_not_reduce_coordinate_or_intensity_precision(device):
    image, target, geometry = sample()
    image, target = image.to(device), target.to(device)
    probabilities = {key: 1 for key in StrongInplaneAugment().probabilities}
    baseline = StrongInplaneAugment(42, probabilities=probabilities).joint_views(image, target, geometry)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        actual = StrongInplaneAugment(42, probabilities=probabilities).joint_views(image, target, geometry)
    for before, after in zip(baseline[:3], actual[:3]):
        assert torch.equal(before, after)
        assert before.dtype == after.dtype


def test_spatial_operations_do_not_swap_or_mix_distinct_z_slices():
    image, target, geometry = sample()
    for z in range(3):
        image[0, 0, z] *= (z + 1) / 3
        target[0, z][target[0, z] > 0] = z + 1
    probabilities = {key: 0 for key in StrongInplaneAugment().probabilities}
    probabilities.update(rotate=1, scale=1, elastic=1, flip_y=1, flip_x=1, rot90=1)
    out, mask, _ = StrongInplaneAugment(3, probabilities=probabilities).spatial(image, target, geometry)
    for z in range(3):
        assert set(mask[0, z].unique().tolist()) == {0, z + 1}
        torch.testing.assert_close(out[0, 0, z] / (z + 1), out[0, 0, 0])
