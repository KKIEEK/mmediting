# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.core.evaluation.metrics import FID, KID, InceptionV3


def test_inception():
    img1 = np.random.randint(0, 256, (224, 224, 3))
    img2 = np.random.randint(0, 256, (224, 224, 3))

    # test `img2tensor` and `forward_inception` method
    # for StyleGAN style's inception
    inception = InceptionV3(style='StyleGAN')
    t = inception.img2tensor(img1)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.uint8
    assert t.shape == (1, 3, 224, 224)
    t = inception.forward_inception(t)
    assert t.shape == (1, 2048)

    # test `img2tensor` and `forward_inception` method for PyTorch style's one
    inception = InceptionV3(style=None)
    t = inception.img2tensor(img1)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.shape == (1, 3, 224, 224)
    assert 0 <= t <= 1
    t = inception.forward_inception(t)
    assert t.shape == (1, 2048)

    # test `__call__` method in cpu
    inception = InceptionV3(device='cpu')
    feats = inception(img1, img2)
    assert isinstance(feats, tuple) and len(feats) == 2
    assert feats[0].shape == (1, 2048)
    assert feats[1].shape == (1, 2048)

    # test `__call__` method in cuda
    if torch.cuda.is_available():
        inception = InceptionV3(device='cuda')
        feats = inception(img1, img2)
        assert isinstance(feats, tuple) and len(feats) == 2
        assert feats[0].shape == (1, 2048)
        assert feats[1].shape == (1, 2048)


def test_fid():
    fid = FID()
    fid_result = fid(np.ones((10, 2048)), np.ones((10, 2048)))
    assert isinstance(fid_result, float)
    assert np.testing.assert_equal(fid_result, 0)


def test_kid():
    kid = KID(num_repeats=1, sample_size=10)
    kid_result = kid(np.ones((10, 2048)), np.ones((10, 2048)))
    assert isinstance(kid_result, dict)
    assert 'KID_MEAN' in kid_result and 'KID_STD' in kid_result
    assert np.testing.assert_equal(kid_result['KID_MEAN'], 0)
    assert np.testing.assert_equal(kid_result['KID_STD'], 0)

    # if sample size > number of samples
    with pytest.raises(ValueError):
        kid = KID(sample_size=100)
        kid_result = kid(np.ones((10, 2048)), np.ones((10, 2048)))
