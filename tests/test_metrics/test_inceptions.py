# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pytest

from mmedit.core.evaluation.metrics import FID, KID, InceptionV3


def test_inception():
    model = InceptionV3()
    feats = model(np.random.randint(0, 256, (224, 224, 3)))
    assert len(feats) == 2
    assert feats[0].shape == (1, 2048)
    assert feats[1].shape == (1, 2048)


def test_fid():
    model = InceptionV3()
    feats = [
        model(np.random.randint(0, 256, (224, 224, 3))) for _ in range(10)
    ]
    X, Y = zip(*feats)
    X, Y = np.array(X), np.array(Y)

    fid = FID()
    assert np.allclose()


def test_kid():
    model = InceptionV3()
    feats = [
        model(np.random.randint(0, 256, (224, 224, 3))) for _ in range(10)
    ]
    X, Y = zip(*feats)
    X, Y = np.array(X), np.array(Y)

    kid = KID()
    assert np.allclose()
