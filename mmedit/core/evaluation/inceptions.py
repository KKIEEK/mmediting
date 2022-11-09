# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from scipy import linalg

from ..registry import METRICS
from .inception_utils import InceptionV3 as _InceptionV3


class InceptionV3:
    """Feature extractor features using InceptionV3 model.

    Args:
        device (torch.device): device to extract feature.
        inception_kwargs (**kwargs): kwargs for InceptionV3.
    """

    def __init__(self, device='cpu', **inception_kwargs):
        # self.inception = _InceptionV3(**inception_kwargs).to(device)
        self.inception = _load_inception_from_url(
            'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        ).to(device).eval()
        self.device = device

    def __call__(self, img1, img2, crop_border=0):
        """Extract features of real and fake images.

        Args:
            img1, img2 (np.ndarray): Images with range [0, 255]
                and shape (H, W, C).

        Returns:
            (tuple): Pair of features extracted from InceptionV3 model.
        """
        return (
            self.forward_inception(self.img2tensor(img1)).numpy(),
            self.forward_inception(self.img2tensor(img2)).numpy(),
        )

    def img2tensor(self, img):
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        #return torch.from_numpy(img / 255.).to(
        #    device=self.device, dtype=torch.float32)
        return torch.tensor(img).to(device=self.device, dtype=torch.uint8)

    def forward_inception(self, x):
        #with torch.no_grad():
        with disable_gpu_fuser_on_pt19():
            return self.inception(
                x, return_features=True).cpu()  #[0].view(x.shape[0], -1).cpu()


def frechet_distance(X, Y, eps=1e-6):
    """Compute the frechet distance."""

    muX, covX = np.mean(X, axis=0), np.cov(X, rowvar=False) + eps
    muY, covY = np.mean(Y, axis=0), np.cov(Y, rowvar=False) + eps

    diff = muX - muY
    cov_sqrt = linalg.sqrtm(covX.dot(covY))
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    frechet_distance = diff.dot(diff) + np.trace(covX) + np.trace(
        covY) - 2 * np.trace(cov_sqrt)
    return frechet_distance


@METRICS.register_module()
class FID:
    """FID metric."""

    def __call__(self, X, Y):
        """Calculate FID.

        Args:
            X (np.ndarray): Input feature X with shape (n_samples, dims).
            Y (np.ndarray): Input feature Y with shape (n_samples, dims).

        Returns:
            (float): fid value.
        """
        return frechet_distance(X, Y)


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef=1):
    """Create a polynomial kernel."""
    Y = X if Y is None else Y
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = ((X @ Y.T) * gamma + coef)**degree
    return K


def mmd2(X, Y, biased=False):
    """Compute the Maximum Mean Discrepancy."""
    XX = polynomial_kernel(X, X)
    YY = polynomial_kernel(Y, Y)
    XY = polynomial_kernel(X, Y)

    m = X.shape[0]
    if biased:
        return (np.sum(XX) + np.sum(YY) - 2 * np.sum(XY)) / m**2

    trX = np.trace(XX)
    trY = np.trace(YY)
    return ((np.sum(XX) - trX) / (m * (m - 1)) + (np.sum(YY) - trY) /
            (m * (m - 1)) - 2 * np.sum(XY) / m**2)


@METRICS.register_module()
class KID:
    """Implementation of `KID <https://arxiv.org/abs/1801.01401>`.

    Args:
        num_repeats (int): Number of repetitions. Default: 100.
        sample_size (int): Size to sample. Default: 1000.
    """

    def __init__(self, num_repeats=100, sample_size=1000, biased=False):
        self.num_repeats = num_repeats
        self.sample_size = sample_size
        self.biased = biased

    def __call__(self, X, Y):
        """Calculate KID.

        Args:
            X (np.ndarray): Input feature X with shape (n_samples, dims).
            Y (np.ndarray): Input feature Y with shape (n_samples, dims).

        Returns:
            (dict): dict containing mean and std of KID values.
        """
        num_samples = X.shape[0]
        kid = list()
        for i in range(self.num_repeats):
            X_ = X[np.random.choice(
                num_samples, self.sample_size, replace=False)]
            Y_ = Y[np.random.choice(
                num_samples, self.sample_size, replace=False)]
            kid.append(mmd2(X_, Y_, biased=self.biased))
        kid = np.array(kid)
        return dict(KID_MEAN=kid.mean(), KID_STD=kid.std())


import hashlib
import os

import click
import requests
import torch.distributed as dist
import torch.nn as nn
from requests.exceptions import InvalidURL, RequestException, Timeout

MMEDIT_CACHE_DIR = os.path.expanduser('~') + '/.cache/openmmlab/mmedit/'


def get_content_from_url(url, timeout=15, stream=False):
    """Get content from url.

    Args:
        url (str): Url for getting content.
        timeout (int): Set the socket timeout. Default: 15.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=stream)
    except InvalidURL as err:
        raise err  # type: ignore
    except Timeout as err:
        raise err  # type: ignore
    except RequestException as err:
        raise err  # type: ignore
    except Exception as err:
        raise err  # type: ignore
    return response


def download_from_url(url,
                      dest_path=None,
                      dest_dir=MMEDIT_CACHE_DIR,
                      hash_prefix=None):
    """Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download.
        dest_path (str): Path where object will be saved.
        dest_dir (str): The directory of the destination. Defaults to
            ``'~/.cache/openmmlab/mmgen/'``.
        hash_prefix (string, optional): If not None, the SHA256 downloaded
            file should start with `hash_prefix`. Default: None.
    Return:
        str: path for the downloaded file.
    """
    # get the exact destination path
    if dest_path is None:
        filename = url.split('/')[-1]
        dest_path = os.path.join(dest_dir, filename)

    if dest_path.startswith('~'):
        dest_path = os.path.expanduser('~') + dest_path[1:]

    # advoid downloading existed file
    if os.path.exists(dest_path):
        return dest_path

    rank, ws = 0, 1

    # only download from the master process
    if rank == 0:
        # mkdir
        _dir = os.path.dirname(dest_path)
        os.makedirs(_dir, exist_ok=True)

        if hash_prefix is not None:
            sha256 = hashlib.sha256()

        response = get_content_from_url(url, stream=True)
        size = int(response.headers.get('content-length'))
        with open(dest_path, 'wb') as fw:
            content_iter = response.iter_content(chunk_size=1024)
            with click.progressbar(content_iter, length=size / 1024) as chunks:
                for chunk in chunks:
                    if chunk:
                        fw.write(chunk)
                        fw.flush()
                        if hash_prefix is not None:
                            sha256.update(chunk)

        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    f'invalid hash value, expected "{hash_prefix}", but got '
                    f'"{digest}"')

    # sync the other processes
    if ws > 1:
        dist.barrier()

    return dest_path


def _load_inception_from_path(inception_path):
    """Load inception from passed path.

    Args:
        inception_path (str): The path of inception.
    Returns:
        nn.Module: The loaded inception.
    """
    print('Try to load Tero\'s Inception Model from '
          f'\'{inception_path}\'.', 'current')
    try:
        model = torch.jit.load(inception_path)
        print('Load Tero\'s Inception Model successfully.', 'current')
    except Exception as e:
        model = None
        print('Load Tero\'s Inception Model failed. '
              f'\'{e}\' occurs.', 'current')
    return model


def _load_inception_from_url(inception_url: str) -> nn.Module:
    """Load Inception network from the give `inception_url`"""
    inception_url = inception_url if inception_url else TERO_INCEPTION_URL
    print(f'Try to download Inception Model from {inception_url}...',
          'current')
    try:
        path = download_from_url(inception_url, dest_dir=MMEDIT_CACHE_DIR)
        print('Download Finished.', 'current')
        return _load_inception_from_path(path)
    except Exception as e:
        print(f'Download Failed. {e} occurs.', 'current')
        return None


from contextlib import contextmanager


@contextmanager
def disable_gpu_fuser_on_pt19():
    """On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run.

    Refers to:
      https://github.com/GaParmar/clean-fid/blob/5e1e84cdea9654b9ac7189306dfa4057ea2213d8/cleanfid/inception_torchscript.py#L9  # noqa
      https://github.com/GaParmar/clean-fid/issues/5
      https://github.com/pytorch/pytorch/issues/64062
    """
    if torch.__version__.startswith('1.9.'):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith('1.9.'):
        torch._C._jit_override_can_fuse_on_gpu(old_val)
