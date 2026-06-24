import numpy as np

from caligo.background import kzz_profile_from_settings


def test_kzz_profile_shape():
    Pbar = np.logspace(-8, 2, 50)

    settings = {
        "upper": 1e9,
        "mid": 1e8,
        "deep": 1e8,
        "break_top_bar": 1e-5,
        "break_deep_bar": 30.0,
    }

    Kzz = kzz_profile_from_settings(Pbar, settings)

    assert Kzz.shape == Pbar.shape
    assert np.all(np.isfinite(Kzz))
    assert np.all(Kzz > 0)