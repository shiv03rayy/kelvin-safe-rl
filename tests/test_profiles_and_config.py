import numpy as np

from kelvin_rl.config import iter_params, load_config, load_raw
from kelvin_rl.envs.load_profiles import LoadProfiles


def test_profiles_are_reproducible(cfg):
    a, b = LoadProfiles(cfg), LoadProfiles(cfg)
    np.testing.assert_array_equal(a.get("test", 3).current, b.get("test", 3).current)


def test_train_and_test_profiles_differ(cfg):
    lp = LoadProfiles(cfg)
    train = {lp.get("train", i).current.tobytes() for i in range(lp.count("train"))}
    for i in range(lp.count("test")):
        assert lp.get("test", i).current.tobytes() not in train


def test_profiles_have_right_length_and_mix(cfg):
    lp = LoadProfiles(cfg)
    profiles = [lp.get("train", i) for i in range(50)]
    assert all(len(p.current) == lp.n_samples for p in profiles)
    fc = sum(p.has_fast_charge for p in profiles)
    assert 10 < fc < 40  # roughly fast_charge_prob = 0.5
    assert any(lp.get("test", i).has_fast_charge for i in range(lp.count("test")))


def test_every_parameter_has_a_unit_and_source():
    for name, record in iter_params(load_raw()):
        assert "unit" in record, f"{name} has no unit"
        assert record.get("source"), f"{name} has no source"


def test_overrides_reject_unknown_keys():
    import pytest

    assert load_config(overrides={"mismatch.scale": 0.0})["mismatch"]["scale"] == 0.0
    with pytest.raises(KeyError):
        load_config(overrides={"mismatch.scal": 0.0})
