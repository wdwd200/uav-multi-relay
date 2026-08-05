from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from uav_multi_relay.learning import ParameterSharingMASAC
from uav_multi_relay.training import MASACCheckpointMetadata, load_masac_checkpoint, save_masac_checkpoint


def _agent() -> ParameterSharingMASAC:
    return ParameterSharingMASAC(4, 5, 2, hidden_dims=(8, 8), device="cpu")


def test_checkpoint_round_trip_restores_agent_and_metadata() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        tmp_path = Path(directory)
        agent = _agent()
        path = save_masac_checkpoint(tmp_path / "agent.pt", agent, MASACCheckpointMetadata(4, 2, 1))
        loaded, metadata = load_masac_checkpoint(path)
        assert metadata == MASACCheckpointMetadata(4, 2, 1)
        assert loaded.hidden_dims == (8, 8)
        assert np.array_equal(agent.act(np.zeros((2, 4)), deterministic=True), loaded.act(np.zeros((2, 4)), deterministic=True))
        assert all(not parameter.requires_grad for parameter in loaded.target_critic.parameters())
        assert not list(tmp_path.glob(".*.tmp"))


def test_checkpoint_rejects_missing_version_and_corrupt_file() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        path = Path(directory) / "bad.pt"
        torch.save({"format_version": 999}, path)
        with pytest.raises(ValueError):
            load_masac_checkpoint(path)
        path.write_bytes(b"not a torch checkpoint")
        with pytest.raises(ValueError):
            load_masac_checkpoint(path)


def test_checkpoint_metadata_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        MASACCheckpointMetadata(-1, 0, 0)
    with pytest.raises(ValueError):
        MASACCheckpointMetadata(True, 0, 0)
