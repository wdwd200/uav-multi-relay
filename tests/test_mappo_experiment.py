import json
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from uav_multi_relay import MultiRelayEnvironment
from uav_multi_relay.learning import MAPPOAgent, MAPPOConfig, MAPPORollout
from uav_multi_relay.training import MAPPOCheckpointMetadata, MAPPOExperimentConfig, MAPPOTrainingConfig, load_mappo_checkpoint, run_mappo_experiment, save_mappo_checkpoint

def _parts(directory:Path):
    base=MultiRelayEnvironment(); config=replace(base.config,num_relays=1,max_steps=2); train=MultiRelayEnvironment(config); evaluate=MultiRelayEnvironment(config); observation,_=train.reset(seed=0)
    agent=MAPPOAgent(observation["local"].shape[1],observation["global"].shape[0],1,hidden_dims=(8,8),config=MAPPOConfig(update_epochs=1,mini_batch_size=2))
    return train,evaluate,agent,MAPPOTrainingConfig(4,2,0),MAPPOExperimentConfig(directory,2,2,1,100,2)

def test_experiment_writes_artifacts_and_checkpoint_round_trip() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        output=Path(directory)/"run"; result=run_mappo_experiment(*_parts(output))
        assert {item.name for item in output.iterdir()} == {"run_config.json","training_metrics.jsonl","evaluation_metrics.jsonl","summary.json","best_checkpoint.pt","final_checkpoint.pt","checkpoints"}
        loaded,metadata=load_mappo_checkpoint(result.final_checkpoint); observation,_=MultiRelayEnvironment().reset(seed=0)
        assert metadata.environment_steps == 4 and np.array_equal(loaded.act(observation["local"][:1],True),loaded.act(observation["local"][:1],True))
        assert json.loads(result.summary_file.read_text(encoding="utf-8"))["total_updates"] == 2

def test_checkpoint_rejects_corruption_and_nonempty_output() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        path=Path(directory)/"bad.pt"; path.write_bytes(b"bad")
        with pytest.raises(ValueError): load_mappo_checkpoint(path)
        output=Path(directory)/"run"; output.mkdir(); (output/"x").write_text("x")
        with pytest.raises(ValueError): run_mappo_experiment(*_parts(output))

def test_checkpoint_restores_metadata_and_allows_update() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        _,_,agent,*_= _parts(Path(directory)/"unused"); path=save_mappo_checkpoint(Path(directory)/"agent.pt",agent,MAPPOCheckpointMetadata(2,1,1)); loaded,metadata=load_mappo_checkpoint(path)
        assert metadata == MAPPOCheckpointMetadata(2,1,1) and loaded.config == agent.config
        rollout=MAPPORollout(2,1,agent.local_observation_dim,agent.global_state_dim)
        for _ in range(2):
            rollout.add(np.zeros((1,agent.local_observation_dim),np.float32),np.zeros(agent.global_state_dim,np.float32),np.zeros((1,3),np.float32),np.zeros((1,3),np.float32),0.,1.,0.,0.,False,False)
        assert np.isfinite(loaded.update(rollout).policy_loss)
