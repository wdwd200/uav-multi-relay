import json
from pathlib import Path
import numpy as np
from uav_multi_relay import MultiRelayEnvironment, scenario_environment_config
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMADDPG
from uav_multi_relay.training import DeterministicTrainingConfig, train_deterministic
def test_deterministic_training_collects_applied_actions_and_metrics():
    env=MultiRelayEnvironment(scenario_environment_config(MultiRelayEnvironment().config,num_relays=4,max_steps=10)); obs,_=env.reset(seed=0); agent=ParameterSharingMADDPG(obs['local'].shape[-1],obs['global'].shape[-1],4,hidden_dims=(16,16)); buffer=MultiAgentReplayBuffer(40,4,obs['local'].shape[-1],obs['global'].shape[-1],3)
    result=train_deterministic(env,agent,buffer,DeterministicTrainingConfig(20,40,4,4,4,1,.1,0)); assert result.total_updates>0 and buffer.size==20 and np.all(np.isfinite(buffer.applied_actions[:buffer.size]))
