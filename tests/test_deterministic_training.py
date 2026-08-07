import numpy as np
import torch
from uav_multi_relay import MultiRelayEnvironment, scenario_environment_config
from uav_multi_relay.learning import MultiAgentReplayBuffer, ParameterSharingMADDPG
from uav_multi_relay.training import DeterministicTrainingConfig, train_deterministic
def test_deterministic_training_collects_applied_actions_and_metrics():
    env=MultiRelayEnvironment(scenario_environment_config(MultiRelayEnvironment().config,num_relays=4,max_steps=10)); obs,_=env.reset(seed=0); agent=ParameterSharingMADDPG(obs['local'].shape[-1],obs['global'].shape[-1],4,hidden_dims=(16,16)); buffer=MultiAgentReplayBuffer(40,4,obs['local'].shape[-1],obs['global'].shape[-1],3)
    result=train_deterministic(env,agent,buffer,DeterministicTrainingConfig(20,40,4,4,4,1,.1,0)); assert result.total_updates>0 and buffer.size==20 and np.all(np.isfinite(buffer.applied_actions[:buffer.size]))

def test_seeded_training_reproduces_and_different_seed_changes_parameters():
    def run(seed):
        np.random.seed(seed); torch.manual_seed(seed)
        env=MultiRelayEnvironment(scenario_environment_config(MultiRelayEnvironment().config,num_relays=4,max_steps=8)); obs,_=env.reset(seed=seed)
        agent=ParameterSharingMADDPG(obs['local'].shape[-1],obs['global'].shape[-1],4,hidden_dims=(16,16)); buffer=MultiAgentReplayBuffer(30,4,obs['local'].shape[-1],obs['global'].shape[-1],3,seed=seed)
        result=train_deterministic(env,agent,buffer,DeterministicTrainingConfig(12,30,4,4,4,1,.1,seed))
        return result, {key:value.detach().clone() for key,value in agent.actor.state_dict().items()}
    first, first_state=run(7); second, second_state=run(7); third, third_state=run(8)
    assert first == second and all(torch.equal(first_state[key],second_state[key]) for key in first_state)
    assert any(not torch.equal(first_state[key],third_state[key]) for key in first_state)
