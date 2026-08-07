"""Experiment artifacts for both deterministic parameter-sharing algorithms."""
from __future__ import annotations
import json,math
from dataclasses import asdict,dataclass
from numbers import Integral
from pathlib import Path
import numpy as np
from ..environment import MultiRelayEnvironment
from ..learning import MultiAgentReplayBuffer,ParameterSharingMADDPG,ParameterSharingMATD3
from .deterministic_checkpoints import DeterministicCheckpointMetadata,save_deterministic_checkpoint
from .deterministic_evaluator import DeterministicEvaluationConfig,evaluate_deterministic
from .deterministic_trainer import DeterministicTrainingConfig,DeterministicTrainingProgress,train_deterministic
@dataclass(frozen=True)
class DeterministicExperimentConfig:
    output_directory:str|Path; log_interval_steps:int=1000; evaluation_interval_steps:int=5000; evaluation_episodes:int=10; evaluation_seed:int=10000; checkpoint_interval_steps:int|None=None
    def __post_init__(self)->None:
        output=Path(self.output_directory)
        if output.exists() and not output.is_dir(): raise ValueError("output_directory must be a directory")
        object.__setattr__(self,"output_directory",output)
        for name in ("log_interval_steps","evaluation_interval_steps","evaluation_episodes"):
            if isinstance(getattr(self,name),bool) or not isinstance(getattr(self,name),Integral) or getattr(self,name)<=0: raise ValueError(f"{name} must be positive")
        if self.checkpoint_interval_steps is not None and (not isinstance(self.checkpoint_interval_steps,Integral) or self.checkpoint_interval_steps<=0): raise ValueError("checkpoint_interval_steps must be positive or None")
@dataclass(frozen=True)
class DeterministicExperimentResult:
    output_directory:Path;final_checkpoint:Path;best_checkpoint:Path;training_log:Path;evaluation_log:Path;summary_file:Path;best_mean_return:float
def run_deterministic_experiment(training_env:MultiRelayEnvironment,evaluation_env:MultiRelayEnvironment,agent:ParameterSharingMADDPG|ParameterSharingMATD3,replay_buffer:MultiAgentReplayBuffer,training_config:DeterministicTrainingConfig,experiment_config:DeterministicExperimentConfig)->DeterministicExperimentResult:
    output=experiment_config.output_directory
    if output.exists() and any(output.iterdir()): raise ValueError("output_directory must be missing or empty")
    output.mkdir(parents=True,exist_ok=True); checkpoints=output/"checkpoints"; training_log=output/"training_metrics.jsonl"; evaluation_log=output/"evaluation_metrics.jsonl"; best=output/"best_checkpoint.pt"; final=output/"final_checkpoint.pt"; summary=output/"summary.json"
    with (output/"run_config.json").open("w",encoding="utf-8") as h: json.dump({"algorithm":agent.algorithm,"training_config":asdict(training_config),"experiment_config":{**asdict(experiment_config),"output_directory":str(output)},"agent_config":{k:(list(v) if k=="hidden_dims" else v) for k,v in agent.__dict__.items() if k in _agent_names(agent)}},h,allow_nan=False,indent=2)
    if experiment_config.checkpoint_interval_steps is not None: save_deterministic_checkpoint(checkpoints/"step_000000.pt",agent,DeterministicCheckpointMetadata(0,0,0))
    best_return=-float("inf");latest=None
    with training_log.open("w",encoding="utf-8") as th,evaluation_log.open("w",encoding="utf-8") as eh:
        def progress(p:DeterministicTrainingProgress)->None:
            nonlocal best_return,latest
            if p.environment_steps%experiment_config.log_interval_steps==0 or p.environment_steps==training_config.total_environment_steps:
                metrics={name:(None if p.last_update_metrics is None else float(getattr(p.last_update_metrics,name))) for name in p.last_update_metrics.__dataclass_fields__} if p.last_update_metrics else {}
                th.write(json.dumps({**asdict(p),"last_update_metrics":metrics},allow_nan=False)+"\n");th.flush()
            if experiment_config.checkpoint_interval_steps is not None and p.environment_steps%experiment_config.checkpoint_interval_steps==0: save_deterministic_checkpoint(checkpoints/f"step_{p.environment_steps:06d}.pt",agent,DeterministicCheckpointMetadata(p.environment_steps,p.total_updates,p.completed_episodes))
            if p.environment_steps%experiment_config.evaluation_interval_steps==0 or p.environment_steps==training_config.total_environment_steps:
                latest=evaluate_deterministic(evaluation_env,agent,DeterministicEvaluationConfig(experiment_config.evaluation_episodes,experiment_config.evaluation_seed)); payload={"environment_steps":p.environment_steps,**asdict(latest)};eh.write(json.dumps(payload,allow_nan=False)+"\n");eh.flush()
                if latest.mean_return>best_return: best_return=latest.mean_return;save_deterministic_checkpoint(best,agent,DeterministicCheckpointMetadata(p.environment_steps,p.total_updates,p.completed_episodes))
        result=train_deterministic(training_env,agent,replay_buffer,training_config,progress_interval_steps=math.gcd(experiment_config.log_interval_steps,experiment_config.evaluation_interval_steps),progress_callback=progress)
    save_deterministic_checkpoint(final,agent,DeterministicCheckpointMetadata(result.total_environment_steps,result.total_updates,result.completed_episodes))
    if latest is None: raise ValueError("final evaluation was not completed")
    payload={**asdict(result),"last_update_metrics":None if result.last_update_metrics is None else asdict(result.last_update_metrics),"algorithm":agent.algorithm,"best_mean_return":best_return,"best_checkpoint":str(best),"final_checkpoint":str(final)}
    with summary.open("w",encoding="utf-8") as h: json.dump(payload,h,allow_nan=False,indent=2)
    return DeterministicExperimentResult(output,final,best,training_log,evaluation_log,summary,float(best_return))
def _agent_names(agent:object)->tuple[str,...]: return ("local_observation_dim","global_state_dim","num_relays","action_dim","hidden_dims","gamma","tau","actor_learning_rate","critic_learning_rate","policy_noise_std","noise_clip","policy_delay")
__all__=["DeterministicExperimentConfig","DeterministicExperimentResult","run_deterministic_experiment"]
