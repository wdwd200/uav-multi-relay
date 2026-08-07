"""Logged MAPPO experiment orchestration."""
from __future__ import annotations
import json, math
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any
import numpy as np
from ..environment import MultiRelayEnvironment
from ..learning import MAPPOAgent, MAPPOUpdateMetrics
from .mappo_checkpoints import MAPPOCheckpointMetadata, save_mappo_checkpoint
from .mappo_evaluator import MAPPOEvaluationConfig, MAPPOEvaluationSummary, evaluate_mappo
from .mappo_trainer import MAPPOTrainingConfig, MAPPOTrainingProgress, train_mappo

@dataclass(frozen=True)
class MAPPOExperimentConfig:
    output_directory: str | Path
    log_interval_steps: int=1000
    evaluation_interval_steps: int=5000
    evaluation_episodes: int=10
    evaluation_seed: int=10000
    checkpoint_interval_steps: int|None=None
    def __post_init__(self)->None:
        object.__setattr__(self,"output_directory",Path(self.output_directory))
        for name in ("log_interval_steps","evaluation_interval_steps","evaluation_episodes"):
            if isinstance(getattr(self,name),bool) or not isinstance(getattr(self,name),Integral) or getattr(self,name)<=0: raise ValueError(f"{name} must be positive")
        if isinstance(self.evaluation_seed,bool) or not isinstance(self.evaluation_seed,Integral): raise ValueError("evaluation_seed must be an integer")
        if self.checkpoint_interval_steps is not None and (isinstance(self.checkpoint_interval_steps,bool) or not isinstance(self.checkpoint_interval_steps,Integral) or self.checkpoint_interval_steps<=0): raise ValueError("checkpoint_interval_steps must be positive or None")

@dataclass(frozen=True)
class MAPPOExperimentResult:
    output_directory: Path
    final_checkpoint: Path
    best_checkpoint: Path
    training_log: Path
    evaluation_log: Path
    summary_file: Path
    best_mean_return: float

def _line(handle: Any, payload: dict[str,object])->None: handle.write(json.dumps(payload,allow_nan=False,separators=(",",":"))+"\n"); handle.flush()
def _metrics(metrics: MAPPOUpdateMetrics|None)->dict[str,float|None]:
    names=tuple(MAPPOUpdateMetrics.__dataclass_fields__)
    if metrics is None:return {name:None for name in names}
    result={name:float(getattr(metrics,name)) for name in names}
    if not all(np.isfinite(value) for value in result.values()):raise ValueError("MAPPO metrics are non-finite")
    return result
def _evaluation(step:int,result:MAPPOEvaluationSummary)->dict[str,object]: return {"environment_steps":step,**asdict(result)}

def run_mappo_experiment(training_env:MultiRelayEnvironment,evaluation_env:MultiRelayEnvironment,agent:MAPPOAgent,training_config:MAPPOTrainingConfig,experiment_config:MAPPOExperimentConfig)->MAPPOExperimentResult:
    if not isinstance(experiment_config,MAPPOExperimentConfig):raise ValueError("experiment_config must be a MAPPOExperimentConfig")
    output=experiment_config.output_directory
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise ValueError("output_directory must be missing or empty")
    output.mkdir(parents=True,exist_ok=True)
    paths={name:output/name for name in ("run_config.json","training_metrics.jsonl","evaluation_metrics.jsonl","best_checkpoint.pt","final_checkpoint.pt","summary.json")}; checkpoints=output/"checkpoints"
    with paths["run_config.json"].open("w",encoding="utf-8") as handle: json.dump({"training_config":asdict(training_config),"experiment_config":{**asdict(experiment_config),"output_directory":str(output)},"agent":{"local_observation_dim":agent.local_observation_dim,"global_state_dim":agent.global_state_dim,"num_relays":agent.num_relays,"action_dim":agent.action_dim,"hidden_dims":list(agent.hidden_dims),"config":asdict(agent.config)}},handle,allow_nan=False,indent=2)
    if experiment_config.checkpoint_interval_steps is not None: save_mappo_checkpoint(checkpoints/"step_000000.pt",agent,MAPPOCheckpointMetadata(0,0,0))
    best=-float("inf"); latest:MAPPOEvaluationSummary|None=None
    with paths["training_metrics.jsonl"].open("w",encoding="utf-8") as train_handle, paths["evaluation_metrics.jsonl"].open("w",encoding="utf-8") as eval_handle:
        def progress(item:MAPPOTrainingProgress)->None:
            nonlocal best,latest
            log=item.environment_steps%experiment_config.log_interval_steps==0 or item.environment_steps==training_config.total_environment_steps
            evaluate=item.environment_steps%experiment_config.evaluation_interval_steps==0 or item.environment_steps==training_config.total_environment_steps
            if log:_line(train_handle,{"environment_steps":item.environment_steps,"total_updates":item.total_updates,"completed_episodes":item.completed_episodes,"mean_rate_e2e_bps":item.mean_rate_e2e_bps,"termination_rate":item.termination_rate,"intervention_rate":item.intervention_rate,"requested_applied_mismatch_rate":item.requested_applied_mismatch_rate,**_metrics(item.last_update_metrics)})
            if experiment_config.checkpoint_interval_steps is not None and item.environment_steps%experiment_config.checkpoint_interval_steps==0:save_mappo_checkpoint(checkpoints/f"step_{item.environment_steps:06d}.pt",agent,MAPPOCheckpointMetadata(item.environment_steps,item.total_updates,item.completed_episodes))
            if evaluate:
                latest=evaluate_mappo(evaluation_env,agent,MAPPOEvaluationConfig(experiment_config.evaluation_episodes,experiment_config.evaluation_seed));_line(eval_handle,_evaluation(item.environment_steps,latest))
                if latest.mean_return>best:best=latest.mean_return;save_mappo_checkpoint(paths["best_checkpoint.pt"],agent,MAPPOCheckpointMetadata(item.environment_steps,item.total_updates,item.completed_episodes))
        summary=train_mappo(training_env,agent,training_config,progress_interval_steps=math.gcd(experiment_config.log_interval_steps,experiment_config.evaluation_interval_steps),progress_callback=progress)
    save_mappo_checkpoint(paths["final_checkpoint.pt"],agent,MAPPOCheckpointMetadata(summary.total_environment_steps,summary.total_updates,summary.completed_episodes))
    if latest is None:raise ValueError("final evaluation did not run")
    with paths["summary.json"].open("w",encoding="utf-8") as handle:json.dump({"total_environment_steps":summary.total_environment_steps,"total_updates":summary.total_updates,"completed_episodes":summary.completed_episodes,"discarded_partial_rollout_steps":summary.discarded_partial_rollout_steps,"best_mean_return":best,"final_evaluation":_evaluation(summary.total_environment_steps,latest),"best_checkpoint":str(paths["best_checkpoint.pt"]),"final_checkpoint":str(paths["final_checkpoint.pt"])},handle,allow_nan=False,indent=2)
    return MAPPOExperimentResult(output,paths["final_checkpoint.pt"],paths["best_checkpoint.pt"],paths["training_metrics.jsonl"],paths["evaluation_metrics.jsonl"],paths["summary.json"],best)

__all__=["MAPPOExperimentConfig","MAPPOExperimentResult","run_mappo_experiment"]
