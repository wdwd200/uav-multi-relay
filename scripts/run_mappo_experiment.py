"""Run a parameter-sharing MAPPO experiment without touching MASAC artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from uav_multi_relay import MultiRelayEnvironment, RewardWeights, scenario_environment_config
from uav_multi_relay.learning import MAPPOAgent, MAPPOConfig
from uav_multi_relay.training import MAPPOExperimentConfig, MAPPOTrainingConfig, run_mappo_experiment

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True); parser.add_argument("--steps",type=int,default=10_000); parser.add_argument("--rollout-steps",type=int,default=1024); parser.add_argument("--max-steps",type=int); parser.add_argument("--waypoint-radius",type=float,default=30.0); parser.add_argument("--update-epochs",type=int,default=10); parser.add_argument("--mini-batch-size",type=int,default=256); parser.add_argument("--evaluation-interval",type=int,default=5_000); parser.add_argument("--evaluation-episodes",type=int,default=10); parser.add_argument("--checkpoint-interval",type=int); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--evaluation-seed",type=int,default=10_000); parser.add_argument("--device",default="cpu"); parser.add_argument("--reward-rate",type=float,default=1.0); parser.add_argument("--reward-link",type=float,default=1.0); parser.add_argument("--reward-separation",type=float,default=1.0); parser.add_argument("--reward-intervention",type=float,default=1.0); parser.add_argument("--reward-motion",type=float,default=1.0); parser.add_argument("--reward-failure",type=float,default=1.0)
    args=parser.parse_args(); np.random.seed(args.seed); torch.manual_seed(args.seed)
    weights=RewardWeights(args.reward_rate,args.reward_link,args.reward_separation,args.reward_intervention,args.reward_motion,args.reward_failure)
    env_config=scenario_environment_config(MultiRelayEnvironment().config,num_relays=4,waypoint_radius_m=args.waypoint_radius,max_steps=args.max_steps,reward_weights=weights)
    training_env=MultiRelayEnvironment(env_config); evaluation_env=MultiRelayEnvironment(env_config); observation,_=training_env.reset(seed=args.seed)
    agent=MAPPOAgent(observation["local"].shape[-1],observation["global"].shape[-1],env_config.num_relays,config=MAPPOConfig(update_epochs=args.update_epochs,mini_batch_size=args.mini_batch_size),device=args.device)
    result=run_mappo_experiment(training_env,evaluation_env,agent,MAPPOTrainingConfig(args.steps,args.rollout_steps,args.seed),MAPPOExperimentConfig(args.output_dir,log_interval_steps=min(args.rollout_steps,args.evaluation_interval),evaluation_interval_steps=args.evaluation_interval,evaluation_episodes=args.evaluation_episodes,evaluation_seed=args.evaluation_seed,checkpoint_interval_steps=args.checkpoint_interval))
    config_path=result.output_directory/"run_config.json"; config=json.loads(config_path.read_text(encoding="utf-8")); config["environment_config"]={"num_relays":env_config.num_relays,"waypoint_radius_m":args.waypoint_radius,"max_steps":env_config.max_steps,"reward_weights":vars(weights)};config_path.write_text(json.dumps(config,allow_nan=False,indent=2),encoding="utf-8")
    print(json.dumps({"output_directory":str(result.output_directory),"final_checkpoint":str(result.final_checkpoint),"best_checkpoint":str(result.best_checkpoint),"best_mean_return":result.best_mean_return},allow_nan=False))
if __name__=="__main__":main()
