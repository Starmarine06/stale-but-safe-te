import numpy as np
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.topology import abilene, synthetic
from stale_but_safe.baselines.heuristics import LPOracle, Greedy
from stale_but_safe.eval.runner import evaluate
tops={"abilene":abilene(), "ba12":synthetic("ba",12,1), "er15":synthetic("er",15,2), "geo15":synthetic("geo",15,3)}
for name,t in tops.items():
    for load in (0.5,0.9,1.2):
        for hp in (0.02,0.1):
            cfg=EnvCfg(load=load,horizon=40,traffic=dict(hotspot_prob=hp))
            o=evaluate(t,cfg,LPOracle(),[0,1])
            print(f"{name:8s} load={load} hot={hp}: oracle gain {o['improvement_vs_ecmp']*100:5.1f}%  ecmp_mu={o['mean_ecmp_max_util']:.2f}")
