import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--algorithm", action="store", help="Algorithm to run", default="mbpo_sac_discrete", type=str,
                    choices=["sac_discrete", "ddqn", "mbpo_sac_discrete", "mbpo_ddqn"])

args = parser.parse_args()

algor = args.algorithm

if algor == "sac_discrete":
    print("Running SAC-Discrete")
    from experiments.sac_experiment import SACExperiment
    SACExperiment.run()

elif algor == "ddqn":
    print("Running DDQN")
    from experiments.ddqn_experiment import DDQNExperiment
    DDQNExperiment().run()
elif algor == "mbpo_sac_discrete":
    print("Running MBPO (SAC-Discrete)")
    from experiments.mbpo_sac_discrete_experiment import MbpoSacDiscreteExperiment
    MbpoSacDiscreteExperiment().run()
elif algor == "mbpo_ddqn":
    print("Running MBPO (DDQN)")
    from experiments.mbpo_ddqn_experiment import MbpoDdqnExperiment
    MbpoDdqnExperiment().run()
else:
    raise ValueError("Algorithm not supported")





