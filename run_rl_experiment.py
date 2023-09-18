from experiments.dqn_experiement import DQNExperiment

from experiments.mbpo_dqn_experiement import MbpoDqnExperiment


for _ in range(4):
    DQNExperiment().run()

MbpoDqnExperiment().run()
