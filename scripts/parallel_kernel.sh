#!/usr/bin/env bash

for seed in {0..40};
do
    python 'src/fs_grl/scripts/run_graph_kernel.py' train.seed_index=${seed} nn.data.num_test_episodes=25 nn/data=reddit &
done
