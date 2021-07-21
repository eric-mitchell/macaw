#!/bin/bash

source env/bin/activate
which python

# Cheetah vel multiseed fine-tuning (from random/online data)
./scripts/runner.sh \
  macaw_vel_iid_ft_seeds_random \
  log \
  config/cheetah_vel/40tasks_offline.json \
  config/alg/standard_rand_inner.json \
  config/alg/overrides/vel_ft_seeds_random.json
