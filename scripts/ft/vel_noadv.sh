#!/bin/bash

source env/bin/activate "$(conda shell.bash hook)"
which python

# Cheetah vel multiseed fine-tuning (from random/online data)
########
# NO ADV HEAD
########
. scripts/runner.sh \
  macaw_vel_iid_ft_seeds_random_noadv \
  log/icml_rebuttal/ftr4 \
  config/cheetah_vel/40tasks_offline.json \
  config/alg/standard_rand_inner.json \
  config/alg/overrides/vel_ft_seeds_random_noadv.json
