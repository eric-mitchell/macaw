#!/bin/bash

source env/bin/activate
which python

# Walker multiseed fine-tuning (from random/online data)
./scripts/runner.sh \
  macaw_walker_iid_ft_seeds_random \
  log \
  config/walker_params/50tasks_offline.json \
  config/alg/standard_rand_inner.json \
  config/alg/overrides/walker_ft_seeds_random.json
