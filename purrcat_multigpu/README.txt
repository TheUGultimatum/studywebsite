PURRCAT MULTI-GPU MINER

This package uses one live PurrCat job shared by all NVIDIA GPUs and disjoint nonce partitions.

Requirements:
- NVIDIA GPUs
- CUDA with sm_120 support for RTX 5090
- Ubuntu/Debian

Run:
  bash install_and_run.sh

Optional:
  PURRCAT_GPUS=0,1,2,3 bash install_and_run.sh

Private keys are entered only in the terminal.