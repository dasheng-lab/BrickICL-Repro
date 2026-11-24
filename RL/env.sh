# Runtime settings for the RL scripts. Source it from a bash script:
#   source "$(dirname "$0")/../env.sh"
# It activates the conda env, cds to ~/RL and sets
#   SRC            /home/nzd0420/brickICL          read only: brickicl/, nzd-supertry/, its runs/ (data pools, checkpoints)
#   BRICKNET_DATA  $SRC/baseline/bricknet/data     dataset/ and the collision meshes cvx/
#   RUNS           ~/RL/runs                      our outputs
#   LOGS           ~/RL/logs                      our logs
# Any of these, and CONDA_DIR / ENV_NAME, can be overridden in the environment.
RFT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=${ENV_NAME:-bricknet}
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    for d in "${CONDA_DIR:-}" "$HOME/miniconda3" "$HOME/miniforge3" /opt/anaconda3; do
        if [ -n "$d" ] && [ -f "$d/bin/activate" ]; then
            source "$d/bin/activate" "$ENV_NAME"
            break
        fi
    done
fi
export SRC=${SRC:-/home/nzd0420/brickICL}
export NZD_RUNS=${NZD_RUNS:-$SRC/runs/nzd-supertry}
export BRICKNET_DATA=${BRICKNET_DATA:-$SRC/baseline/bricknet/data}
export RUNS=${RUNS:-$RFT/runs}
export LOGS=${LOGS:-$RFT/logs}
# bricknet: the editable install of the bricknet env; other envs (vllm, step 9) find the same copy on the path
export BRICKNET_SRC=${BRICKNET_SRC:-$HOME/brickICL/BrickICL/baseline/bricknet/BrickNet/src}
export PYTHONPATH=$RFT:$SRC:$BRICKNET_SRC${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
[ -w /tmp ] || export TMPDIR=${TMPDIR:-$HOME/.cache/tmp}  # /tmp is read only on limzhmac
mkdir -p "$RUNS" "$LOGS" ${TMPDIR:+"$TMPDIR"}
cd "$RFT"
