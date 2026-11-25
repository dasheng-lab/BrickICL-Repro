# Shared runtime settings for the brickicl scripts. Source it from a bash script:
#   source "$(dirname "$0")/../env.sh"
# It activates the conda env, cds to the repo root and points data, outputs and logs at the standard places:
#   BRICKNET_DATA  $REPO/data/bricknet   dataset/ and the collision meshes cvx/
#   RUNS           $REPO/runs                     experiment outputs (not tracked)
#   LOGS           $REPO/logs                     logs (not tracked)
# Model weights come from the Hugging Face cache (~/.cache/huggingface unless HF_HOME is set).
# Any of these, and CONDA_DIR / ENV_NAME, can be overridden in the environment.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_NAME=${ENV_NAME:-bricknet}
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    for d in "${CONDA_DIR:-}" "$HOME/miniforge3" "$HOME/miniconda3" /opt/anaconda3; do
        if [ -n "$d" ] && [ -f "$d/bin/activate" ]; then
            source "$d/bin/activate" "$ENV_NAME"
            break
        fi
    done
fi
export BRICKNET_DATA=${BRICKNET_DATA:-$REPO/data/bricknet}
export RUNS=${RUNS:-$REPO/runs}
export LOGS=${LOGS:-$REPO/logs}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
[ -w /tmp ] || export TMPDIR=${TMPDIR:-$HOME/.cache/tmp}  # some shared containers have a read-only /tmp
mkdir -p "$RUNS" "$LOGS" ${TMPDIR:+"$TMPDIR"}
cd "$REPO"
