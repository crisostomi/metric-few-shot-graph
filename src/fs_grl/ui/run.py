from pathlib import Path

import streamlit as st
import wandb

from nn_core.ui import select_checkpoint

from fs_grl.pl_modules.distance_metric_learning import DistanceMetricLearning


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    return DistanceMetricLearning.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


if wandb.api.api_key is None:
    st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
    st.stop()

st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")

checkpoint_path = select_checkpoint()
model: DistanceMetricLearning = get_model(checkpoint_path=checkpoint_path)
