import numpy as np
import streamlit as st
import importlib

import cluster_simulation as cs
from src.features import affinity_matrix

cs = importlib.reload(cs)


@st.cache_data(show_spinner=False)
def build_ratings(
    U: int,
    I: int,
    density: float,
    alpha_user: float,
    alpha_item: float,
    seed: int,
    min_user_ratings: int | None,
    min_item_ratings: int | None,
    n_user_clusters: int,
    n_item_clusters: int,
    affinity_kind: str,
    beta: float,
    tau: tuple[float, float, float, float],
    mu: float,
    sigma_bu: float,
    sigma_bi: float,
    sigma_eps: float,
    gamma: float,
):
    A = affinity_matrix(
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        kind=affinity_kind,
        seed=seed,
    )
    return cs.generate_simulation(
        U=U,
        I=I,
        density=density,
        alpha_user=alpha_user,
        alpha_item=alpha_item,
        seed=seed,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        affinity=A,
        beta=beta,
        tau=np.array(tau),
        mu=mu,
        sigma_bu=sigma_bu,
        sigma_bi=sigma_bi,
        sigma_eps=sigma_eps,
        gamma=gamma,
        return_state=True,
    )


st.set_page_config(page_title="Cluster Simulation", layout="wide")
st.title("Cluster Simulation: Distributions and Heatmap")

with st.sidebar:
    st.header("Simulation Parameters")
    U = st.number_input("Users (U)", min_value=10, max_value=5000, value=1000, step=50)
    I = st.number_input("Items (I)", min_value=10, max_value=5000, value=2000, step=50)
    density = st.slider("Density", min_value=0.001, max_value=0.2, value=0.063, step=0.001)
    alpha_user = st.slider("Alpha user", min_value=0.0, max_value=2.0, value=0.9, step=0.05)
    alpha_item = st.slider("Alpha item", min_value=0.0, max_value=2.0, value=0.9, step=0.05)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
    min_user_ratings = st.number_input(
        "Min ratings per user",
        min_value=0,
        max_value=500,
        value=20,
        step=1,
    )
    min_item_ratings = st.number_input(
        "Min ratings per item",
        min_value=0,
        max_value=500,
        value=5,
        step=1,
    )
    st.divider()
    st.subheader("Clusters")
    n_user_clusters = st.number_input(
        "User clusters",
        min_value=2,
        max_value=50,
        value=12,
        step=1,
    )
    n_item_clusters = st.number_input(
        "Item clusters",
        min_value=2,
        max_value=50,
        value=18,
        step=1,
    )
    affinity_kind = st.selectbox(
        "Affinity kind",
        options=["normal", "outliers", "two_types"],
        index=0,
    )
    beta = st.slider("Beta (taste strength)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    st.divider()
    st.subheader("Rating Model")
    mu = st.slider("mu", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    sigma_bu = st.slider("sigma_bu", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    sigma_bi = st.slider("sigma_bi", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    sigma_eps = st.slider("sigma_eps", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    gamma = st.slider("gamma", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    st.subheader("Thresholds")
    t1 = st.slider("t1", min_value=-2.0, max_value=2.0, value=-1.2, step=0.1)
    t2 = st.slider("t2", min_value=-2.0, max_value=2.0, value=-0.4, step=0.1)
    t3 = st.slider("t3", min_value=-2.0, max_value=2.0, value=0.4, step=0.1)
    t4 = st.slider("t4", min_value=-2.0, max_value=2.0, value=1.2, step=0.1)

    st.divider()
    st.subheader("Heatmap")
    heatmap_users = st.number_input(
        "Users shown", min_value=10, max_value=5000, value= U, step=10
    )
    heatmap_items = st.number_input(
        "Items shown", min_value=10, max_value=5000, value= I, step=10
    )

tau = (t1, t2, t3, t4)
if not (t1 < t2 < t3 < t4):
    st.error("Thresholds must be strictly increasing.")
    st.stop()

R_syn, sim_state = build_ratings(
    U=U,
    I=I,
    density=density,
    alpha_user=alpha_user,
    alpha_item=alpha_item,
    seed=seed,
    min_user_ratings=min_user_ratings if min_user_ratings > 0 else None,
    min_item_ratings=min_item_ratings if min_item_ratings > 0 else None,
    n_user_clusters=n_user_clusters,
    n_item_clusters=n_item_clusters,
    affinity_kind=affinity_kind,
    beta=beta,
    tau=tau,
    mu=mu,
    sigma_bu=sigma_bu,
    sigma_bi=sigma_bi,
    sigma_eps=sigma_eps,
    gamma=gamma,
)

try:
    fig, fig2 = cs.plot_simulation_distributions_and_heatmap(
        R_syn,
        heatmap_users=heatmap_users,
        heatmap_items=heatmap_items,
        heatmap_title=f"Rating heatmap ({heatmap_users} users x {heatmap_items} items)",
    )
    fig3, _, _ = cs.plot_cluster_ordered_heatmap(
        R_syn=R_syn,
        user_cluster=sim_state["user_cluster"],
        item_cluster=sim_state["item_cluster"],
        A=sim_state["A"],
        heatmap_users=heatmap_users,
        heatmap_items=heatmap_items,
        heatmap_title="Cluster-ordered heatmap (affinity-colored boundaries)",
    )
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

st.subheader("Distributions")
st.pyplot(fig, clear_figure=True)

st.subheader("Rating heatmap (subset)")
st.pyplot(fig2, clear_figure=True)

st.subheader("Cluster-ordered heatmap")
st.pyplot(fig3, clear_figure=True)
