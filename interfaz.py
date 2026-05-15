import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from src.utils import cluster_simulation as cs
from src.features import affinity_matrix

TRANSLATIONS = {
    "en": {
        "page_cluster": "Modeling",
        "page_model": "Experimental Analysis",
        "sim_params": "Simulation Parameters",
        "users": "Users (U)",
        "items": "Items (I)",
        "density": "Density",
        "alpha_user": "Alpha user",
        "alpha_item": "Alpha item",
        "seed": "Seed",
        "min_user_ratings": "Min ratings per user",
        "min_item_ratings": "Min ratings per item",
        "clusters": "Clusters",
        "user_clusters": "User clusters",
        "item_clusters": "Item clusters",
        "affinity_kind": "Affinity kind",
        "beta": "Beta (taste strength)",
        "rating_model": "Rating Model",
        "thresholds": "Thresholds",
        "heatmap_section": "Heatmap",
        "users_shown": "Users shown",
        "items_shown": "Items shown",
        "cluster_title": "Modeling: Distributions and Heatmap",
        "tau_error": "Thresholds must be strictly increasing.",
        "distributions": "Distributions",
        "rating_heatmap_subset": "Rating heatmap (subset)",
        "cluster_heatmap": "Cluster-ordered heatmap",
        "heatmap_title": "Rating heatmap ({u} users x {i} items)",
        "cluster_heatmap_title": "Cluster-ordered heatmap (affinity-colored boundaries)",
        "model_title": "Experimental Analysis — Model evolution across runs",
        "no_animations": "No pre-computed animations found. Run notebooks/Simulation tests/export_animations.ipynb first.",
        "scenario": "Scenario",
        "base_distributions": "Base simulation — Distributions",
        "rating_heatmap": "Rating heatmap",
        "matrix_evolution": "Model evolution across runs",
        "upd_mis": "UPD vs Misalignment by group (across runs)",
    },
    "es": {
        "page_cluster": "Modelado",
        "page_model": "Análisis Experimental",
        "sim_params": "Parámetros de Simulación",
        "users": "Usuarios (U)",
        "items": "Ítems (I)",
        "density": "Densidad",
        "alpha_user": "Alpha usuario",
        "alpha_item": "Alpha ítem",
        "seed": "Semilla",
        "min_user_ratings": "Mín. valoraciones por usuario",
        "min_item_ratings": "Mín. valoraciones por ítem",
        "clusters": "Clusters",
        "user_clusters": "Clusters de usuarios",
        "item_clusters": "Clusters de ítems",
        "affinity_kind": "Tipo de afinidad",
        "beta": "Beta (intensidad de gusto)",
        "rating_model": "Modelo de valoración",
        "thresholds": "Umbrales",
        "heatmap_section": "Mapa de calor",
        "users_shown": "Usuarios mostrados",
        "items_shown": "Ítems mostrados",
        "cluster_title": "Modelado: Distribuciones y Mapa de Calor",
        "tau_error": "Los umbrales deben ser estrictamente crecientes.",
        "distributions": "Distribuciones",
        "rating_heatmap_subset": "Mapa de calor de valoraciones (subconjunto)",
        "cluster_heatmap": "Mapa de calor ordenado por clusters",
        "heatmap_title": "Mapa de calor ({u} usuarios x {i} ítems)",
        "cluster_heatmap_title": "Mapa de calor ordenado por clusters (bordes coloreados por afinidad)",
        "model_title": "Análisis Experimental — Evolución de modelos a lo largo de las iteraciones",
        "no_animations": "No se encontraron animaciones precomputadas. Ejecuta primero notebooks/Simulation tests/export_animations.ipynb.",
        "scenario": "Escenario",
        "base_distributions": "Simulación base — Distribuciones",
        "rating_heatmap": "Mapa de calor de valoraciones",
        "matrix_evolution": "Evolución de modelos a lo largo de las iteraciones",
        "upd_mis": "UPD vs Misalignment por grupo (a lo largo de las iteraciones)",
    },
}



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


@st.cache_data(show_spinner="Loading scenario…")
def load_scenario_context(scenario_dir: Path):
    state = np.load(scenario_dir / "simulation_state.npz")
    U = int(state["num_users"][0])
    base = pd.read_parquet(scenario_dir / "ratings_long.parquet").dropna(subset=["UserID", "MovieID", "Rating"])
    I = int(base["MovieID"].max()) + 1
    R = np.full((U, I), np.nan, dtype=np.float32)
    R[base["UserID"].to_numpy(np.int64), base["MovieID"].to_numpy(np.int64)] = base["Rating"].to_numpy(np.float32)
    return R, state["user_clusters"], state["item_clusters"], state["affinity"]


def _make_responsive(html: str) -> str:
    patch = (
        "<style>"
        "body{margin:0;overflow-x:hidden}"
        ".animation{display:block!important;width:100%!important}"
        ".animation img{width:100%!important;height:auto!important;max-width:100%!important}"
        "input.anim-slider{width:80%!important}"
        "</style>"
    )
    return patch + html


st.set_page_config(page_title="Cluster Simulation", layout="wide")

with st.sidebar:
    lang_choice = st.selectbox("🌐", ["English", "Español"], label_visibility="collapsed")
    T = TRANSLATIONS["en" if lang_choice == "English" else "es"]

    page = st.radio(
        "View",
        [T["page_cluster"], T["page_model"]],
        label_visibility="collapsed",
    )

    if page == T["page_cluster"]:
        st.divider()
        st.header(T["sim_params"])
        U = st.number_input(T["users"], min_value=10, max_value=5000, value=1000, step=50)
        I = st.number_input(T["items"], min_value=10, max_value=5000, value=2000, step=50)
        density = st.slider(T["density"], min_value=0.001, max_value=0.2, value=0.063, step=0.001, format="%.3f")
        alpha_user = st.slider(T["alpha_user"], min_value=0.0, max_value=2.0, value=0.9, step=0.05)
        alpha_item = st.slider(T["alpha_item"], min_value=0.0, max_value=2.0, value=0.9, step=0.05)
        seed = st.number_input(T["seed"], min_value=0, max_value=10_000, value=42, step=1)
        min_user_ratings = st.number_input(
            T["min_user_ratings"], min_value=0, max_value=500, value=20, step=1,
        )
        min_item_ratings = st.number_input(
            T["min_item_ratings"], min_value=0, max_value=500, value=5, step=1,
        )
        st.divider()
        st.subheader(T["clusters"])
        n_user_clusters = st.number_input(
            T["user_clusters"], min_value=2, max_value=50, value=10, step=1,
        )
        n_item_clusters = st.number_input(
            T["item_clusters"], min_value=2, max_value=50, value=10, step=1,
        )
        affinity_kind = st.selectbox(
            T["affinity_kind"], options=["normal", "outliers", "bipolar"], index=0,
        )
        beta = st.slider(T["beta"], min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        st.divider()
        st.subheader(T["rating_model"])
        mu = st.slider("mu", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
        sigma_bu = st.slider("sigma_bu", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        sigma_bi = st.slider("sigma_bi", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        sigma_eps = st.slider("sigma_eps", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        gamma = st.slider("gamma", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        st.subheader(T["thresholds"])
        t1 = st.slider("t1", min_value=-2.0, max_value=2.0, value=-1.2, step=0.1)
        t2 = st.slider("t2", min_value=-2.0, max_value=2.0, value=-0.4, step=0.1)
        t3 = st.slider("t3", min_value=-2.0, max_value=2.0, value=0.4, step=0.1)
        t4 = st.slider("t4", min_value=-2.0, max_value=2.0, value=1.2, step=0.1)
        st.divider()
        st.subheader(T["heatmap_section"])
        heatmap_users = st.number_input(
            T["users_shown"], min_value=10, max_value=5000, value=U, step=10,
        )
        heatmap_items = st.number_input(
            T["items_shown"], min_value=10, max_value=5000, value=I, step=10,
        )

# ── Cluster Simulation page ──────────────────────────────────────────────────
if page == T["page_cluster"]:
    st.title(T["cluster_title"])

    tau = (t1, t2, t3, t4)
    if not (t1 < t2 < t3 < t4):
        st.error(T["tau_error"])
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
            heatmap_title=T["heatmap_title"].format(u=heatmap_users, i=heatmap_items),
        )
        fig3, _, _ = cs.plot_cluster_ordered_heatmap(
            R_syn=R_syn,
            user_cluster=sim_state["user_cluster"],
            item_cluster=sim_state["item_cluster"],
            A=sim_state["A"],
            heatmap_users=heatmap_users,
            heatmap_items=heatmap_items,
            heatmap_title=T["cluster_heatmap_title"],
        )
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    fig.set_size_inches(10, 3)
    fig2.set_size_inches(6, 4)
    fig3.set_size_inches(7, 5)

    st.subheader(T["distributions"])
    st.pyplot(fig, clear_figure=True)

    st.subheader(T["rating_heatmap_subset"])
    st.pyplot(fig2, clear_figure=True)

    st.subheader(T["cluster_heatmap"])
    st.pyplot(fig3, clear_figure=True)

# ── Model Simulation page ────────────────────────────────────────────────────
else:
    st.title(T["model_title"])

    ARTIFACTS_ROOT = Path(__file__).parent / "artifacts"
    scenarios = sorted([
        d.name for d in ARTIFACTS_ROOT.iterdir()
        if d.is_dir() and (d / "animations").is_dir()
    ]) if ARTIFACTS_ROOT.exists() else []

    if not scenarios:
        st.info(T["no_animations"])
    else:
        selected = st.selectbox(T["scenario"], scenarios)
        scenario_dir = ARTIFACTS_ROOT / selected

        R_base, uc, ic, A = load_scenario_context(scenario_dir)

        try:
            fig_d, fig_h = cs.plot_simulation_distributions_and_heatmap(R_base)
            fig_c, _, _ = cs.plot_cluster_ordered_heatmap(
                R_syn=R_base,
                user_cluster=uc,
                item_cluster=ic,
                A=A,
            )
        except ValueError as exc:
            st.warning(str(exc))
            st.stop()

        st.subheader(T["base_distributions"])
        st.pyplot(fig_d, clear_figure=True)

        for fig in (fig_h, fig_c):
            fig.set_size_inches(8, 6)

        col_h, col_c = st.columns(2)
        with col_h:
            st.subheader(T["rating_heatmap"])
            st.pyplot(fig_h, clear_figure=True, width='stretch')
        with col_c:
            st.subheader(T["cluster_heatmap"])
            st.pyplot(fig_c, clear_figure=True, width='stretch')

        st.divider()
        st.subheader(T["matrix_evolution"])
        anim_dir = scenario_dir / "animations"
        html_files = sorted(f for f in anim_dir.glob("anim_*.html") if f.name != "anim_upd_mis_groups.html")
        model_names = [f.stem.removeprefix("anim_") for f in html_files]

        tabs = st.tabs(model_names)
        for tab, html_file in zip(tabs, html_files):
            with tab:
                html = _make_responsive(html_file.read_text(encoding="utf-8"))
                components.html(html, height=750, scrolling=False)

        group_anim_file = anim_dir / "anim_upd_mis_groups.html"
        if group_anim_file.exists() and selected != "Normal":
            st.divider()
            st.subheader(T["upd_mis"])
            html = _make_responsive(group_anim_file.read_text(encoding="utf-8"))
            components.html(html, height=950, scrolling=False)
