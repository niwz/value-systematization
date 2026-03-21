from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from advice_reflection_platform.backend.analysis import summarize_runs
from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.gateway import HeuristicDemoGateway, LiveModelGateway
from advice_reflection_platform.backend.orchestrator import (
    DEFAULT_REFLECTION_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    load_batch_jobs,
    run_batch,
    run_single_scenario,
)
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry
from advice_reflection_platform.backend.schemas import AdviceOption, ScenarioRecord

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "Streamlit is required. Install it and rerun `streamlit run advice_reflection_platform/app.py`."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REGISTRY = ScenarioRegistry(DATA_DIR / "scenarios")
STORE = ArtifactStore(BASE_DIR)

DEMO_MODEL = "demo-model"
LIVE_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_gateway(mode: str):
    return HeuristicDemoGateway() if mode == "demo" else LiveModelGateway()


def _init_form_state() -> None:
    st.session_state.setdefault("s_request", "")
    st.session_state.setdefault("s_a_title", "")
    st.session_state.setdefault("s_a_text", "")
    st.session_state.setdefault("s_b_title", "")
    st.session_state.setdefault("s_b_text", "")
    st.session_state.setdefault("s_id", "")
    st.session_state.setdefault("s_domain", "")
    st.session_state.setdefault("run_result", None)


def _load_into_state(s: ScenarioRecord) -> None:
    st.session_state["s_request"] = s.request_text
    st.session_state["s_a_title"] = s.option_a.title
    st.session_state["s_a_text"] = s.option_a.text
    st.session_state["s_b_title"] = s.option_b.title
    st.session_state["s_b_text"] = s.option_b.text
    st.session_state["s_id"] = s.scenario_id
    st.session_state["s_domain"] = s.domain
    st.session_state["run_result"] = None


def _option_label(choice: str | None, order: str, scenario: ScenarioRecord) -> str:
    if choice is None:
        return "—"
    if order == "AB":
        opt = scenario.option_a if choice == "A" else scenario.option_b
    else:
        opt = scenario.option_b if choice == "A" else scenario.option_a
    return f"{choice} — {opt.title}"


# ── Result card ───────────────────────────────────────────────────────────────

def _render_run_card(label: str, record, scenario: ScenarioRecord) -> None:
    p = record.parsed
    with st.container(border=True):
        st.markdown(f"**{label}**")

        if p.within_response_revision:
            st.warning("Within-response revision detected", icon="↩️")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("First choice")
            st.markdown(_option_label(p.first_choice, record.presentation_order, scenario))
        with c2:
            st.caption("Final choice")
            st.markdown(_option_label(p.final_choice, record.presentation_order, scenario))

        if p.final_reason:
            st.caption("Reason")
            st.markdown(p.final_reason)

        with st.expander("Raw response"):
            st.code(record.raw_response, language=None)

        if record.reflection_text:
            with st.expander("Reflection text"):
                st.code(record.reflection_text, language=None)


# ── Single run tab ────────────────────────────────────────────────────────────

def render_single_run() -> None:
    _init_form_state()

    # Sidebar: run configuration
    with st.sidebar:
        st.header("Run config")
        gateway_mode = st.radio("Gateway", ["demo", "live"], horizontal=True)
        if gateway_mode == "demo":
            model_name = DEMO_MODEL
            thinking = False
            st.caption(f"Model: `{DEMO_MODEL}`")
        else:
            model_name = st.selectbox("Model", LIVE_MODELS)
            is_claude = not ("/" in model_name)
            thinking = st.toggle(
                "Extended thinking",
                value=False,
                disabled=not is_claude,
                help="Only available for Claude models. Forces temperature=1 and adds a thinking budget of 8k tokens.",
            )

        presentation_order = st.radio("Presentation order", ["AB", "BA"], horizontal=True)

        with st.expander("System prompt"):
            system_prompt = st.text_area(
                "sp", value=DEFAULT_SYSTEM_PROMPT, height=130, label_visibility="collapsed"
            )
        with st.expander("Reflection prompt"):
            reflection_prompt = st.text_area(
                "rp", value=DEFAULT_REFLECTION_PROMPT, height=100, label_visibility="collapsed"
            )

    # Load from library
    scenarios = {s.scenario_id: s for s in REGISTRY.load_all()}
    if scenarios:
        with st.expander("Load from scenario library"):
            lib_col, btn_col = st.columns([5, 1])
            with lib_col:
                selected_id = st.selectbox(
                    "scenario", sorted(scenarios.keys()), label_visibility="collapsed"
                )
            with btn_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Load into form", use_container_width=True):
                    _load_into_state(scenarios[selected_id])
                    st.rerun()

            # Preview the selected scenario inline
            preview = scenarios[selected_id]
            st.markdown(f"**Request:** {preview.request_text}")
            prev_a, prev_b = st.columns(2)
            with prev_a:
                st.markdown(f"**Option A — {preview.option_a.title}**")
                st.markdown(preview.option_a.text)
            with prev_b:
                st.markdown(f"**Option B — {preview.option_b.title}**")
                st.markdown(preview.option_b.text)

    # Scenario form
    st.markdown("### Scenario")
    st.text_area(
        "Advice request",
        key="s_request",
        height=110,
        placeholder="Describe the situation and what advice is being sought…",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Option A**")
        st.text_input("Title", key="s_a_title", placeholder="e.g. Direct Candor")
        st.text_area(
            "Description", key="s_a_text", height=130,
            placeholder="What does this option recommend?",
        )
    with col_b:
        st.markdown("**Option B**")
        st.text_input("Title", key="s_b_title", placeholder="e.g. Supportive Softening")
        st.text_area(
            "Description", key="s_b_text", height=130,
            placeholder="What does this option recommend?",
        )

    # Save + Run row
    save_col, run_col = st.columns([3, 1])
    with save_col:
        with st.expander("Save as reusable scenario"):
            id_col, domain_col, save_btn_col = st.columns([2, 2, 1])
            with id_col:
                st.text_input("Scenario ID", key="s_id", placeholder="my_scenario_001")
            with domain_col:
                st.text_input("Domain (optional)", key="s_domain", placeholder="e.g. management")
            with save_btn_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Save", use_container_width=True):
                    _do_save()

    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("▶  Run", type="primary", use_container_width=True)

    if run_clicked:
        _do_run(gateway_mode, model_name, presentation_order, system_prompt, reflection_prompt, thinking)

    # Results
    result = st.session_state.get("run_result")
    if result:
        bundle, scenario = result
        st.markdown("---")
        if bundle.changed:
            st.error("CHANGED — the model revised its advice after reflection", icon="🔄")
        else:
            st.success("NO CHANGE — the model held its position after reflection", icon="✅")

        base_col, refl_col = st.columns(2)
        with base_col:
            _render_run_card("Baseline", bundle.baseline, scenario)
        with refl_col:
            if bundle.reflection:
                _render_run_card("Reflection", bundle.reflection, scenario)


def _do_save() -> None:
    save_id = st.session_state.get("s_id", "").strip()
    request_text = st.session_state.get("s_request", "").strip()
    a_title = st.session_state.get("s_a_title", "").strip()
    b_title = st.session_state.get("s_b_title", "").strip()

    if not save_id:
        st.error("Scenario ID is required.")
        return
    if not request_text or not a_title or not b_title:
        st.error("Request text and both option titles are required.")
        return

    rec = ScenarioRecord(
        scenario_id=save_id,
        family_id=save_id,
        request_text=request_text,
        option_a=AdviceOption(
            title=a_title, text=st.session_state.get("s_a_text", "")
        ),
        option_b=AdviceOption(
            title=b_title, text=st.session_state.get("s_b_text", "")
        ),
        domain=st.session_state.get("s_domain", ""),
        latent_dimensions={},
        paraphrase_group="",
    )
    REGISTRY.save([rec], f"{save_id}.json")
    st.success(f"Saved as `{save_id}.json`")


def _do_run(gateway_mode, model_name, presentation_order, system_prompt, reflection_prompt, thinking=False) -> None:
    request_text = st.session_state.get("s_request", "").strip()
    a_title = st.session_state.get("s_a_title", "").strip()
    b_title = st.session_state.get("s_b_title", "").strip()

    if not request_text:
        st.error("Please enter an advice request.")
        return
    if not a_title or not b_title:
        st.error("Both options need a title.")
        return

    scenario = ScenarioRecord(
        scenario_id=f"adhoc_{uuid4().hex[:8]}",
        family_id="adhoc",
        request_text=request_text,
        option_a=AdviceOption(title=a_title, text=st.session_state.get("s_a_text", "")),
        option_b=AdviceOption(title=b_title, text=st.session_state.get("s_b_text", "")),
        domain="",
        latent_dimensions={},
        paraphrase_group="",
    )
    with st.spinner("Running baseline and reflection…"):
        bundle = run_single_scenario(
            scenario=scenario,
            model_name=model_name,
            gateway=build_gateway(gateway_mode),
            presentation_order=presentation_order,
            system_prompt=system_prompt,
            reflection_prompt=reflection_prompt,
            thinking=thinking,
        )
    STORE.write_bundle(bundle)
    st.session_state["run_result"] = (bundle, scenario)


# ── Batch run tab ─────────────────────────────────────────────────────────────

def render_batch_run() -> None:
    st.markdown("### Batch Run")
    scenarios = {s.scenario_id: s for s in REGISTRY.load_all()}

    upload_col, config_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader("Upload CSV or JSON batch file", type=["csv", "json"])
    with config_col:
        gateway_mode = st.radio("Gateway", ["demo", "live"], horizontal=True, key="batch_gw")
        if gateway_mode == "live":
            batch_model = st.selectbox("Model", LIVE_MODELS, key="batch_model")
        else:
            batch_model = DEMO_MODEL
            st.caption(f"Model: `{DEMO_MODEL}`")

    sample_path = DATA_DIR / "uploads" / "sample_batch.csv"
    if uploaded is not None:
        temp_path = DATA_DIR / "uploads" / uploaded.name
        temp_path.write_bytes(uploaded.getvalue())
        jobs = load_batch_jobs(temp_path)
    elif sample_path.exists():
        jobs = load_batch_jobs(sample_path)
        st.caption("Previewing `sample_batch.csv`. Upload a file to override.")
    else:
        st.info("No batch file loaded.")
        return

    st.dataframe(jobs, use_container_width=True)

    if st.button("▶  Run Batch", type="primary"):
        with st.spinner(f"Running {len(jobs)} jobs…"):
            records = run_batch(
                scenarios_by_id=scenarios,
                jobs=jobs,
                gateway=build_gateway(gateway_mode),
                default_model_name=batch_model,
            )
        STORE.write_records(records, "batch_run")
        st.success(f"{len(records)} records written.")
        with st.expander("Summary metrics"):
            st.json(summarize_runs(records))
        st.dataframe([r.to_flat_dict() for r in records], use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Advice Reflection Platform", layout="wide")
    st.title("Advice Reflection Platform")
    st.caption("Does the model change its advice when prompted to reflect?")
    tabs = st.tabs(["Single Run", "Batch Run"])
    with tabs[0]:
        render_single_run()
    with tabs[1]:
        render_batch_run()


if __name__ == "__main__":
    main()
