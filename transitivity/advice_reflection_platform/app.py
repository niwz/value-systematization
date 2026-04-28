from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from advice_reflection_platform.backend.analysis import summarize_runs
from advice_reflection_platform.backend.artifacts import ArtifactStore
from advice_reflection_platform.backend.gateway import LiveModelGateway
from advice_reflection_platform.backend.orchestrator import (
    DEFAULT_OPEN_ADVICE_SYSTEM_PROMPT,
    DEFAULT_PARSER_MODEL_NAME,
    DEFAULT_REFLECTION_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    load_batch_jobs,
    run_batch,
    run_single_scenario,
)
from advice_reflection_platform.backend.scenario_registry import ScenarioRegistry
from advice_reflection_platform.backend.schemas import AdviceOption, RunRecord, ScenarioRecord

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "Streamlit is required for the UI. Install `advice-reflection-platform[ui]` (or `streamlit`) and rerun "
        "`streamlit run advice_reflection_platform/app.py`."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REGISTRY = ScenarioRegistry(DATA_DIR / "scenarios")
STORE = ArtifactStore(BASE_DIR)

LIVE_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
]

RUN_MODE_LABELS = {
    "Structured A/B": "structured_ab",
    "Open-ended advice": "open_advice",
}


def _model_supports_reasoning_controls(model_name: str) -> bool:
    lowered = model_name.lower()
    if "/" not in model_name:
        return True
    supported_prefixes = (
        "openai/gpt-5",
        "openai/o1",
        "openai/o3",
        "anthropic/claude",
        "google/gemini-2.5",
        "google/gemini-2.0-flash-thinking",
        "x-ai/grok",
    )
    supported_keywords = (
        "reasoning",
        "thinking",
        "r1",
        "qwen3",
        "qwq",
        "deepseek-r1",
    )
    return lowered.startswith(supported_prefixes) or any(keyword in lowered for keyword in supported_keywords)


def _init_form_state() -> None:
    st.session_state.setdefault("s_request", "")
    st.session_state.setdefault("s_a_title", "")
    st.session_state.setdefault("s_a_text", "")
    st.session_state.setdefault("s_a_sig", "")
    st.session_state.setdefault("s_b_title", "")
    st.session_state.setdefault("s_b_text", "")
    st.session_state.setdefault("s_b_sig", "")
    st.session_state.setdefault("s_id", "")
    st.session_state.setdefault("s_domain", "")
    st.session_state.setdefault("run_result", None)


def _load_into_state(s: ScenarioRecord) -> None:
    st.session_state["s_request"] = s.request_text
    st.session_state["s_a_title"] = s.option_a.title
    st.session_state["s_a_text"] = s.option_a.text
    st.session_state["s_a_sig"] = s.option_a.action_signature
    st.session_state["s_b_title"] = s.option_b.title
    st.session_state["s_b_text"] = s.option_b.text
    st.session_state["s_b_sig"] = s.option_b.action_signature
    st.session_state["s_id"] = s.scenario_id
    st.session_state["s_domain"] = s.domain
    st.session_state["run_result"] = None


def _option_label(choice: str | None, order: str, scenario: ScenarioRecord, run_mode: str) -> str:
    if choice is None:
        return "—"
    if choice in {"NEITHER", "AMBIGUOUS"}:
        return choice
    if run_mode == "open_advice" or order == "AB":
        opt = scenario.option_a if choice == "A" else scenario.option_b
    else:
        opt = scenario.option_b if choice == "A" else scenario.option_a
    return f"{choice} — {opt.title}"


def _render_structured_card(label: str, record: RunRecord, scenario: ScenarioRecord) -> None:
    p = record.parsed
    with st.container(border=True):
        st.markdown(f"**{label}**")
        if p.within_response_revision:
            st.warning("Within-response revision detected", icon="↩️")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("First choice")
            st.markdown(_option_label(p.first_choice, record.presentation_order, scenario, record.run_mode))
        with c2:
            st.caption("Final choice")
            st.markdown(_option_label(p.final_choice, record.presentation_order, scenario, record.run_mode))
        if p.final_reason:
            st.caption("Reason")
            st.markdown(p.final_reason)
        with st.expander("Raw response"):
            st.code(record.raw_response, language=None)
        if record.reflection_text:
            with st.expander("Reflection text"):
                st.code(record.reflection_text, language=None)


def _render_open_advice_card(label: str, record: RunRecord, scenario: ScenarioRecord) -> None:
    with st.container(border=True):
        st.markdown(f"**{label}**")
        top_left, top_right = st.columns(2)
        with top_left:
            st.caption("Parsed fit")
            st.markdown(_option_label(record.parsed.final_choice, "AB", scenario, record.run_mode))
        with top_right:
            st.caption("Secondary fit")
            st.markdown(record.parser_secondary_fit or "—")
        st.caption("Mixed or conditional")
        st.markdown("Yes" if record.mixed_or_conditional else "No")
        if record.parser_primary_action_summary:
            st.caption("Primary action summary")
            st.markdown(record.parser_primary_action_summary)
        if record.parser_why_not_clean_fit:
            st.caption("Why not a clean fit")
            st.markdown(record.parser_why_not_clean_fit)
        if record.advice_text:
            st.caption("Advice text")
            st.markdown(record.advice_text)
        if record.recommendation_text:
            st.caption("Explicit recommendation")
            st.markdown(record.recommendation_text)
        if record.parsed.final_reason:
            st.caption("Parser reason")
            st.markdown(record.parsed.final_reason)
        with st.expander("Parser raw response"):
            st.code(record.parser_raw_response, language="json")
        if record.reflection_text:
            with st.expander("Reflection text"):
                st.code(record.reflection_text, language=None)


def _render_run_card(label: str, record: RunRecord, scenario: ScenarioRecord) -> None:
    if record.run_mode == "open_advice":
        _render_open_advice_card(label, record, scenario)
    else:
        _render_structured_card(label, record, scenario)


def render_single_run() -> None:
    _init_form_state()

    with st.sidebar:
        st.header("Run config")
        run_mode_label = st.radio("Mode", list(RUN_MODE_LABELS.keys()), horizontal=False)
        run_mode = RUN_MODE_LABELS[run_mode_label]
        model_name = st.selectbox("Model", LIVE_MODELS)
        parser_model_name = DEFAULT_PARSER_MODEL_NAME
        if run_mode == "open_advice":
            parser_model_name = st.selectbox(
                "Parser model",
                [DEFAULT_PARSER_MODEL_NAME, *[m for m in LIVE_MODELS if m != DEFAULT_PARSER_MODEL_NAME]],
                index=0,
            )

        supports_reasoning_controls = _model_supports_reasoning_controls(model_name)
        thinking = st.toggle(
            "Reasoning / extended thinking",
            value=False,
            disabled=not supports_reasoning_controls,
            help=(
                "Available for native Claude models and selected OpenRouter reasoning-capable models. "
                "Anthropic uses a token budget; OpenRouter maps the budget onto each model's reasoning controls where supported."
            ),
        )
        thinking_budget_tokens = st.number_input(
            "Thinking budget tokens",
            min_value=1024,
            max_value=32000,
            step=1024,
            value=8000,
            disabled=not thinking,
            help="Exact budget for Claude-native reasoning; used as the normalized OpenRouter reasoning budget where supported.",
        )
        if thinking and "/" in model_name:
            st.caption("OpenRouter support is model-specific. Unsupported models may ignore this setting in the current stack.")

        if run_mode == "structured_ab":
            presentation_order = st.radio("Presentation order", ["AB", "BA"], horizontal=True)
            default_system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            presentation_order = "AB"
            st.caption("Open-ended mode does not expose A/B options to the generation model.")
            default_system_prompt = DEFAULT_OPEN_ADVICE_SYSTEM_PROMPT

        with st.expander("System prompt"):
            system_prompt = st.text_area(
                "sp",
                value=default_system_prompt,
                height=130,
                label_visibility="collapsed",
                key=f"system_prompt_{run_mode}",
            )
        with st.expander("Reflection prompt"):
            reflection_prompt = st.text_area(
                "rp",
                value=DEFAULT_REFLECTION_PROMPT,
                height=100,
                label_visibility="collapsed",
                key=f"reflection_prompt_{run_mode}",
            )

    scenarios = {s.scenario_id: s for s in REGISTRY.load_all()}
    if scenarios:
        with st.expander("Load from scenario library"):
            lib_col, btn_col = st.columns([5, 1])
            with lib_col:
                selected_id = st.selectbox("scenario", sorted(scenarios.keys()), label_visibility="collapsed")
            with btn_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Load into form", use_container_width=True):
                    _load_into_state(scenarios[selected_id])
                    st.rerun()

            preview = scenarios[selected_id]
            st.markdown(f"**Request:** {preview.request_text}")
            with st.expander("Hidden stance definitions"):
                prev_a, prev_b = st.columns(2)
                with prev_a:
                    st.markdown(f"**Stance A — {preview.option_a.title}**")
                    st.markdown(preview.option_a.text)
                    if preview.option_a.action_signature:
                        st.caption(f"Action signature: {preview.option_a.action_signature}")
                with prev_b:
                    st.markdown(f"**Stance B — {preview.option_b.title}**")
                    st.markdown(preview.option_b.text)
                    if preview.option_b.action_signature:
                        st.caption(f"Action signature: {preview.option_b.action_signature}")

    st.markdown("### Scenario")
    st.text_area(
        "Advice request",
        key="s_request",
        height=110,
        placeholder="Describe the situation and what advice is being sought…",
    )

    stance_header = "Hidden parser labels / benchmark stances" if run_mode == "open_advice" else "A/B options"
    with st.expander(stance_header, expanded=run_mode == "structured_ab"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Stance A**" if run_mode == "open_advice" else "**Option A**")
            st.text_input("Title", key="s_a_title", placeholder="e.g. Direct Candor")
            st.text_area("Description", key="s_a_text", height=130, placeholder="What does this stance recommend?")
            if run_mode == "open_advice":
                st.text_input("Action signature", key="s_a_sig", placeholder="Short mutually exclusive action signature")
        with col_b:
            st.markdown("**Stance B**" if run_mode == "open_advice" else "**Option B**")
            st.text_input("Title ", key="s_b_title", placeholder="e.g. Supportive Softening")
            st.text_area("Description ", key="s_b_text", height=130, placeholder="What does this stance recommend?")
            if run_mode == "open_advice":
                st.text_input("Action signature ", key="s_b_sig", placeholder="Short mutually exclusive action signature")

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
        _do_run(
            model_name=model_name,
            presentation_order=presentation_order,
            system_prompt=system_prompt,
            reflection_prompt=reflection_prompt,
            thinking=thinking,
            thinking_budget_tokens=int(thinking_budget_tokens),
            run_mode=run_mode,
            parser_model_name=parser_model_name,
        )

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
        st.error("Request text and both stance titles are required.")
        return

    rec = ScenarioRecord(
        scenario_id=save_id,
        family_id=save_id,
        request_text=request_text,
        option_a=AdviceOption(
            title=a_title,
            text=st.session_state.get("s_a_text", ""),
            action_signature=st.session_state.get("s_a_sig", ""),
        ),
        option_b=AdviceOption(
            title=b_title,
            text=st.session_state.get("s_b_text", ""),
            action_signature=st.session_state.get("s_b_sig", ""),
        ),
        domain=st.session_state.get("s_domain", ""),
        latent_dimensions={},
        paraphrase_group="",
    )
    REGISTRY.save([rec], f"{save_id}.json")
    st.success(f"Saved as `{save_id}.json`")


def _do_run(
    *,
    model_name: str,
    presentation_order: str,
    system_prompt: str,
    reflection_prompt: str,
    thinking: bool = False,
    thinking_budget_tokens: int = 8000,
    run_mode: str = "structured_ab",
    parser_model_name: str = DEFAULT_PARSER_MODEL_NAME,
) -> None:
    request_text = st.session_state.get("s_request", "").strip()
    a_title = st.session_state.get("s_a_title", "").strip()
    b_title = st.session_state.get("s_b_title", "").strip()
    a_sig = st.session_state.get("s_a_sig", "").strip()
    b_sig = st.session_state.get("s_b_sig", "").strip()

    if not request_text:
        st.error("Please enter an advice request.")
        return
    if not a_title or not b_title:
        st.error("Both hidden stance titles are required.")
        return
    if run_mode == "open_advice" and (not a_sig or not b_sig):
        st.error("Open-ended mode requires short action signatures for both hidden stances.")
        return

    scenario = ScenarioRecord(
        scenario_id=f"adhoc_{uuid4().hex[:8]}",
        family_id="adhoc",
        request_text=request_text,
        option_a=AdviceOption(title=a_title, text=st.session_state.get("s_a_text", ""), action_signature=a_sig),
        option_b=AdviceOption(title=b_title, text=st.session_state.get("s_b_text", ""), action_signature=b_sig),
        domain="",
        latent_dimensions={},
        paraphrase_group="",
    )
    with st.spinner("Running baseline and reflection…"):
        bundle = run_single_scenario(
            scenario=scenario,
            model_name=model_name,
            gateway=LiveModelGateway(),
            presentation_order=presentation_order,
            system_prompt=system_prompt,
            reflection_prompt=reflection_prompt,
            thinking=thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            run_mode=run_mode,
            parser_model_name=parser_model_name,
        )
    STORE.write_bundle(bundle)
    st.session_state["run_result"] = (bundle, scenario)


def render_batch_run() -> None:
    st.markdown("### Batch Run")
    scenarios = {s.scenario_id: s for s in REGISTRY.load_all()}

    upload_col, config_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader("Upload CSV or JSON batch file", type=["csv", "json"])
    with config_col:
        batch_model = st.selectbox("Generation model", LIVE_MODELS, key="batch_model")
        batch_parser_model = st.selectbox(
            "Parser model",
            [DEFAULT_PARSER_MODEL_NAME, *[m for m in LIVE_MODELS if m != DEFAULT_PARSER_MODEL_NAME]],
            index=0,
            key="batch_parser_model",
        )
        st.caption(
            "Batch files may include `run_mode`, `thinking`, `thinking_budget_tokens`, and `parser_model_name` columns. "
            "Defaults are `structured_ab`, `false`, `8000`, and the parser model shown here."
        )

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
                gateway=LiveModelGateway(),
                default_model_name=batch_model,
                parser_model_name=batch_parser_model,
            )
        STORE.write_records(records, "batch_run")
        st.success(f"{len(records)} records written.")
        with st.expander("Summary metrics"):
            st.json(summarize_runs(records))
        st.dataframe([r.to_flat_dict() for r in records], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Advice Reflection Platform", layout="wide")
    st.title("Advice Reflection Platform")
    st.caption("Compare structured A/B decisions with open-ended advice plus parsed recommendations.")
    tabs = st.tabs(["Single Run", "Batch Run"])
    with tabs[0]:
        render_single_run()
    with tabs[1]:
        render_batch_run()


if __name__ == "__main__":
    main()
