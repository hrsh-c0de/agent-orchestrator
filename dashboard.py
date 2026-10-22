"""Streamlit dashboard for the multi-agent orchestrator.

Run:  streamlit run dashboard.py
Requires the FastAPI backend (default http://localhost:8000).
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180

SAMPLE_TASKS: dict[str, str] = {
    "🐛 GitHub issue → PR": (
        "Title: TypeError when loading a user profile\n"
        "Repo: acme/webapp\n"
        "File: app/users.py line 84\n\n"
        "Traceback (most recent call last):\n"
        '  File "app/users.py", line 84, in get_profile\n'
        "    return user.profile.name\n"
        "TypeError: 'NoneType' object has no attribute 'profile'\n\n"
        "A deleted user still has a session. Expected behaviour is a clean 404. "
        "Please fix the root cause and open a PR."
    ),
    "✉️ Email → reply": (
        "Draft a reply to this customer email.\n\n"
        "From: Priya Nair <priya@atlas.example>\n"
        "Subject: Refund request for duplicate charge\n\n"
        "Hi team,\n"
        "We were charged twice for the September invoice. Could you refund the duplicate "
        "charge and confirm when it will appear? Our finance close is Friday."
    ),
    "🔎 Research → report": (
        "Research the trade-offs between event-driven and request-response architectures "
        "for a payments platform. Compare reliability, latency, operational complexity and "
        "cost. Include a recommendation and open questions."
    ),
    "🚨 Alert → triage": (
        "ALERT: checkout-api 5xx error rate is 12% for the last 10 minutes; p95 latency "
        "2.4s. Service: checkout-api. A deploy went out 25 minutes ago. Draft an incident "
        "triage summary for the on-call channel."
    ),
}

STATUS_ICONS = {
    "completed": "✅",
    "running": "⏳",
    "awaiting_approval": "⏸️",
    "rejected": "🚫",
    "failed": "❌",
    "pending": "•",
}
EVENT_ICONS = {
    "started": "▶️",
    "completed": "✅",
    "failed": "❌",
    "retry": "🔁",
    "awaiting_approval": "⏸️",
}
GRAPH_COLORS = {
    "pending": ("#f1f5f9", "#94a3b8"),
    "active": ("#fef9c3", "#ca8a04"),
    "completed": ("#dcfce7", "#16a34a"),
    "failed": ("#fee2e2", "#dc2626"),
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_request(
    method: str,
    path: str,
    api_url: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    api_key: str = "",
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[Any | None, str | None]:
    url = f"{api_url.rstrip('/')}{path}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.request(
            method, url, json=payload, params=params, headers=headers, timeout=timeout
        )
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {url}. Start it with `uvicorn app:app`. ({exc})"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"HTTP {response.status_code}: {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The API returned a non-JSON response."


@st.cache_data(ttl=60, show_spinner=False)
def fetch_graph(api_url: str, api_key: str = "") -> tuple[Any | None, str | None]:
    return api_request("GET", "/graph", api_url, api_key=api_key, timeout=10)


def fetch_health(api_url: str) -> tuple[Any | None, str | None]:
    return api_request("GET", "/health", api_url, timeout=3)


def fetch_runs(
    api_url: str, api_key: str = "", limit: int = 12
) -> tuple[Any | None, str | None]:
    return api_request(
        "GET", "/runs", api_url, params={"limit": limit}, api_key=api_key, timeout=10
    )


# ---------------------------------------------------------------------------
# Graph rendering
# ---------------------------------------------------------------------------
def derive_node_status(run: dict[str, Any] | None, node_id: str) -> str:
    if not run:
        return "pending"
    state = run.get("state") or {}
    logs = state.get("logs") or []
    status = run.get("status")

    if node_id == "start":
        return "completed"
    if node_id == "end":
        return (
            "completed"
            if status == "completed"
            else "failed"
            if status in {"failed", "rejected"}
            else "pending"
        )
    if node_id == "approval" and status == "awaiting_approval":
        return "active"

    node_status = "pending"
    for log in logs:
        if log.get("agent") != node_id:
            continue
        event = log.get("event")
        if event == "started":
            node_status = "active"
        elif event == "completed":
            node_status = "completed"
        elif event == "failed":
            node_status = "failed" if node_status != "completed" else node_status
        elif event == "retry":
            node_status = "active"

    if node_id == "output" and state.get("output"):
        node_status = "completed"
    if node_id == "approval":
        approval_status = state.get("approval_status")
        if approval_status in {"approved", "auto_approved"}:
            node_status = "completed"
        elif approval_status == "rejected":
            node_status = "failed"
    if status == "awaiting_approval" and node_id == "approval":
        node_status = "active"
    if status == "completed" and node_id in {"reader", "planner", "executor", "validator", "output"}:
        node_status = "completed"
    if status == "rejected" and node_id == "approval":
        node_status = "failed"
    if status == "failed" and node_status == "pending" and node_id in {"give_up", "validator"}:
        node_status = "failed"
    return node_status


def workflow_dot(spec: dict[str, Any], run: dict[str, Any] | None) -> str:
    lines = [
        "digraph workflow {",
        "  rankdir=LR;",
        '  graph [bgcolor="transparent", pad="0.15", nodesep="0.35", ranksep="0.55"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, '
        'fontcolor="#0f172a", penwidth=2, margin="0.18,0.10"];',
        '  edge [color="#94a3b8", fontname="Helvetica", fontsize=10, fontcolor="#475569", '
        'arrowsize=0.7];',
    ]
    for node in spec.get("nodes", []):
        status = derive_node_status(run, node["id"])
        fill, border = GRAPH_COLORS[status]
        shape = "hexagon" if node["type"] == "human" else "oval" if node["type"] == "terminal" else "box"
        lines.append(
            f'  "{node["id"]}" [label="{node["label"]}", shape={shape}, '
            f'fillcolor="{fill}", color="{border}"];'
        )
    for edge in spec.get("edges", []):
        attrs: list[str] = []
        if edge.get("label"):
            attrs.append(f'label="{edge["label"]}"')
        if edge.get("conditional"):
            attrs.append("style=dashed")
        attr_text = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}"{attr_text};'
        )
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI fragments
# ---------------------------------------------------------------------------
def render_status_banner(run: dict[str, Any]) -> None:
    status = run.get("status", "unknown")
    icon = STATUS_ICONS.get(status, "•")
    label = status.replace("_", " ").title()
    if status == "completed":
        st.success(f"{icon} {label} · run `{run['run_id'][:10]}`")
    elif status == "awaiting_approval":
        st.warning(f"{icon} {label} · run `{run['run_id'][:10]}`")
    elif status in {"failed", "rejected"}:
        st.error(f"{icon} {label} · run `{run['run_id'][:10]}`")
    else:
        st.info(f"{icon} {label} · run `{run['run_id'][:10]}`")


def render_approval(run: dict[str, Any], api_url: str, api_key: str = "") -> None:
    prompt = run.get("approval") or {}
    run_id = run["run_id"]
    with st.container(border=True):
        st.markdown("#### 🛑 Human approval required")
        risk = str(prompt.get("risk_level", "medium")).upper()
        raw_score = (run.get("state") or {}).get("validation", {}).get("score")
        score_label = f"{float(raw_score):.0%}" if isinstance(raw_score, (int, float)) else "—"
        st.markdown(
            f"**{prompt.get('title', 'Action awaiting approval')}**  \n"
            f"Risk: `{risk}` · Type: `{prompt.get('task_type', '')}` · "
            f"Validation score: `{score_label}`"
        )
        if prompt.get("summary"):
            st.write(prompt["summary"])
        preview = prompt.get("diff_preview")
        if preview:
            language = "diff" if "--- a/" in preview else "text"
            with st.expander("Preview the pending artifact", expanded=True):
                st.code(preview[:6000], language=language)

        reviewer = st.text_input(
            "Reviewer", value="dashboard-user", key=f"reviewer_{run_id}"
        )
        comment = st.text_area(
            "Comment (optional)", key=f"comment_{run_id}", height=80
        )
        approve_col, reject_col, refresh_col = st.columns([1, 1, 1])
        if approve_col.button(
            "✅ Approve & deliver", type="primary", key=f"approve_{run_id}", use_container_width=True
        ):
            with st.spinner("Resuming workflow…"):
                data, error = api_request(
                    "POST",
                    "/approve",
                    api_url,
                    api_key=api_key,
                    payload={
                        "run_id": run_id,
                        "approved": True,
                        "reviewer": reviewer or "dashboard-user",
                        "comment": comment or "",
                    },
                )
            if error:
                st.error(error)
            else:
                st.session_state.run_data = data
                st.rerun()
        if reject_col.button(
            "🚫 Reject", key=f"reject_{run_id}", use_container_width=True
        ):
            with st.spinner("Recording rejection…"):
                data, error = api_request(
                    "POST",
                    "/approve",
                    api_url,
                    api_key=api_key,
                    payload={
                        "run_id": run_id,
                        "approved": False,
                        "reviewer": reviewer or "dashboard-user",
                        "comment": comment or "Rejected from dashboard.",
                    },
                )
            if error:
                st.error(error)
            else:
                st.session_state.run_data = data
                st.rerun()
        if refresh_col.button("🔄 Refresh", key=f"refresh_{run_id}", use_container_width=True):
            data, error = api_request("GET", f"/runs/{run_id}", api_url, api_key=api_key)
            if error:
                st.error(error)
            else:
                st.session_state.run_data = data
                st.rerun()


def render_output(run: dict[str, Any]) -> None:
    state = run.get("state") or {}
    output = state.get("output") or {}
    status = run.get("status")

    st.subheader("📦 Final output")
    if not output:
        if status == "awaiting_approval":
            st.info("The workflow is paused at the human approval gate. Approve or reject it above.")
        elif status == "rejected":
            st.warning("The reviewer rejected this run; no deliverable was produced.")
        elif status == "failed":
            st.error(run.get("error") or "The workflow failed before producing output.")
            issues = state.get("validation", {}).get("issues") or []
            if issues:
                st.markdown("**Validation issues**")
                for issue in issues:
                    st.markdown(f"- {issue}")
        else:
            st.info("Run a task to see the final deliverable.")
        return

    st.markdown(f"### {output.get('title', 'Deliverable')}")
    st.caption(output.get("summary", ""))

    metrics = st.columns(3)
    metrics[0].metric("Kind", str(output.get("kind", "—")).replace("_", " ").title())
    metrics[1].metric("Confidence", f"{float(output.get('confidence', 0)):.0%}")
    metrics[2].metric("Attempts", state.get("attempts", 0))

    links = output.get("links") or []
    if links:
        link_cols = st.columns(min(len(links), 3))
        for index, link in enumerate(links):
            url = str(link.get("url", ""))
            label = str(link.get("label", "Open"))
            target = link_cols[index % len(link_cols)]
            if url.startswith(("http://", "https://")):
                target.link_button(label, url, use_container_width=True)
            else:
                target.markdown(f"[{label}]({url})")

    st.markdown(output.get("body", ""))
    download_col, diff_col = st.columns(2)
    download_col.download_button(
        "⬇️ Download Markdown",
        data=output.get("body", ""),
        file_name=f"{run['run_id'][:8]}-{output.get('kind', 'result')}.md",
        mime="text/markdown",
        use_container_width=True,
        key=f"download_{run['run_id']}",
    )
    artifact = (state.get("result") or {}).get("artifact") or {}
    if artifact.get("diff"):
        diff_col.download_button(
            "⬇️ Download patch",
            data=str(artifact["diff"]),
            file_name=f"{run['run_id'][:8]}.diff",
            mime="text/x-diff",
            use_container_width=True,
            key=f"download_diff_{run['run_id']}",
        )

    if output.get("next_steps"):
        with st.expander("Suggested next steps"):
            for step in output["next_steps"]:
                st.markdown(f"- {step}")
    with st.expander("Output metadata"):
        st.json(output.get("metadata", {}))


def render_logs(run: dict[str, Any]) -> None:
    state = run.get("state") or {}
    logs = state.get("logs") or []
    metrics = st.columns(4)
    metrics[0].metric("Strategy", str(state.get("strategy") or "—").title())
    metrics[1].metric("Attempts", state.get("attempts", 0))
    validation = state.get("validation") or {}
    score = validation.get("score")
    metrics[2].metric("Validation", f"{float(score):.0%}" if isinstance(score, (int, float)) else "—")
    duration = run.get("duration_ms")
    if isinstance(duration, (int, float)):
        duration_label = (
            f"{float(duration):.0f}ms" if float(duration) < 1000 else f"{float(duration) / 1000:.1f}s"
        )
    else:
        duration_label = "—"
    metrics[3].metric("Elapsed", duration_label)

    if not logs:
        st.info("No agent logs yet.")
        return
    with st.container(height=520, border=False):
        for index, log in enumerate(logs):
            event = str(log.get("event", "info"))
            icon = EVENT_ICONS.get(event, "•")
            agent = str(log.get("agent", "agent")).title()
            message = str(log.get("message", ""))
            duration_ms = log.get("duration_ms")
            suffix = f" · {float(duration_ms):.0f}ms" if isinstance(duration_ms, (int, float)) else ""
            expanded = event in {"failed", "awaiting_approval"}
            with st.expander(f"{icon} {agent} · {event}{suffix}", expanded=expanded):
                st.write(message)
                st.caption(str(log.get("timestamp", "")))
                if log.get("data"):
                    st.json(log["data"])
            if index < len(logs) - 1:
                st.markdown(
                    "<div style='border-left:1px dashed #cbd5e1;height:6px;margin-left:12px'></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Raw log events (JSON)"):
        st.json(logs)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Orchestrator",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, value in {
    "task_text": "",
    "context_json": "{}",
    "run_data": None,
    "last_error": None,
    "api_url_input": DEFAULT_API_URL,
    "api_key_input": os.getenv("API_KEY", ""),
}.items():
    st.session_state.setdefault(key, value)


def load_sample(text: str) -> None:
    st.session_state.task_text = text
    st.session_state.run_data = None
    st.session_state.last_error = None


with st.sidebar:
    st.markdown("## 🕸️ Orchestrator")
    api_url = st.text_input("API URL", key="api_url_input")
    api_key = st.text_input(
        "API key",
        key="api_key_input",
        type="password",
        help="Required when API_AUTH_ENABLED=true on the backend.",
    )
    health, health_error = fetch_health(api_url)
    if health:
        provider = "OpenAI" if health.get("llm_enabled") else "Offline (deterministic)"
        st.success(f"API online · v{health.get('version', '?')}\n\nProvider: {provider}")
        if health.get("auth_enabled"):
            if api_key:
                st.caption("🔐 Authenticated workflow endpoints")
            else:
                st.warning("This API requires a key. Enter one above.")
    else:
        st.error("API offline")
        st.caption(health_error)

    st.divider()
    st.markdown("### Run settings")
    auto_approve = st.checkbox(
        "Auto-approve risky actions",
        value=False,
        help="Skip the human gate. Useful for demos; leave off to exercise approvals.",
    )
    max_attempts = st.slider(
        "Max correction loops", min_value=1, max_value=5, value=3,
        help="Validator -> Executor retries before the run is marked failed.",
    )

    st.divider()
    st.markdown("### Example tasks")
    for label, sample in SAMPLE_TASKS.items():
        st.button(
            label,
            key=f"sample_{label}",
            use_container_width=True,
            on_click=load_sample,
            args=(sample,),
        )

    st.divider()
    st.markdown("### Recent runs")
    runs, runs_error = fetch_runs(api_url, api_key=api_key)
    if runs_error:
        st.caption(runs_error)
    elif not runs:
        st.caption("No runs yet.")
    else:
        options = {
            f"{STATUS_ICONS.get(run['status'], '•')} {run['task'][:34]} · {run['run_id'][:6]}": run["run_id"]
            for run in runs
        }
        selected = st.selectbox("Load a previous run", list(options), label_visibility="collapsed")
        if st.button("Load selected run", use_container_width=True):
            data, error = api_request(
                "GET", f"/runs/{options[selected]}", api_url, api_key=api_key
            )
            if error:
                st.error(error)
            else:
                st.session_state.run_data = data
                st.session_state.last_error = None
                st.rerun()

    if st.session_state.run_data:
        if st.button("Clear current run", use_container_width=True):
            st.session_state.run_data = None
            st.session_state.last_error = None
            st.rerun()

    st.divider()
    st.caption(
        "Reader → Planner → Executor → Validator → Approval → Output. "
        "Offline mode produces deterministic artifacts; set `LLM_PROVIDER=openai` for model reasoning."
    )

st.title("Multi-Agent Orchestrator")
st.caption(
    "Universal task runner for bug reports, email replies, research briefs, CRM queries and alerts — "
    "LangGraph workflow with validation, retries and human approval."
)

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    st.session_state.last_error = None

st.text_area(
    "Task",
    key="task_text",
    height=160,
    placeholder="Paste a GitHub issue, an email, a research question, a CRM request or an alert…",
)
with st.expander("Optional context (JSON)", expanded=False):
    st.caption("Attach structured data such as `{\"from\": \"...\", \"subject\": \"...\", \"repo\": \"...\"}`.")
    st.text_area(
        "Context",
        key="context_json",
        height=90,
        label_visibility="collapsed",
    )
submitted = st.button(
    "▶ Run workflow", type="primary", use_container_width=True, key="run_workflow"
)

if submitted:
    task = st.session_state.task_text.strip()
    if not task:
        st.session_state.last_error = "Enter a task before running the workflow."
        st.rerun()
    try:
        context = json.loads(st.session_state.context_json or "{}")
        if not isinstance(context, dict):
            raise ValueError("context must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        st.session_state.last_error = f"Context JSON is invalid: {exc}"
        st.rerun()
    else:
        with st.spinner("Running Reader → Planner → Executor → Validator…"):
            data, error = api_request(
                "POST",
                "/run",
                api_url,
                api_key=api_key,
                payload={
                    "task": task,
                    "context": context,
                    "auto_approve": auto_approve,
                    "max_attempts": max_attempts,
                },
                timeout=REQUEST_TIMEOUT,
            )
        if error:
            st.session_state.last_error = error
        else:
            st.session_state.run_data = data
        st.rerun()

run = st.session_state.run_data

if run:
    state = run.get("state") or {}
    render_status_banner(run)
    if run.get("status") == "awaiting_approval":
        render_approval(run, api_url, api_key)

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.subheader("🗺️ Workflow")
        spec, graph_error = fetch_graph(api_url, api_key)
        if graph_error:
            st.warning(graph_error)
        elif spec:
            st.graphviz_chart(workflow_dot(spec, run))
            st.caption("🟩 completed · 🟨 active · 🟥 failed · ⬜ pending · dashed = conditional edge")
            with st.expander("Mermaid diagram & node details"):
                st.code(spec.get("mermaid", ""), language="mermaid")
                for node in spec.get("nodes", []):
                    st.markdown(f"- **{node['label']}** — {node['description']}")
        st.caption(
            f"Strategy: `{state.get('strategy') or '—'}` · Task type: "
            f"`{run.get('task_type') or '—'}` · Attempts: `{state.get('attempts', 0)}`"
        )
    with right:
        render_output(run)

    st.divider()
    st.subheader("📜 Agent logs")
    logs_tab, state_tab, execution_tab = st.tabs(["Logs", "Full state", "Execution"])
    with logs_tab:
        render_logs(run)
    with state_tab:
        st.json(state)
    with execution_tab:
        st.markdown("**Plan**")
        st.json(state.get("plan") or {})
        st.markdown("**Result artifact**")
        st.json((state.get("result") or {}).get("artifact") or {})
        st.markdown("**Validation**")
        st.json(state.get("validation") or {})
        if run.get("error"):
            st.error(run["error"])
else:
    st.info("Choose an example in the sidebar or paste a task above to start.")

st.divider()
st.caption(
    "Backend: FastAPI · `/run`, `/approve`, `/runs`, `/graph` · "
    "Workflow: LangGraph StateGraph · State: `AgentState` dataclass · "
    "Validation: Pydantic"
)
