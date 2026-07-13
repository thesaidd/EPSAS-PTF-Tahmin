from datetime import timedelta
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components import (
    create_day_ahead_forecast_figure,
    create_forecast_figure,
    create_interval_width_figure,
    create_risk_distribution_figure,
    create_risk_error_figure,
    format_metric_value,
    prepare_day_ahead_table,
    prepare_prediction_table,
    readable_timestamp,
    to_istanbul_time,
)
from dashboard.data_access import (
    load_decision_metrics,
    load_decision_predictions,
    load_decision_runs,
    load_latest_day_ahead_forecast,
    load_latest_monitoring_snapshot,
    load_latest_pipeline_run,
)

st.set_page_config(
    page_title="EPİAŞ PTF Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(ttl=60)
def cached_decision_runs() -> pd.DataFrame:
    return load_decision_runs()


@st.cache_data(ttl=60)
def cached_decision_metrics(decision_run_id: str) -> dict[str, Any] | None:
    return load_decision_metrics(decision_run_id)


@st.cache_data(ttl=60)
def cached_predictions(
    decision_run_id: str,
    start_date,
    end_date,
    risk_levels: tuple[str, ...],
    limit: int,
) -> pd.DataFrame:
    return load_decision_predictions(
        decision_run_id=decision_run_id,
        start_date=start_date,
        end_date=end_date,
        risk_levels=list(risk_levels),
        limit=limit,
    )


@st.cache_data(ttl=60)
def cached_latest_day_ahead_forecast() -> tuple[dict[str, Any] | None, pd.DataFrame]:
    return load_latest_day_ahead_forecast()


@st.cache_data(ttl=60)
def cached_latest_pipeline_run() -> dict[str, Any] | None:
    return load_latest_pipeline_run()


@st.cache_data(ttl=60)
def cached_latest_monitoring_snapshot() -> dict[str, Any] | None:
    return load_latest_monitoring_snapshot()


def show_metric_cards(metrics: dict[str, Any]) -> None:
    first = st.columns(4)
    first[0].metric("Selected model", str(metrics.get("selected_model", "—")))
    first[1].metric("MAE", format_metric_value(metrics.get("mae")))
    first[2].metric("RMSE", format_metric_value(metrics.get("rmse")))
    first[3].metric("R²", format_metric_value(metrics.get("r2"), decimals=3))

    second = st.columns(4)
    second[0].metric(
        "95% coverage",
        format_metric_value(metrics.get("interval_coverage_95"), suffix="%"),
    )
    second[1].metric(
        "Mean interval width",
        format_metric_value(metrics.get("mean_interval_width")),
    )
    second[2].metric("Rows", format_metric_value(metrics.get("count"), decimals=0))
    second[3].metric(
        "Evaluation window",
        f"{readable_timestamp(metrics.get('evaluation_start'))} → "
        f"{readable_timestamp(metrics.get('evaluation_end'))}",
    )


def show_monitoring_quality_section() -> None:
    st.subheader("Monitoring & Quality")
    try:
        snapshot = cached_latest_monitoring_snapshot()
    except Exception as exc:
        st.error("Could not load latest monitoring snapshot.")
        st.exception(exc)
        return

    if not snapshot:
        st.info(
            "Generate monitoring snapshot using "
            "POST /api/monitoring/ptf/snapshot or the CLI."
        )
        st.code(
            "docker compose exec api python scripts/build_monitoring_snapshot.py",
            language="bash",
        )
        return

    status = str(snapshot.get("status", "—"))
    status_method = {
        "HEALTHY": st.success,
        "WARNING": st.warning,
        "CRITICAL": st.error,
    }.get(status, st.info)
    status_method(f"Overall monitoring status: {status}")

    first = st.columns(4)
    first[0].metric("Created", readable_timestamp(snapshot.get("created_at")))
    first[1].metric(
        "Latest PTF",
        readable_timestamp((snapshot.get("data_freshness") or {}).get("max_timestamp")),
    )
    first[2].metric(
        "Pipeline",
        str((snapshot.get("pipeline_health") or {}).get("latest_status", "—")),
    )
    first[3].metric(
        "Forecast rows",
        format_metric_value(
            (snapshot.get("forecast_health") or {}).get("latest_rows"),
            decimals=0,
        ),
    )

    second = st.columns(4)
    second[0].metric(
        "Model R²",
        format_metric_value((snapshot.get("model_quality") or {}).get("r2"), decimals=3),
    )
    second[1].metric(
        "Model MAE",
        format_metric_value((snapshot.get("model_quality") or {}).get("mae")),
    )
    second[2].metric(
        "95% coverage",
        format_metric_value(
            (snapshot.get("uncertainty_quality") or {}).get("interval_coverage_95"),
            suffix="%",
        ),
    )
    second[3].metric(
        "High-risk hours",
        format_metric_value(
            (snapshot.get("risk_summary") or {}).get("high_risk_hours"),
            decimals=0,
        ),
    )

    section_rows = []
    for section_name in [
        "data_freshness",
        "data_quality",
        "pipeline_health",
        "forecast_health",
        "model_quality",
        "uncertainty_quality",
        "risk_summary",
    ]:
        section = snapshot.get(section_name) or {}
        section_rows.append(
            {
                "section": section_name,
                "status": section.get("status"),
                "warnings": len(section.get("warnings") or []),
                "errors": len(section.get("errors") or []),
            }
        )
    st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

    if snapshot.get("warnings"):
        with st.expander("Monitoring warnings"):
            st.json(snapshot.get("warnings"))
    if snapshot.get("errors"):
        with st.expander("Monitoring errors"):
            st.json(snapshot.get("errors"))


def show_pipeline_status_section() -> None:
    st.subheader("Pipeline Status")
    try:
        pipeline_run = cached_latest_pipeline_run()
    except Exception as exc:
        st.error("Could not load latest daily forecast pipeline status.")
        st.exception(exc)
        return

    if not pipeline_run:
        st.info(
            "No daily forecast pipeline run found yet. Run it from the API or CLI."
        )
        st.code(
            "docker compose exec api python scripts/run_daily_forecast_pipeline.py --skip-ingestion --skip-feature-build",
            language="bash",
        )
        return

    first = st.columns(4)
    first[0].metric("Status", str(pipeline_run.get("status", "—")))
    first[1].metric("Target date", str(pipeline_run.get("target_date", "—")))
    first[2].metric("Started", readable_timestamp(pipeline_run.get("started_at")))
    first[3].metric("Finished", readable_timestamp(pipeline_run.get("finished_at")))

    st.write(f"Forecast run: `{pipeline_run.get('forecast_run_id') or '—'}`")
    steps = pipeline_run.get("steps") or {}
    if steps:
        step_rows = [
            {
                "step": step_name,
                "status": step_payload.get("status"),
                "details": {
                    key: value
                    for key, value in step_payload.items()
                    if key not in {"status", "warnings", "errors"}
                },
            }
            for step_name, step_payload in steps.items()
            if isinstance(step_payload, dict)
        ]
        st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)

    warnings = pipeline_run.get("warnings") or []
    errors = pipeline_run.get("errors") or []
    if warnings:
        with st.expander("Pipeline warnings"):
            st.json(warnings)
    if errors:
        with st.expander("Pipeline errors"):
            st.json(errors)


def show_day_ahead_forecast_section() -> None:
    st.subheader("Day-ahead Forecast")
    try:
        summary, forecast_rows = cached_latest_day_ahead_forecast()
    except Exception as exc:
        st.error("Could not load latest day-ahead forecast.")
        st.exception(exc)
        return

    if summary is None or forecast_rows.empty:
        st.info(
            "Generate a day-ahead forecast using "
            "POST /api/forecasts/ptf/day-ahead/generate or the CLI script."
        )
        st.code(
            "docker compose exec api python scripts/generate_day_ahead_ptf.py",
            language="bash",
        )
        return

    risk_counts = summary.get("risk_level_counts", {})
    first = st.columns(4)
    first[0].metric("Target date", str(summary.get("target_date", "—")))
    first[1].metric("Generated at", readable_timestamp(summary.get("generated_at")))
    first[2].metric("Mean forecast", format_metric_value(summary.get("mean_forecast")))
    first[3].metric(
        "Mean interval width",
        format_metric_value(summary.get("mean_interval_width")),
    )

    second = st.columns(4)
    second[0].metric("Min forecast", format_metric_value(summary.get("min_forecast")))
    second[1].metric("Max forecast", format_metric_value(summary.get("max_forecast")))
    second[2].metric(
        "Risk counts",
        f"L {risk_counts.get('LOW', 0)} / "
        f"M {risk_counts.get('MEDIUM', 0)} / "
        f"H {risk_counts.get('HIGH', 0)}",
    )
    second[3].metric("Selected model", str(summary.get("selected_model", "xgboost")))

    st.plotly_chart(
        create_day_ahead_forecast_figure(forecast_rows),
        use_container_width=True,
    )
    st.dataframe(
        prepare_day_ahead_table(forecast_rows),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("⚡ EPİAŞ PTF Forecast Dashboard")
    st.caption("Production-ready point forecast with uncertainty and risk bands")
    show_monitoring_quality_section()
    st.divider()
    show_pipeline_status_section()
    st.divider()
    show_day_ahead_forecast_section()
    st.divider()

    try:
        runs = cached_decision_runs()
    except Exception as exc:
        st.error(
            "Could not connect to PostgreSQL or load forecast decision runs. "
            "Make sure the Docker Compose stack is running and migration 008 has run."
        )
        st.exception(exc)
        return

    if runs.empty:
        st.warning(
            "No forecast decision runs found yet. Generate data by running "
            "XGBoost training, GPR residual modeling, then the forecast decision layer."
        )
        st.code(
            "docker compose exec api python scripts/run_forecast_decision_ptf.py",
            language="bash",
        )
        return

    run_labels = [
        f"{row.decision_run_id[:8]} · {row.selected_model} · "
        f"{readable_timestamp(row.created_at)}"
        for row in runs.itertuples()
    ]

    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

        selected_label = st.selectbox("Decision run", options=run_labels, index=0)
        selected_index = run_labels.index(selected_label)
        decision_run_id = str(runs.iloc[selected_index]["decision_run_id"])

        metrics = cached_decision_metrics(decision_run_id)
        if metrics is None:
            st.error("Selected decision run has no metrics.")
            return

        evaluation_start = to_istanbul_time(metrics.get("evaluation_start"))
        evaluation_end = to_istanbul_time(metrics.get("evaluation_end"))
        default_end = evaluation_end or pd.Timestamp.now(tz="Europe/Istanbul")
        default_start = max(
            evaluation_start or default_end - timedelta(days=7),
            default_end - timedelta(days=7),
        )
        date_range = st.date_input(
            "Date range",
            value=(default_start.date(), default_end.date()),
            min_value=(
                evaluation_start.date() if evaluation_start is not None else None
            ),
            max_value=(evaluation_end.date() if evaluation_end is not None else None),
        )
        risk_levels = st.multiselect(
            "Risk levels",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
        )
        max_rows = st.slider(
            "Max rows / chart range",
            min_value=24,
            max_value=5000,
            value=168,
            step=24,
            help="Default is latest 7 days / 168 hourly rows.",
        )

    start_date = None
    end_date = None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range

    try:
        predictions = cached_predictions(
            decision_run_id,
            start_date,
            end_date,
            tuple(risk_levels),
            max_rows,
        )
    except Exception as exc:
        st.error("Could not load forecast decision predictions for the selected run.")
        st.exception(exc)
        return

    st.subheader("Latest decision metrics")
    show_metric_cards(metrics)

    st.subheader("Model decision")
    st.info(
        "The dashboard uses the selected_prediction from the forecast decision "
        "layer. In the current run, XGBoost is selected as the point forecast, "
        "while GPR provides uncertainty intervals and risk levels."
    )
    st.write(f"Selected model: `{metrics.get('selected_model', '—')}`")
    st.write(metrics.get("selection_reason") or "No selection reason recorded.")

    comparison_columns = st.columns(2)
    with comparison_columns[0]:
        st.markdown("**XGBoost comparison**")
        st.json(metrics.get("xgboost_comparison") or {})
    with comparison_columns[1]:
        st.markdown("**GPR comparison**")
        st.json(metrics.get("gpr_comparison") or {})

    if predictions.empty:
        st.warning("No predictions match the selected filters.")
        return

    st.subheader("Forecast vs actual with 95% confidence interval")
    st.plotly_chart(create_forecast_figure(predictions), use_container_width=True)

    st.subheader("Risk diagnostics")
    risk_columns = st.columns(3)
    risk_columns[0].plotly_chart(
        create_risk_distribution_figure(predictions),
        use_container_width=True,
    )
    risk_columns[1].plotly_chart(
        create_risk_error_figure(predictions),
        use_container_width=True,
    )
    risk_columns[2].plotly_chart(
        create_interval_width_figure(predictions),
        use_container_width=True,
    )

    st.subheader("Recent forecast rows")
    st.dataframe(
        prepare_prediction_table(predictions.tail(min(200, len(predictions)))),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
