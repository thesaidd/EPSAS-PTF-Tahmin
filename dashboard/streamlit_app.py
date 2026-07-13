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
    translate_monitoring_section,
    translate_risk_level,
    translate_status,
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
    page_title="EPİAŞ PTF Tahmin Dashboard",
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


def show_metric_with_caption(
    column: Any,
    label: str,
    value: str,
    caption: str,
) -> None:
    column.metric(label, value)
    column.caption(caption)


def show_system_flow_section() -> None:
    st.subheader("Sistem Nasıl Çalışıyor?")
    st.info(
        "Bu MVP, EPİAŞ Şeffaflık Platformu verisinden saatlik PTF tahmini üretir. "
        "XGBoost ana nokta tahmini sağlar; GPR modeli belirsizlik bandı ve risk "
        "seviyesi üretir; karar katmanı ise kullanıcıya gösterilecek güvenilir "
        "tahmin çıktısını seçer."
    )
    st.markdown(
        """
        **Akış:** EPİAŞ → Veri Tabanı → Özellikler → XGBoost → GPR Belirsizlik
        → Karar Katmanı → 24 Saatlik Tahmin → Dashboard
        """
    )
    steps = [
        "EPİAŞ verisi alınır.",
        "Saatlik PTF verisi PostgreSQL/TimescaleDB veritabanına kaydedilir.",
        "Bu veriden takvim, gecikme ve hareketli ortalama özellikleri üretilir.",
        "XGBoost modeli ana PTF tahminini üretir.",
        "GPR modeli tahmin belirsizliğini ve güven aralığını hesaplar.",
        "Karar katmanı, en güvenilir nokta tahmin modelini seçer.",
        "Gün öncesi tahmin çıktısı 24 saatlik PTF beklentisini üretir.",
        "Monitoring katmanı veri, model ve pipeline sağlığını kontrol eder.",
        "Dashboard tüm sonuçları iş kullanıcılarına anlaşılır şekilde gösterir.",
    ]
    st.markdown("\n".join(f"- {step}" for step in steps))


def show_metric_cards(metrics: dict[str, Any]) -> None:
    first = st.columns(4)
    show_metric_with_caption(
        first[0],
        "Seçilen Model",
        str(metrics.get("selected_model", "—")),
        "Karar katmanının nokta tahmin için seçtiği model.",
    )
    show_metric_with_caption(
        first[1],
        "Ortalama Mutlak Hata (MAE)",
        format_metric_value(metrics.get("mae")),
        "Tahminlerin gerçekleşen PTF'den ortalama sapması.",
    )
    show_metric_with_caption(
        first[2],
        "Kök Ortalama Kare Hata (RMSE)",
        format_metric_value(metrics.get("rmse")),
        "Büyük hataları daha güçlü cezalandıran hata metriği.",
    )
    show_metric_with_caption(
        first[3],
        "R² Skoru",
        format_metric_value(metrics.get("r2"), decimals=3),
        "Modelin genel açıklama gücü / uyum skoru.",
    )

    second = st.columns(4)
    show_metric_with_caption(
        second[0],
        "Güven Aralığı Kapsama",
        format_metric_value(metrics.get("interval_coverage_95"), suffix="%"),
        "Gerçek değerlerin tahmin bandı içinde kalma oranı.",
    )
    show_metric_with_caption(
        second[1],
        "Ortalama Bant Genişliği",
        format_metric_value(metrics.get("mean_interval_width")),
        "Belirsizlik bandı genişledikçe piyasa belirsizliği artar.",
    )
    show_metric_with_caption(
        second[2],
        "Kayıt Sayısı",
        format_metric_value(metrics.get("count"), decimals=0),
        "Değerlendirme dönemindeki saatlik tahmin sayısı.",
    )
    show_metric_with_caption(
        second[3],
        "Değerlendirme Dönemi",
        f"{readable_timestamp(metrics.get('evaluation_start'))} → "
        f"{readable_timestamp(metrics.get('evaluation_end'))}",
        "Model performansının ölçüldüğü zaman aralığı.",
    )


def show_monitoring_quality_section() -> None:
    st.subheader("✅ İzleme ve Kalite Kontrol")
    st.caption(
        "Bu bölüm sistemin günlük operasyon için güvenilir olup olmadığını "
        "gösterir."
    )
    st.info(
        "Monitoring katmanı veri tazeliğini, eksik saatleri, pipeline başarısını, "
        "model kalitesini, tahmin sağlığını, güven aralığı kalitesini ve risk "
        "dağılımını kontrol eder."
    )
    try:
        snapshot = cached_latest_monitoring_snapshot()
    except Exception as exc:
        st.error("Son monitoring snapshot kaydı okunamadı.")
        st.exception(exc)
        return

    if not snapshot:
        st.info(
            "Henüz monitoring snapshot bulunamadı. API üzerinden "
            "POST /api/monitoring/ptf/snapshot çağrısı yapılabilir."
        )
        st.code(
            "docker compose exec api python scripts/build_monitoring_snapshot.py",
            language="bash",
        )
        return

    status = str(snapshot.get("status", "—"))
    status_label = translate_status(status)
    status_method = {
        "HEALTHY": st.success,
        "WARNING": st.warning,
        "CRITICAL": st.error,
    }.get(status, st.info)
    status_method(f"Genel sistem durumu: {status_label}")
    st.caption(
        "Sağlıklı: kritik sorun yok. Uyarı: dikkat edilmesi gereken sinyal var. "
        "Kritik: operasyon öncesi incelenmesi gereken ciddi sorun var."
    )

    first = st.columns(4)
    show_metric_with_caption(
        first[0],
        "Snapshot Zamanı",
        readable_timestamp(snapshot.get("created_at")),
        "Bu kalite kontrol özetinin üretildiği zaman.",
    )
    show_metric_with_caption(
        first[1],
        "Son PTF Verisi",
        readable_timestamp((snapshot.get("data_freshness") or {}).get("max_timestamp")),
        "Veritabanındaki en güncel saatlik PTF kaydı.",
    )
    show_metric_with_caption(
        first[2],
        "Son Pipeline Durumu",
        translate_status((snapshot.get("pipeline_health") or {}).get("latest_status")),
        "Günlük tahmin pipeline çalışmasının son sonucu.",
    )
    show_metric_with_caption(
        first[3],
        "Son Tahmin Satırı",
        format_metric_value(
            (snapshot.get("forecast_health") or {}).get("latest_rows"),
            decimals=0,
        ),
        "Son gün öncesi tahmin çıktısındaki saatlik kayıt sayısı.",
    )

    second = st.columns(4)
    show_metric_with_caption(
        second[0],
        "Model Kalitesi (R²)",
        format_metric_value((snapshot.get("model_quality") or {}).get("r2"), decimals=3),
        "Modelin genel açıklama gücü.",
    )
    show_metric_with_caption(
        second[1],
        "Tahmin Hatası (MAE)",
        format_metric_value((snapshot.get("model_quality") or {}).get("mae")),
        "Gerçekleşen PTF'ye göre ortalama tahmin hatası.",
    )
    show_metric_with_caption(
        second[2],
        "Belirsizlik Kalitesi",
        format_metric_value(
            (snapshot.get("uncertainty_quality") or {}).get("interval_coverage_95"),
            suffix="%",
        ),
        "Gerçek değerlerin %95 güven aralığında kalma oranı.",
    )
    show_metric_with_caption(
        second[3],
        "Yüksek Riskli Saat",
        format_metric_value(
            (snapshot.get("risk_summary") or {}).get("high_risk_hours"),
            decimals=0,
        ),
        "Belirsizliği yüksek olan saat sayısı.",
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
                "Kontrol Alanı": translate_monitoring_section(section_name),
                "Durum": translate_status(section.get("status")),
                "Uyarı Sayısı": len(section.get("warnings") or []),
                "Hata Sayısı": len(section.get("errors") or []),
            }
        )
    st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

    if snapshot.get("warnings"):
        with st.expander("Monitoring uyarıları"):
            st.json(snapshot.get("warnings"))
    if snapshot.get("errors"):
        with st.expander("Monitoring hataları"):
            st.json(snapshot.get("errors"))


def show_pipeline_status_section() -> None:
    st.subheader("🛠 Pipeline Durumu")
    st.info(
        "Pipeline; veri hazırlama, tahmin üretimi ve monitoring adımlarının "
        "operasyonel olarak tamamlanıp tamamlanmadığını gösterir."
    )
    try:
        pipeline_run = cached_latest_pipeline_run()
    except Exception as exc:
        st.error("Son günlük tahmin pipeline durumu okunamadı.")
        st.exception(exc)
        return

    if not pipeline_run:
        st.info("Henüz günlük tahmin pipeline çalışması bulunamadı.")
        st.code(
            "docker compose exec api python scripts/run_daily_forecast_pipeline.py --skip-ingestion --skip-feature-build",
            language="bash",
        )
        return

    first = st.columns(4)
    show_metric_with_caption(
        first[0],
        "Durum",
        translate_status(pipeline_run.get("status")),
        "Son pipeline çalışmasının sonucu.",
    )
    show_metric_with_caption(
        first[1],
        "Hedef Tarih",
        str(pipeline_run.get("target_date", "—")),
        "Tahmin üretilen gün.",
    )
    show_metric_with_caption(
        first[2],
        "Başlangıç",
        readable_timestamp(pipeline_run.get("started_at")),
        "Pipeline çalışma başlangıcı.",
    )
    show_metric_with_caption(
        first[3],
        "Bitiş",
        readable_timestamp(pipeline_run.get("finished_at")),
        "Pipeline çalışma bitişi.",
    )

    st.write(f"Tahmin run ID: `{pipeline_run.get('forecast_run_id') or '—'}`")
    steps = pipeline_run.get("steps") or {}
    if steps:
        step_rows = [
            {
                "Adım": step_name,
                "Durum": translate_status(step_payload.get("status")),
                "Detay": {
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
        with st.expander("Pipeline uyarıları"):
            st.json(warnings)
    if errors:
        with st.expander("Pipeline hataları"):
            st.json(errors)


def show_day_ahead_forecast_section() -> None:
    st.subheader("📊 Gün Öncesi PTF Tahmini")
    st.info(
        "Bu bölüm, hedef gün için 24 saatlik PTF beklentisini gösterir. "
        "Tedarik ve portföy yönetimi kararlarında saatlik riskleri görmek için "
        "kullanılabilir."
    )
    try:
        summary, forecast_rows = cached_latest_day_ahead_forecast()
    except Exception as exc:
        st.error("Son gün öncesi tahmin okunamadı.")
        st.exception(exc)
        return

    if summary is None or forecast_rows.empty:
        st.info("Henüz gün öncesi tahmin bulunamadı.")
        st.code(
            "docker compose exec api python scripts/generate_day_ahead_ptf.py",
            language="bash",
        )
        return

    risk_counts = summary.get("risk_level_counts", {})
    high_risk_hours = risk_counts.get("HIGH", 0)
    first = st.columns(4)
    show_metric_with_caption(
        first[0],
        "Hedef Tarih",
        str(summary.get("target_date", "—")),
        "24 saatlik tahminin ait olduğu gün.",
    )
    show_metric_with_caption(
        first[1],
        "Ortalama PTF Tahmini",
        format_metric_value(summary.get("mean_forecast")),
        "Hedef gün için ortalama saatlik PTF beklentisi.",
    )
    show_metric_with_caption(
        first[2],
        "Minimum Tahmin",
        format_metric_value(summary.get("min_forecast")),
        "Hedef gündeki en düşük saatlik PTF tahmini.",
    )
    show_metric_with_caption(
        first[3],
        "Maksimum Tahmin",
        format_metric_value(summary.get("max_forecast")),
        "Hedef gündeki en yüksek saatlik PTF tahmini.",
    )

    second = st.columns(4)
    show_metric_with_caption(
        second[0],
        "Yüksek Riskli Saat",
        format_metric_value(high_risk_hours, decimals=0),
        "Belirsizlik bandı geniş olan saat sayısı.",
    )
    show_metric_with_caption(
        second[1],
        "Ortalama Bant Genişliği",
        format_metric_value(summary.get("mean_interval_width")),
        "Güven aralığı genişledikçe piyasa belirsizliği artar.",
    )
    show_metric_with_caption(
        second[2],
        "Risk Dağılımı",
        f"D {risk_counts.get('LOW', 0)} / O {risk_counts.get('MEDIUM', 0)} / "
        f"Y {risk_counts.get('HIGH', 0)}",
        "Düşük / Orta / Yüksek riskli saat sayısı.",
    )
    show_metric_with_caption(
        second[3],
        "Seçilen Model",
        str(summary.get("selected_model", "xgboost")),
        "Nokta tahmin için kullanılan model.",
    )
    st.info(
        "`PTF Tahmini`, karar katmanının seçtiği ürün çıktısıdır. Alt ve üst "
        "bantlar tahminin beklenen belirsizlik aralığını gösterir. Bant "
        "genişledikçe piyasa belirsizliği artar."
    )
    st.caption(
        "Risk seviyesi: LOW düşük belirsizlik, MEDIUM orta belirsizlik, HIGH "
        "yüksek belirsizlik anlamına gelir."
    )

    st.plotly_chart(
        create_day_ahead_forecast_figure(forecast_rows),
        use_container_width=True,
    )
    st.dataframe(
        prepare_day_ahead_table(forecast_rows),
        use_container_width=True,
        hide_index=True,
    )


def show_forecast_decision_section() -> None:
    try:
        runs = cached_decision_runs()
    except Exception as exc:
        st.error(
            "PostgreSQL bağlantısı kurulamadı veya model karar kayıtları okunamadı."
        )
        st.exception(exc)
        return

    if runs.empty:
        st.warning(
            "Henüz model karar katmanı çalışması bulunamadı. Önce XGBoost, "
            "GPR ve karar katmanı çalıştırılmalıdır."
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
        st.header("Kontroller")
        if st.button("Veriyi Yenile"):
            st.cache_data.clear()
            st.rerun()

        selected_label = st.selectbox("Model karar çalışması", options=run_labels, index=0)
        selected_index = run_labels.index(selected_label)
        decision_run_id = str(runs.iloc[selected_index]["decision_run_id"])

        metrics = cached_decision_metrics(decision_run_id)
        if metrics is None:
            st.error("Seçilen karar çalışmasına ait metrik bulunamadı.")
            return

        evaluation_start = to_istanbul_time(metrics.get("evaluation_start"))
        evaluation_end = to_istanbul_time(metrics.get("evaluation_end"))
        default_end = evaluation_end or pd.Timestamp.now(tz="Europe/Istanbul")
        default_start = max(
            evaluation_start or default_end - timedelta(days=7),
            default_end - timedelta(days=7),
        )
        date_range = st.date_input(
            "Tarih aralığı",
            value=(default_start.date(), default_end.date()),
            min_value=(
                evaluation_start.date() if evaluation_start is not None else None
            ),
            max_value=(evaluation_end.date() if evaluation_end is not None else None),
        )
        risk_levels = st.multiselect(
            "Risk seviyeleri",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
            format_func=translate_risk_level,
        )
        max_rows = st.slider(
            "Maksimum satır / grafik aralığı",
            min_value=24,
            max_value=5000,
            value=168,
            step=24,
            help="Varsayılan değer son 7 gün / 168 saatlik kayıttır.",
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
        st.error("Seçilen karar çalışması için tahmin kayıtları okunamadı.")
        st.exception(exc)
        return

    st.subheader("🧠 Model Karar Katmanı")
    st.info(
        "GPR düzeltilmiş tahmin, aynı test döneminde XGBoost'tan daha iyi sonuç "
        "vermediği için sistem nokta tahmin için XGBoost'u kullanır. Ancak "
        "GPR'nin ürettiği belirsizlik bilgisi güven aralığı ve risk seviyesi "
        "için korunur."
    )
    show_metric_cards(metrics)

    st.markdown("### Neden XGBoost seçildi?")
    st.write(f"Seçilen nokta tahmin modeli: `{metrics.get('selected_model', '—')}`")
    st.write(metrics.get("selection_reason") or "Karar nedeni kaydedilmemiş.")
    st.caption(
        "`selected_prediction`, karar katmanının XGBoost ve GPR düzeltilmiş "
        "tahminleri karşılaştırdıktan sonra ürün çıktısı olarak seçtiği değerdir."
    )

    comparison_columns = st.columns(2)
    with comparison_columns[0]:
        st.markdown("**XGBoost performansı**")
        st.json(metrics.get("xgboost_comparison") or {})
    with comparison_columns[1]:
        st.markdown("**GPR düzeltilmiş tahmin performansı**")
        st.json(metrics.get("gpr_comparison") or {})

    if predictions.empty:
        st.warning("Seçilen filtrelere uygun tahmin kaydı bulunamadı.")
        return

    st.subheader("⚡ PTF Tahmini ve Gerçekleşen Değerler")
    st.info(
        "Bu grafik, seçilen modelin saatlik PTF tahminini ve gerçekleşen "
        "değerlerle karşılaştırmasını gösterir. Güven aralığı, tahminin beklenen "
        "belirsizlik bandıdır."
    )
    st.plotly_chart(create_forecast_figure(predictions), use_container_width=True)

    st.subheader("Risk ve Belirsizlik Analizi")
    st.caption(
        "LOW düşük belirsizlik, MEDIUM orta belirsizlik, HIGH yüksek belirsizlik "
        "anlamına gelir. Yüksek riskli saatlerde portföy ve tedarik kararları "
        "daha dikkatli değerlendirilmelidir."
    )
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

    st.subheader("Son Tahmin Kayıtları")
    st.dataframe(
        prepare_prediction_table(predictions.tail(min(200, len(predictions)))),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("⚡ EPİAŞ PTF Tahmin Dashboard")
    st.caption(
        "Türkiye elektrik piyasası için gün öncesi PTF tahmini, belirsizlik "
        "bandı ve operasyonel kalite kontrol ekranı."
    )
    st.info(
        "Bu dashboard enerji şirketi yöneticileri, portföy yöneticileri ve "
        "tedarik ekipleri için saatlik PTF beklentisini, risk seviyesini ve "
        "sistemin güvenilirlik durumunu anlaşılır şekilde sunar."
    )
    st.markdown(
        "[Swagger API](http://localhost:8000/docs) · "
        "[MLflow](http://localhost:5000) · "
        "[Readiness](http://localhost:8000/api/system/readiness)"
    )

    show_system_flow_section()
    st.divider()
    show_monitoring_quality_section()
    st.divider()
    show_pipeline_status_section()
    st.divider()
    show_day_ahead_forecast_section()
    st.divider()
    show_forecast_decision_section()


if __name__ == "__main__":
    main()
