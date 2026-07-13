import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.daily_forecast_pipeline import DailyForecastPipelineService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    enabled = os.getenv("DAILY_PIPELINE_ENABLED", "false").lower() == "true"
    timezone = ZoneInfo(os.getenv("DAILY_PIPELINE_TIMEZONE", "Europe/Istanbul"))
    run_time = os.getenv("DAILY_PIPELINE_RUN_TIME_LOCAL", "13:00")
    if not enabled:
        logger.info("Daily pipeline scheduler is disabled. Set DAILY_PIPELINE_ENABLED=true.")
        while True:
            time.sleep(3600)

    logger.info("Daily pipeline scheduler enabled for %s %s.", run_time, timezone)
    while True:
        sleep_seconds = _seconds_until_next_run(run_time, timezone)
        logger.info("Sleeping %.0f seconds until next daily forecast run.", sleep_seconds)
        time.sleep(sleep_seconds)
        try:
            summary = DailyForecastPipelineService().run_pipeline()
            logger.info(
                "Daily forecast pipeline finished: run_id=%s status=%s",
                summary.get("pipeline_run_id"),
                summary.get("status"),
            )
        except Exception:
            logger.exception("Scheduled daily forecast pipeline run failed.")
        time.sleep(60)


def _seconds_until_next_run(run_time: str, timezone: ZoneInfo) -> float:
    hour_text, minute_text = run_time.split(":", maxsplit=1)
    now = datetime.now(tz=timezone)
    next_run = now.replace(
        hour=int(hour_text),
        minute=int(minute_text),
        second=0,
        microsecond=0,
    )
    if next_run <= now:
        next_run += timedelta(days=1)
    return max((next_run - now).total_seconds(), 1.0)


if __name__ == "__main__":
    main()
