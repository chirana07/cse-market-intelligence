from __future__ import annotations

from src.cse_announcements import CSEAnnouncementsClient
from src.services import RetryConfig, run_with_retry
from src.tools.schemas import DisclosureFetchInput, DisclosureFetchOutput, ToolError


def fetch_disclosures(
    input_data: DisclosureFetchInput,
    client: CSEAnnouncementsClient | None = None,
) -> DisclosureFetchOutput | ToolError:
    client = client or CSEAnnouncementsClient()
    result = run_with_retry(
        lambda: client.fetch_announcements(input_data.category),
        RetryConfig(attempts=2, backoff_sec=0.5),
    )

    if result.ok and result.value is not None:
        df = result.value
        rows = df.head(input_data.limit).to_dict(orient="records") if not df.empty else []
        return DisclosureFetchOutput(
            rows=rows,
            row_count=len(rows),
            attempts=result.attempts,
            elapsed_sec=result.elapsed_sec,
        )

    exc = result.error or RuntimeError("Unknown disclosure fetch error")
    return ToolError(
        tool_name="fetch_disclosures",
        error_type=exc.__class__.__name__,
        message=str(exc),
        retryable=True,
        attempts=result.attempts,
        elapsed_sec=result.elapsed_sec,
    )
