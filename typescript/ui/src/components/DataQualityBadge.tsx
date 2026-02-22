import type { AnalyticsResponseMeta } from '../types/api';

function formatPartialReason(reason: AnalyticsResponseMeta['partial_reason']): string {
  switch (reason) {
    case 'TIMEOUT':
      return 'Partial (timeout)';
    case 'ROW_LIMIT':
      return 'Partial (row limit)';
    case 'UPSTREAM_ERROR':
      return 'Partial (upstream error)';
    case 'EMPTY':
      return 'No data in range';
    default:
      return 'Partial data';
  }
}

export function DataQualityBadge({ meta }: { meta?: AnalyticsResponseMeta }) {
  if (!meta?.is_partial) {
    return null;
  }

  const label = formatPartialReason(meta.partial_reason);
  const sampleSuffix =
    typeof meta.sample_rate === 'number'
      ? ` \u2022 ${(meta.sample_rate * 100).toFixed(0)}% sample`
      : '';

  return (
    <span
      className="badge rounded-pill text-bg-warning fw-semibold"
      title={`Data quality: ${meta.partial_reason}`}
      data-testid="data-quality-badge"
    >
      {label}
      {sampleSuffix}
    </span>
  );
}
