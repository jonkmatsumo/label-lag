import type { AnalyticsResponseMeta } from '../types/api';

export function DataQualityBadge({ meta }: { meta?: AnalyticsResponseMeta }) {
  if (!meta || !meta.partial) {
    return null;
  }

  const label = meta.truncated ? 'Partial (truncated)' : 'Partial data';
  const limitSuffix =
    typeof meta.effective_limit === 'number'
      ? ` | limit ${meta.effective_limit}`
      : '';

  return (
    <span
      className="badge rounded-pill text-bg-warning fw-semibold"
      title={meta.truncated ? 'Data quality: server-side limit reached' : 'Data quality: partial result'}
      data-testid="data-quality-badge"
    >
      {label}
      {limitSuffix}
    </span>
  );
}
