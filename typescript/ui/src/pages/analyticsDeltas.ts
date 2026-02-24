export type DeltaFormatOptions = {
  metricFormatter: (value: number) => string;
  deltaFormatter?: (value: number) => string;
};

export function formatKpiDelta(
  current: number | undefined,
  previous: number | undefined,
  options: DeltaFormatOptions
): string | undefined {
  if (previous === undefined || current === undefined) {
    return undefined;
  }
  const absoluteDelta = current - previous;
  const absoluteLabel = (options.deltaFormatter ?? options.metricFormatter)(absoluteDelta);
  const percentLabel =
    previous === 0
      ? absoluteDelta === 0
        ? '0.0%'
        : 'n/a'
      : `${((absoluteDelta / Math.abs(previous)) * 100).toFixed(1)}%`;
  const direction = absoluteDelta > 0 ? '+' : '';
  return `vs previous: ${direction}${absoluteLabel} (${direction}${percentLabel})`;
}

export function getDeltaTone(
  current: number | undefined,
  previous: number | undefined
): 'positive' | 'negative' | 'neutral' {
  if (previous === undefined || current === undefined || current === previous) {
    return 'neutral';
  }
  return current > previous ? 'positive' : 'negative';
}
