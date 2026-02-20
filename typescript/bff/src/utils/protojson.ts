/**
 * Normalization helpers for protojson hazards
 */

/**
 * Parses all int64-backed numeric fields that arrive as strings into JS numbers.
 * Protojson serializes int64 as strings by default to avoid precision loss in JS.
 * For analytics KPIs/Volume, these values are safe to handle as JS numbers.
 */
export function parseInt64(value: unknown): number | undefined {
    if (value === null || value === undefined) {
        return undefined;
    }

    if (typeof value === 'string' && value.trim() === '') {
        throw new Error(`Failed to parse int64 value from empty string. Expected a numeric string.`);
    }

    const num = Number(value);
    if (!Number.isFinite(num)) {
        throw new Error(`Failed to parse int64 value: ${value}. Expected a finite numeric string or number.`);
    }

    return num;
}

/**
 * Converts google.protobuf.Timestamp to ISO8601 strings.
 * Supports both already-ISO strings and protojson objects: { seconds: string|number, nanos?: number }
 */
export function timestampToIso(ts: unknown): string | undefined {
    if (ts === null || ts === undefined) {
        return undefined;
    }

    if (typeof ts === 'string') {
        // Already an ISO string (likely from Go custom JSON marshaller or already processed)
        return ts;
    }

    if (typeof ts === 'object') {
        const t = ts as Record<string, unknown>;
        if ('seconds' in t) {
            const seconds = Number(t.seconds);
            const nanos = Number(t.nanos ?? 0);

            if (Number.isFinite(seconds) && Number.isFinite(nanos)) {
                return new Date(seconds * 1000 + nanos / 1e6).toISOString();
            }
        }
    }

    // If we can't parse it, return as is (might be empty string or other)
    return undefined;
}
