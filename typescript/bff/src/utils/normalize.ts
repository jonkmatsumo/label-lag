/**
 * Normalize JSON objects for stable comparison
 */
export function normalizeJson(obj: any): any {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    // If it's an array of objects with rule_id or id, sort by that for stability
    if (obj.length > 0 && typeof obj[0] === 'object') {
      const first = obj[0];
      if ('rule_id' in first) {
        return obj.map(normalizeJson).sort((a, b) => a.rule_id.localeCompare(b.rule_id));
      }
      if ('id' in first) {
        return obj.map(normalizeJson).sort((a, b) => a.id.localeCompare(b.id));
      }
      if ('record_id' in first) {
        return obj.map(normalizeJson).sort((a, b) => a.record_id.localeCompare(b.record_id));
      }
    }
    return obj.map(normalizeJson);
  }

  return Object.keys(obj)
    .sort()
    .reduce((result: any, key) => {
      // Exclude dynamic fields like timestamps or IDs if needed,
      // but for contract tests we might want to keep them if they are deterministic in mocks
      result[key] = normalizeJson(obj[key]);
      return result;
    }, {});
}
