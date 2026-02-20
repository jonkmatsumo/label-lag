import { describe, it, expect } from 'vitest';
import { parseInt64, timestampToIso } from '../src/utils/protojson.js';

describe('protojson helpers', () => {
    describe('parseInt64', () => {
        it('parses numeric strings into numbers', () => {
            expect(parseInt64('123')).toBe(123);
            expect(parseInt64('-456')).toBe(-456);
            expect(parseInt64('0')).toBe(0);
        });

        it('passes through numbers as-is', () => {
            expect(parseInt64(123)).toBe(123);
            expect(parseInt64(0)).toBe(0);
        });

        it('returns undefined for null/undefined', () => {
            expect(parseInt64(null)).toBeUndefined();
            expect(parseInt64(undefined)).toBeUndefined();
        });

        it('throws on non-numeric strings', () => {
            expect(() => parseInt64('abc')).toThrow();
            expect(() => parseInt64('')).toThrow(); // Number('') is 0, but we want it to throw if it's supposed to be an int64
        });

        it('throws on non-finite values', () => {
            expect(() => parseInt64(Infinity)).toThrow();
            expect(() => parseInt64(NaN)).toThrow();
        });
    });

    describe('timestampToIso', () => {
        it('passes through ISO strings', () => {
            const iso = '2024-02-20T00:00:00.000Z';
            expect(timestampToIso(iso)).toBe(iso);
        });

        it('converts protojson timestamp objects', () => {
            const ts = { seconds: '1708387200', nanos: 500000000 }; // 2024-02-20T00:00:00.500Z
            const result = timestampToIso(ts);
            expect(result).toBe('2024-02-20T00:00:00.500Z');
        });

        it('handles numeric seconds', () => {
            const ts = { seconds: 1708387200 };
            expect(timestampToIso(ts)).toBe('2024-02-20T00:00:00.000Z');
        });

        it('returns undefined for null/undefined/empty', () => {
            expect(timestampToIso(null)).toBeUndefined();
            expect(timestampToIso(undefined)).toBeUndefined();
        });
    });
});
