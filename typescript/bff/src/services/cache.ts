import { Config } from '../config.js';
import pino from 'pino';

interface CacheEntry<T> {
  data: T;
  expiry: number;
}

export class SimpleCache {
  private cache = new Map<string, CacheEntry<unknown>>();
  private inFlight = new Map<string, Promise<unknown>>();
  private config: Config;
  private logger: pino.Logger;

  constructor(config: Config, logger: pino.Logger) {
    this.config = config;
    this.logger = logger.child({ service: 'cache' });
  }

  get<T>(key: string, tenantId: string): T | undefined {
    if (!this.config.cacheEnabled) return undefined;

    const compositeKey = `${tenantId}:${key}`;
    const entry = this.cache.get(compositeKey);
    if (!entry) return undefined;

    if (Date.now() > entry.expiry) {
      this.cache.delete(compositeKey);
      this.logger.debug({ key: compositeKey }, 'Cache miss (expired)');
      return undefined;
    }

    // LRU: Move to end (most recently used)
    this.cache.delete(compositeKey);
    this.cache.set(compositeKey, entry);

    this.logger.debug({ key: compositeKey }, 'Cache hit');
    return entry.data as T;
  }

  set<T>(key: string, data: T, tenantId: string, ttlMs?: number): void {
    if (!this.config.cacheEnabled) return;

    const ttl = ttlMs ?? this.config.cacheTtlMs;
    const expiry = Date.now() + ttl;
    const compositeKey = `${tenantId}:${key}`;

    // LRU: If key exists, delete it first so it moves to end
    if (this.cache.has(compositeKey)) {
      this.cache.delete(compositeKey);
    } else if (this.cache.size >= 1000) {
      // Evict oldest (first in Map)
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
        this.logger.debug({ evictedKey: oldestKey }, 'Cache evicted usage limit');
      }
    }

    this.cache.set(compositeKey, { data, expiry });
  }

  async getOrLoad<T>(
    key: string,
    tenantId: string,
    loader: () => Promise<T>,
    ttlMs?: number
  ): Promise<T> {
    const cached = this.get<T>(key, tenantId);
    if (cached !== undefined) {
      return cached;
    }

    if (!this.config.cacheEnabled) {
      return loader();
    }

    const compositeKey = `${tenantId}:${key}`;
    const inFlight = this.inFlight.get(compositeKey);
    if (inFlight) {
      return inFlight as Promise<T>;
    }

    const loadPromise = loader()
      .then((result) => {
        this.set(key, result, tenantId, ttlMs);
        return result;
      })
      .finally(() => {
        this.inFlight.delete(compositeKey);
      });

    this.inFlight.set(compositeKey, loadPromise);
    return loadPromise;
  }
}
