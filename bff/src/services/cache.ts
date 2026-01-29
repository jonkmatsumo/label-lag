import { Config } from '../config.js';
import pino from 'pino';

interface CacheEntry<T> {
  data: T;
  expiry: number;
}

export class SimpleCache {
  private cache = new Map<string, CacheEntry<unknown>>();
  private config: Config;
  private logger: pino.Logger;

  constructor(config: Config, logger: pino.Logger) {
    this.config = config;
    this.logger = logger.child({ service: 'cache' });
  }

  get<T>(key: string): T | undefined {
    if (!this.config.cacheEnabled) return undefined;

    const entry = this.cache.get(key);
    if (!entry) return undefined;

    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      this.logger.debug({ key }, 'Cache miss (expired)');
      return undefined;
    }

    this.logger.debug({ key }, 'Cache hit');
    return entry.data as T;
  }

  set<T>(key: string, data: T, ttlMs?: number): void {
    if (!this.config.cacheEnabled) return;

    const ttl = ttlMs ?? this.config.cacheTtlMs;
    const expiry = Date.now() + ttl;
    
    // Simple LRU-like safety: if too big, clear it
    if (this.cache.size > 1000) {
      this.cache.clear();
      this.logger.warn('Cache cleared due to size limit');
    }

    this.cache.set(key, { data, expiry });
  }
}
