import pino from 'pino';
import { HttpClient, RequestOptions, HttpResponse } from './http-client.js';

export interface ShadowOptions<T> {
  primary: RequestOptions;
  shadow: RequestOptions;
  compareBodies?: (primary: T, shadow: T) => boolean;
}

export class ShadowService {
  private logger: pino.Logger;
  private httpClient: HttpClient;

  constructor(logger: pino.Logger, httpClient: HttpClient) {
    this.logger = logger.child({ service: 'shadow-service' });
    this.httpClient = httpClient;
  }

  async executeWithShadow<T>(options: ShadowOptions<T>): Promise<HttpResponse<T>> {
    const { primary, shadow, compareBodies = this.defaultCompare.bind(this) } = options;

    if (!this.httpClient.config.shadowModeEnabled) {
      return this.httpClient.request<T>(primary);
    }

    const [primaryResult, shadowResult] = await Promise.allSettled([
      this.httpClient.request<T>(primary),
      this.httpClient.request<T>(shadow)
    ]);

    const primaryOk = primaryResult.status === 'fulfilled';
    const shadowOk = shadowResult.status === 'fulfilled';

    if (!primaryOk) {
      this.logger.error({
        metric: 'gateway_error_rate',
        target: primary.target,
        requestId: primary.requestId,
        path: primary.path,
        error: primaryResult.reason instanceof Error ? primaryResult.reason.message : String(primaryResult.reason)
      }, 'Primary request failed');
      
      if (shadowOk) {
        this.logger.info({
          metric: 'fallback_rate',
          requestId: primary.requestId,
          path: primary.path
        }, 'Falling back to shadow result due to primary failure');
        return shadowResult.value;
      }
      throw primaryResult.reason;
    }

    if (!shadowOk) {
      this.logger.warn({
        requestId: primary.requestId,
        path: primary.path,
        error: shadowResult.reason instanceof Error ? shadowResult.reason.message : String(shadowResult.reason)
      }, 'Shadow request failed');
      return primaryResult.value;
    }

    // Both succeeded, compare
    const primaryResp = primaryResult.value;
    const shadowResp = shadowResult.value;

    const statusMatch = primaryResp.statusCode === shadowResp.statusCode;
    const bodyMatch = compareBodies(primaryResp.data, shadowResp.data);

    if (!statusMatch || !bodyMatch) {
      this.logger.info({
        metric: 'shadow_mismatch_rate',
        requestId: primary.requestId,
        path: primary.path,
        statusMatch,
        bodyMatch,
        primaryStatus: primaryResp.statusCode,
        shadowStatus: shadowResp.statusCode,
        primaryData: primaryResp.data,
        shadowData: shadowResp.data
      }, 'Shadow mismatch detected, serving shadow response');
      return shadowResp;
    }

    this.logger.debug({ requestId: primary.requestId, path: primary.path }, 'Shadow parity confirmed');
    return primaryResp;
  }

  private defaultCompare(a: any, b: any): boolean {
    return JSON.stringify(this.sortObject(a)) === JSON.stringify(this.sortObject(b));
  }

  private sortObject(obj: any): any {
    if (obj === null || typeof obj !== 'object') {
      return obj;
    }
    if (Array.isArray(obj)) {
      return obj.map(this.sortObject.bind(this));
    }
    return Object.keys(obj)
      .sort()
      .reduce((result: any, key) => {
        result[key] = this.sortObject(obj[key]);
        return result;
      }, {});
  }
}
