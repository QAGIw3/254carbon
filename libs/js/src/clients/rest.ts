import fetch, { RequestInit } from "node-fetch";

export interface RestClientOptions {
  baseUrl: string;
  timeoutMs?: number;
  maxRetries?: number;
}

export class RestClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(options: RestClientOptions) {
    this.baseUrl = options.baseUrl;
    this.timeoutMs = options.timeoutMs ?? 10000;
    this.maxRetries = options.maxRetries ?? 3;
  }

  async request<T>(path: string, init?: RequestInit): Promise<T> {
    let attempt = 0;
    let lastError: unknown;

    while (attempt < this.maxRetries) {
      attempt += 1;
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
        const response = await fetch(`${this.baseUrl}${path}`, {
          ...init,
          signal: controller.signal,
        });
        clearTimeout(timeout);

        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        return (await response.json()) as T;
      } catch (error) {
        lastError = error;
        if (attempt >= this.maxRetries) {
          throw error;
        }
        const backoffMs = 200 * 2 ** (attempt - 1);
        await new Promise((resolve) => setTimeout(resolve, backoffMs));
      }
    }

    throw lastError;
  }
}

