export interface RetryOptions {
  retries?: number;
  factor?: number;
  minTimeout?: number;
}

export async function retry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const retries = options.retries ?? 3;
  const factor = options.factor ?? 2;
  const minTimeout = options.minTimeout ?? 200;

  let attempt = 0;
  let lastError: unknown;

  while (attempt <= retries) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt === retries) {
        throw error;
      }
      const timeout = minTimeout * factor ** attempt;
      await new Promise((resolve) => setTimeout(resolve, timeout));
      attempt += 1;
    }
  }

  throw lastError;
}

