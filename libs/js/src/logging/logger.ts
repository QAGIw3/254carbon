interface LogPayload {
  message: string;
  level: "debug" | "info" | "warn" | "error";
  context?: Record<string, unknown>;
}

export function log(payload: LogPayload) {
  const entry = {
    ...payload,
    timestamp: new Date().toISOString(),
  };
  const serialized = JSON.stringify(entry);
  switch (payload.level) {
    case "debug":
    case "info":
      console.log(serialized);
      break;
    case "warn":
      console.warn(serialized);
      break;
    case "error":
      console.error(serialized);
      break;
  }
}

