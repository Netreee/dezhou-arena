import type { CommandAcceptance, GuiSnapshot } from "./models";
import { CliGuiPort } from "./ports";

export class GuiHttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "GuiHttpError";
  }
}

function errorMessage(body: unknown): string {
  if (typeof body === "object" && body !== null && "error" in body) {
    const error: unknown = body.error;
    if (typeof error === "object" && error !== null && "message" in error && typeof error.message === "string") {
      return error.message;
    }
  }
  return "图形客户端接口请求失败";
}

export class HttpCliGuiPort extends CliGuiPort {
  constructor(
    private readonly base = "/api/gui",
    private readonly request: typeof fetch = (input, init) => globalThis.fetch(input, init),
  ) {
    super();
  }

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.request(`${this.base}${path}`, init);
    const body: unknown = await response.json();
    if (!response.ok) throw new GuiHttpError(response.status, errorMessage(body));
    // The Python response model owns validation; this is the typed HTTP boundary.
    return body as T;
  }

  open(name: string, tableId?: string): Promise<GuiSnapshot> {
    return this.json("/sessions", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, ...(tableId ? { table_id: tableId } : {}) }),
    });
  }

  read(sessionId: string): Promise<GuiSnapshot> {
    return this.json(`/sessions/${encodeURIComponent(sessionId)}`);
  }

  submit(sessionId: string, line: string): Promise<CommandAcceptance> {
    return this.json(`/sessions/${encodeURIComponent(sessionId)}/commands`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ line }),
    });
  }

  async close(sessionId: string): Promise<void> {
    const response = await this.request(`${this.base}/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    if (!response.ok) {
      const body: unknown = await response.json();
      throw new GuiHttpError(response.status, errorMessage(body));
    }
  }

  override abandon(sessionId: string): void {
    void this.request(`${this.base}/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE", keepalive: true,
    }).catch(() => undefined);
  }
}
