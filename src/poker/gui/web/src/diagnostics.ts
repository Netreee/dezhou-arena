export function reportDiagnostic(event: "gui.render_failed" | "gui.request_failed", message: string): void {
  if (typeof window === "undefined") return;
  void globalThis.fetch("/api/gui/diagnostics", {
    method: "POST", keepalive: true, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event, message: message.slice(0, 512) }),
  }).catch(() => undefined);
}
