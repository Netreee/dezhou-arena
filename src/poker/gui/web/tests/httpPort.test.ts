import { describe, expect, it, vi } from "vitest";
import { GuiHttpError, HttpCliGuiPort } from "../src/httpPort";

describe("local HTTP adapter", () => {
  it("includes the selected table in the join request", async () => {
    let body: BodyInit | null | undefined;
    const request: typeof fetch = async (_input, init) => {
      body = init?.body;
      return new Response(JSON.stringify({ session_id: "gui-1", status: "opening" }), { status: 201 });
    };
    await new HttpCliGuiPort("/api/gui", request).open("甲", "1002");
    expect(body).toBe(JSON.stringify({ name: "甲", table_id: "1002" }));
  });
  it("keeps the global fetch receiver instead of rebinding it to the adapter", async () => {
    const request: typeof fetch = async function (this: typeof globalThis) {
      expect(this).toBe(globalThis);
      return new Response(JSON.stringify({ session_id: "gui-1", stage: "queued" }), { status: 202 });
    };
    vi.stubGlobal("fetch", request);
    try {
      await new HttpCliGuiPort().submit("gui-1", "call");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("sends CLI text and does not add a game player identity", async () => {
    let body: BodyInit | null | undefined;
    const request: typeof fetch = async (_input, init) => {
      body = init?.body;
      return new Response(JSON.stringify({ session_id: "gui-1", stage: "queued" }), { status: 202 });
    };
    await new HttpCliGuiPort("/api/gui", request).submit("gui-1", "call");
    expect(body).toBe(JSON.stringify({ line: "call" }));
  });

  it("surfaces a rejected HTTP request", async () => {
    const request: typeof fetch = async () => new Response(
      JSON.stringify({ error: { code: "gui_command_pending", message: "等待当前命令确认" } }), { status: 409 },
    );
    const port = new HttpCliGuiPort("/api/gui", request);
    await expect(port.open("甲")).rejects.toThrow(GuiHttpError);
    await expect(port.open("甲")).rejects.toThrow("等待当前命令确认");
  });

  it("uses keepalive DELETE when the page leaves", () => {
    const calls: RequestInit[] = [];
    const request: typeof fetch = async (_input, init) => {
      if (init !== undefined) calls.push(init);
      return new Response(null, {status:204});
    };
    new HttpCliGuiPort("/api/gui",request).abandon("opaque");
    expect(calls).toEqual([{method:"DELETE",keepalive:true}]);
  });
});
