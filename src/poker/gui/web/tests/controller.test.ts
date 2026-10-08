import { describe, expect, it } from "vitest";
import { GuiController } from "../src/controller";
import type { GuiState } from "../src/controller";
import { CliGuiPort } from "../src/ports";
import type { CommandAcceptance, GuiSnapshot } from "../src/models";
import { snapshot } from "./fixtures";

class FakeCliGuiPort extends CliGuiPort {
  readonly lines: string[] = [];
  reads = 0;
  closes = 0;
  current = snapshot();
  error: Error | null = null;
  readGate: Promise<GuiSnapshot> | null = null;
  openGate: Promise<GuiSnapshot> | null = null;

  async open(_name: string): Promise<GuiSnapshot> {
    if (this.error !== null) throw this.error;
    return this.openGate ?? this.current;
  }

  async read(_sessionId: string): Promise<GuiSnapshot> {
    this.reads += 1;
    return this.readGate ?? this.current;
  }

  async submit(sessionId: string, line: string): Promise<CommandAcceptance> {
    this.lines.push(line);
    return { session_id: sessionId, stage: "queued" };
  }

  async close(_sessionId: string): Promise<void> { this.closes += 1; }
}

describe("presentation controller", () => {
  it("keeps game data unchanged while a CLI command is pending", async () => {
    const port = new FakeCliGuiPort();
    const states: GuiState[] = [];
    const controller = new GuiController(port, (state) => states.push(state));
    await controller.connect("甲");
    await controller.submit("raise_to 100");
    expect(port.lines).toEqual(["raise_to 100"]);
    expect(states.at(-1)?.snapshot?.command_pending).toBe(true);
    expect(states.at(-1)?.snapshot?.view).toBe(port.current.view);
  });

  it("does not enqueue a second action while waiting for CLI completion", async () => {
    const port = new FakeCliGuiPort();
    const controller = new GuiController(port, () => undefined);
    await controller.connect("甲");
    await controller.submit("call");
    await controller.submit("all_in");
    expect(port.lines).toEqual(["call"]);
  });

  it("refreshes through the port and uses the bridge's new view", async () => {
    const port = new FakeCliGuiPort();
    const states: GuiState[] = [];
    const controller = new GuiController(port, (state) => states.push(state));
    await controller.connect("甲");
    const current = port.current.view;
    if (current === null) throw new Error("Invalid fixture");
    port.current = { ...port.current, view: { ...current, revision: 2 } };
    await controller.refresh();
    expect(states.at(-1)?.snapshot?.view?.revision).toBe(2);
  });

  it("serializes overlapping browser refreshes", async () => {
    const port = new FakeCliGuiPort();
    const controller = new GuiController(port, () => undefined);
    await controller.connect("甲");
    let release: ((value: GuiSnapshot) => void) | undefined;
    port.readGate = new Promise((resolve) => { release = resolve; });
    const first = controller.refresh();
    await controller.refresh();
    expect(port.reads).toBe(1);
    if (release === undefined) throw new Error("Missing promise resolver");
    release(port.current);
    await first;
  });

  it("shows a connection error without creating a session", async () => {
    const port = new FakeCliGuiPort();
    port.error = new Error("连接失败");
    const states: GuiState[] = [];
    await new GuiController(port, (state) => states.push(state)).connect("甲");
    expect(states.at(-1)?.snapshot).toBeNull();
    expect(states.at(-1)?.issue).toBe("连接失败");
  });

  it("closes the selected CLI session through the port", async () => {
    const port = new FakeCliGuiPort();
    const states: GuiState[] = [];
    const controller = new GuiController(port, (state) => states.push(state));
    await controller.connect("甲");
    await controller.disconnect();
    expect(port.closes).toBe(1);
    expect(states.at(-1)?.snapshot).toBeNull();
  });

  it("closes a session that finishes opening after the page leaves", async () => {
    const port = new FakeCliGuiPort();
    const states: GuiState[] = [];
    const controller = new GuiController(port, (state) => states.push(state));
    let release: ((value: GuiSnapshot) => void) | undefined;
    port.openGate = new Promise((resolve) => { release = resolve; });
    const opening = controller.connect("甲");
    controller.abandon();
    if (release === undefined) throw new Error("Missing resolver");
    release(port.current);
    await opening;
    expect(port.closes).toBe(1);
    expect(states.at(-1)?.snapshot).toBeNull();
  });

  it("does not restore a cached session from a late refresh after the page leaves", async () => {
    const port = new FakeCliGuiPort();
    const states: GuiState[] = [];
    const controller = new GuiController(port, (state) => states.push(state));
    await controller.connect("甲");
    let release: ((value: GuiSnapshot) => void) | undefined;
    port.readGate = new Promise((resolve) => { release = resolve; });
    const reading = controller.refresh();
    controller.abandon();
    if (release === undefined) throw new Error("Missing resolver");
    release(port.current);
    await reading;
    expect(states.at(-1)?.snapshot).toBeNull();
    expect(port.closes).toBe(1);
  });
});
