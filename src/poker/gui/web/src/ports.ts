import type { CommandAcceptance, GuiSnapshot } from "./models";

export abstract class CliGuiPort {
  abstract open(name: string, tableId?: string): Promise<GuiSnapshot>;
  abstract read(sessionId: string): Promise<GuiSnapshot>;
  abstract submit(sessionId: string, line: string): Promise<CommandAcceptance>;
  abstract close(sessionId: string): Promise<void>;
  abandon(sessionId: string): void { void this.close(sessionId); }
}
