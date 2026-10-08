import type { GuiSnapshot } from "./models";
import type { CliGuiPort } from "./ports";
import { reportDiagnostic } from "./diagnostics";

export interface GuiState {
  readonly snapshot: GuiSnapshot | null;
  readonly busy: boolean;
  readonly issue: string | null;
}

export class GuiController {
  private snapshot: GuiSnapshot | null = null;
  private busy = false;
  private issue: string | null = null;
  private generation = 0;

  constructor(private readonly port: CliGuiPort, private readonly publish: (state: GuiState) => void) {}

  private emit(): void {
    this.publish({ snapshot: this.snapshot, busy: this.busy, issue: this.issue });
  }

  private async execute(operation: () => Promise<void>): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    this.issue = null;
    this.emit();
    try {
      await operation();
    } catch (error: unknown) {
      this.issue = error instanceof Error ? error.message : "操作失败";
      reportDiagnostic("gui.request_failed", this.issue);
    } finally {
      this.busy = false;
      this.emit();
    }
  }

  async connect(name: string, tableId?: string): Promise<void> {
    if (this.snapshot !== null) return;
    const generation = this.generation;
    await this.execute(async () => {
      const opened = await this.port.open(name, tableId);
      if (generation !== this.generation) this.port.abandon(opened.session_id);
      else this.snapshot = opened;
    });
  }

  async refresh(): Promise<void> {
    const current = this.snapshot;
    if (current === null || current.status === "closed") return;
    const generation = this.generation;
    await this.execute(async () => {
      const updated = await this.port.read(current.session_id);
      if (generation === this.generation) this.snapshot = updated;
    });
  }

  async submit(line: string): Promise<void> {
    const current = this.snapshot;
    if (current === null || current.command_pending) return;
    const generation = this.generation;
    await this.execute(async () => {
      await this.port.submit(current.session_id, line);
      // Presentation-only pending flag. Never change a game view or guess its result.
      if (generation === this.generation) this.snapshot = { ...current, command_pending: true };
    });
  }

  async disconnect(): Promise<void> {
    const current = this.snapshot;
    if (current === null) return;
    const generation = this.generation;
    await this.execute(async () => {
      await this.port.close(current.session_id);
      if (generation === this.generation) this.snapshot = null;
    });
  }

  abandon(): void {
    this.generation += 1;
    const current = this.snapshot;
    this.snapshot = null;
    if (current !== null) this.port.abandon(current.session_id);
  }
}
