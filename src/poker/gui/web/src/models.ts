// Safe CLI-view contract. There is deliberately no Table, deck or hidden opponent hand.
export type ActionKind = "fold" | "check" | "call" | "bet_to" | "raise_to" | "all_in";
export type HandPhase = "setup" | "preflop" | "flop" | "turn" | "river" | "showdown" | "settlement" | "complete";
export type PlayerStatus = "active" | "folded" | "all_in";

export interface ActionOption {
  readonly kind: ActionKind;
  readonly pay: number | null;
  readonly min_to: number | null;
  readonly max_to: number | null;
}

export interface PublicPlayerView {
  readonly player_id: string;
  readonly name: string;
  readonly seat: number;
  readonly stack: number;
  readonly status: PlayerStatus | null;
  readonly street_commit: number;
  readonly hand_commit: number;
  readonly revealed_cards: readonly string[];
}

export interface ShareView {
  readonly player_id: string;
  readonly amount: number;
}

export interface ResultView {
  readonly hand_id: string;
  readonly awards: readonly {
    readonly amount: number;
    readonly eligible_ids: readonly string[];
    readonly shares: readonly ShareView[];
  }[];
  readonly refunds: readonly ShareView[];
}

export interface TableConfig {
  readonly small_blind: number;
  readonly big_blind: number;
  readonly starting_stack: number;
  readonly max_players: number;
}

export interface PublicActionRecord {
  readonly sequence: number;
  readonly phase: HandPhase;
  readonly player_id: string;
  readonly kind: ActionKind | "small_blind" | "big_blind";
  readonly pay: number;
  /** Resulting total commitment on this street, before any refunds. */
  readonly to: number;
  readonly stack: number;
}

export interface PlayerView {
  readonly table_id: string;
  readonly revision: number;
  readonly hand_id: string | null;
  readonly phase: HandPhase | null;
  readonly board: readonly string[];
  readonly button_seat: number | null;
  readonly actor_id: string | null;
  readonly current_bet: number;
  readonly pot_total: number;
  readonly players: readonly PublicPlayerView[];
  readonly me: {
    readonly player_id: string;
    readonly hole_cards: readonly string[];
    readonly legal_actions: readonly ActionOption[];
  };
  readonly result: ResultView | null;
  // Optional for older CLI peers and saved browser fixtures.
  readonly config?: TableConfig | null;
  readonly action_history?: readonly PublicActionRecord[];
  readonly history_complete?: boolean;
}

export interface GuiSnapshot {
  readonly session_id: string;
  readonly status: "opening" | "ready" | "closed";
  readonly view: PlayerView | null;
  readonly command_pending: boolean;
  readonly error: { readonly code: string; readonly message: string } | null;
}

export interface CommandAcceptance {
  readonly session_id: string;
  readonly stage: "queued";
}
