import type { GuiSnapshot } from "../src/models";

export function snapshot(): GuiSnapshot {
  return {
    session_id: "gui-1", status: "ready", command_pending: false, error: null,
    view: {
      table_id: "test-table", revision: 1, hand_id: "h1", phase: "flop", board: ["Qs", "8h", "2c"],
      button_seat: 0, actor_id: "p1", current_bet: 60, pot_total: 160,
      players: [
        { player_id: "p1", name: "甲", seat: 0, stack: 960, status: "active", street_commit: 20, hand_commit: 40, revealed_cards: [] },
        { player_id: "p2", name: "乙", seat: 1, stack: 920, status: "active", street_commit: 60, hand_commit: 80, revealed_cards: [] },
      ],
      me: {
        player_id: "p1", hole_cards: ["As", "Kd"],
        legal_actions: [
          { kind: "call", pay: 40, min_to: null, max_to: null },
          { kind: "raise_to", pay: null, min_to: 100, max_to: 980 },
        ],
      },
      result: null,
    },
  };
}
