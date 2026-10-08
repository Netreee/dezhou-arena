import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PlayingCard } from "../src/components/PlayingCard";
import { PokerTable } from "../src/components/PokerTable";
import { snapshot } from "./fixtures";

describe("card presentation", () => {
  it("draws rank and suit from an already-visible card code", () => {
    const markup = renderToStaticMarkup(createElement(PlayingCard, { code: "As" }));
    expect(markup).toContain('aria-label="A♠"');
  });

  it("shows opponent backs and the current player's own cards", () => {
    const markup = renderToStaticMarkup(createElement(PokerTable, { view: snapshot().view }));
    expect(markup).toContain('aria-label="A♠"');
    expect(markup).toContain('aria-label="K♦"');
    expect(markup).toContain('aria-label="未公开底牌"');
  });

  it("keeps an unconnected table empty instead of inventing cards", () => {
    const markup = renderToStaticMarkup(createElement(PokerTable, { view: null }));
    expect(markup).toContain("连接后显示玩家座位与筹码");
    expect(markup).not.toContain('aria-label="A♠"');
  });

  it("shows player qualification, payouts and refunds from the safe view", () => {
    const view = snapshot().view;
    if (view === null) throw new Error("Missing fixture");
    const completed = { ...view, phase: "complete" as const,
      players: view.players.map((p) => ({ ...p, status: "all_in" as const })),
      result: { hand_id: view.hand_id ?? "h1", awards: [{ amount: 120, eligible_ids: ["p1"], shares: [{ player_id: "p1", amount: 120 }] }], refunds: [{ player_id: "p2", amount: 20 }] } };
    const markup = renderToStaticMarkup(createElement(PokerTable, { view: completed }));
    expect(markup).toContain("已全下");
    expect(markup).toContain("本手结果");
    expect(markup).toContain("甲 获得 120");
    expect(markup).toContain("退款");
  });
});
