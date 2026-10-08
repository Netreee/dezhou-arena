import { describe, expect, it } from "vitest";
import { toCliLine } from "../src/commands";
import type { ActionOption } from "../src/models";

describe("CLI button translation", () => {
  const raise: ActionOption = { kind: "raise_to", pay: null, min_to: 100, max_to: 980 };

  it("sends call without inventing a client-side payment", () => {
    expect(toCliLine({ kind: "call", pay: 40, min_to: null, max_to: null })).toBe("call");
  });

  it("preserves the total semantic of raise_to", () => {
    expect(toCliLine(raise, 100)).toBe("raise_to 100");
  });

  it("rejects noninteger and out-of-range input using only server bounds", () => {
    for (const total of [undefined, 90, 981, 100.5]) expect(() => toCliLine(raise, total)).toThrow();
  });
});
