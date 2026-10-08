import { Box } from "@mui/material";

const suits: Readonly<Record<string, string>> = { c: "♣", d: "♦", h: "♥", s: "♠" };

export function PlayingCard({ code = null, back = false }: { readonly code?: string | null; readonly back?: boolean }) {
  const suit = code === null ? "" : code.slice(1);
  const rank = code === null ? "" : code.slice(0, 1) === "T" ? "10" : code.slice(0, 1);
  const label = back ? "未公开底牌" : code === null ? "未发牌" : `${rank}${suits[suit] ?? "?"}`;
  return (
    <Box className={`playing-card ${back ? "card-back" : code === null ? "card-slot" : "card-face"}`}
      role="img" aria-label={label} sx={{ color: suit === "d" || suit === "h" ? "#b92f43" : "#17252b" }}>
      {back ? <span className="back-mark">♠</span> : code === null ? <span className="slot-mark">·</span> : <>
        <span className="card-corner">{rank}<br />{suits[suit]}</span>
        <span className="card-suit">{suits[suit]}</span>
      </>}
    </Box>
  );
}
