import { Box, Chip, Paper, Stack, Typography } from "@mui/material";
import type { PlayerView } from "../models";
import { PlayingCard } from "./PlayingCard";

export function PokerTable({ view }: { readonly view: PlayerView | null }) {
  const own = view?.players.find((player) => player.player_id === view.me.player_id);
  const statusLabels = { active: "在局", folded: "已弃牌", all_in: "已全下" };
  const nameOf = (id: string) => view?.players.find((player) => player.player_id === id)?.name ?? id;
  return <Stack spacing={3}>
    <Box className="seat-grid">
      {view?.players.map((player) => <Paper key={player.player_id} className="seat" variant="outlined">
        <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
          <Typography sx={{ fontWeight: 700 }}>{player.name}{player.player_id === view.me.player_id ? " · 你" : ""}</Typography>
          <Chip size="small" label={`座位 ${player.seat} · ${player.status === null ? "未参局" : statusLabels[player.status]}`} />
          {player.player_id === view.actor_id && <Chip label="行动中" color="warning" size="small" />}
        </Stack>
        <Typography variant="body2" color="text.secondary">筹码 {player.stack} · 本街投入 {player.street_commit}</Typography>
        <Stack direction="row" spacing={0.5} className="seat-cards">
          {player.revealed_cards.length > 0
            ? player.revealed_cards.map((code, index) => <PlayingCard key={index} code={code} />)
            : player.status !== null && player.player_id !== view.me.player_id
              ? <><PlayingCard back /><PlayingCard back /></> : null}
        </Stack>
      </Paper>)}
      {view === null && <Typography color="text.secondary">连接后显示玩家座位与筹码</Typography>}
    </Box>
    <Box className="table-felt">
      <Typography variant="overline">公共牌</Typography>
      <Stack direction="row" spacing={1.5} sx={{ justifyContent: "center" }} className="board-cards">
        {Array.from({ length: 5 }, (_, index) => <PlayingCard key={index} code={view?.board[index] ?? null} />)}
      </Stack>
      <Stack direction="row" spacing={2} sx={{ justifyContent: "center", mt: 3 }}>
        <Chip label={`底池 ${view?.pot_total ?? 0}`} color="primary" />
        <Chip label={`当前下注 ${view?.current_bet ?? 0}`} variant="outlined" />
      </Stack>
    </Box>
    <Paper className="own-hand" variant="outlined">
      <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
        <Box><Typography variant="h6">我的底牌</Typography><Typography color="text.secondary">剩余筹码 {own?.stack ?? "—"}</Typography></Box>
        <Stack direction="row" spacing={1}>
          {Array.from({ length: 2 }, (_, index) => <PlayingCard key={index} code={view?.me.hole_cards[index] ?? null} />)}
        </Stack>
      </Stack>
    </Paper>
    {view?.result && <Paper variant="outlined" sx={{ p: 2 }} data-testid="hand-result">
      <Typography variant="h6">{view.result.hand_id === view.hand_id ? "本手结果" : "上一手结果"}</Typography>
      <Typography variant="caption">手号 {view.result.hand_id}</Typography>
      {view.result.awards.map((award, index) => <Typography key={index}>
        池 {index + 1}：{award.amount} · {award.shares.map((share) => `${nameOf(share.player_id)} 获得 ${share.amount}`).join("，")}
      </Typography>)}
      {view.result.refunds.map((refund, index) => <Typography key={`refund-${index}`}>
        退款：{nameOf(refund.player_id)} {refund.amount}
      </Typography>)}
    </Paper>}
  </Stack>;
}
