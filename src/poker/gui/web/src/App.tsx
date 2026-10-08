import { Alert, Box, Button, Chip, Container, Stack, TextField, Typography } from "@mui/material";
import { useEffect, useMemo, useState } from "react";
import { ActionBar } from "./components/ActionBar";
import { PokerTable } from "./components/PokerTable";
import { GuiController } from "./controller";
import type { GuiState } from "./controller";
import type { HandPhase } from "./models";
import { HttpCliGuiPort } from "./httpPort";

const phaseLabels: Readonly<Record<HandPhase, string>> = {
  setup: "准备发牌", preflop: "翻牌前", flop: "翻牌", turn: "转牌", river: "河牌",
  showdown: "摊牌", settlement: "结算", complete: "本手结束",
};

export function App() {
  const [state, setState] = useState<GuiState>({ snapshot: null, busy: false, issue: null });
  const [name, setName] = useState("");
  const [tableId, setTableId] = useState("");
  const controller = useMemo(() => new GuiController(new HttpCliGuiPort(), setState), []);
  useEffect(() => {
    const timer = window.setInterval(() => { void controller.refresh(); }, 1000);
    const leave = () => controller.abandon();
    window.addEventListener("pagehide", leave);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("pagehide", leave);
      controller.abandon();
    };
  }, [controller]);
  const view = state.snapshot?.view ?? null;
  const pending = state.snapshot?.command_pending ?? false;
  return <Container maxWidth="lg" className="app-shell">
    <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ justifyContent: "space-between", mb: 3 }}>
      <Box><Typography variant="overline">LOCAL HOLD’EM</Typography><Typography variant="h4">德州扑克</Typography></Box>
      <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ alignItems: { xs: "stretch", sm: "center" } }}>
        <Chip label={view === null ? "尚未入座" : view.phase === null ? "等待开局" : phaseLabels[view.phase]} variant="outlined" />
        {view !== null && <Chip label={`桌号 ${view.table_id}`} variant="outlined" />}
        {state.snapshot === null ? <>
          <TextField label="桌号（留空默认桌）" size="small" value={tableId} onChange={(event) => setTableId(event.target.value)}
            slotProps={{ htmlInput: { maxLength: 64 } }} />
          <TextField label="玩家姓名" size="small" value={name} onChange={(event) => setName(event.target.value)} />
          <Button variant="contained" disabled={state.busy || name.trim().length === 0} onClick={() => { void controller.connect(name.trim(), tableId.trim() || undefined); }}>入座</Button>
        </> : <Button disabled={state.busy} onClick={() => { void controller.disconnect(); }}>离开界面</Button>}
      </Stack>
    </Stack>
    {state.snapshot === null && <Alert severity="info" sx={{ mb: 3 }}>输入姓名入座；所有游戏操作由服务端确认。</Alert>}
    {state.snapshot?.status === "opening" && <Alert severity="info" sx={{ mb: 3 }}>正在连接牌桌并等待入座确认。</Alert>}
    {state.snapshot?.status === "closed" && <Alert severity="warning" sx={{ mb: 3 }}>连接已结束，请离开界面后重新入座。</Alert>}
    {state.issue !== null && <Alert severity="error" sx={{ mb: 2 }}>{state.issue}</Alert>}
    {state.snapshot?.error && <Alert severity="error" sx={{ mb: 2 }}>{state.snapshot.error.message}</Alert>}
    <PokerTable view={view} />
    <Box sx={{ mt: 3 }}><ActionBar options={view?.me.legal_actions ?? []} disabled={state.busy || pending} send={(line) => { void controller.submit(line); }} /></Box>
    {pending && <Typography sx={{ mt: 1 }} color="text.secondary">命令已排入 CLI，等待服务器响应。</Typography>}
    {view !== null && (view.phase === null || view.phase === "complete") && <Button sx={{ mt: 2 }} variant="outlined" disabled={state.busy || pending} onClick={() => { void controller.submit("start"); }}>开始下一手</Button>}
  </Container>;
}
