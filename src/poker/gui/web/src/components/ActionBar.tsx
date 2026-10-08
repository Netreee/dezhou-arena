import { Alert, Button, Paper, Stack, TextField, Typography } from "@mui/material";
import { useState } from "react";
import { toCliLine } from "../commands";
import type { ActionKind, ActionOption } from "../models";

const labels: Readonly<Record<ActionKind, string>> = {
  fold: "弃牌", check: "过牌", call: "跟注", bet_to: "下注至", raise_to: "加注至", all_in: "全下",
};

function ActionControl({ option, disabled, send }: {
  readonly option: ActionOption; readonly disabled: boolean; readonly send: (line: string) => void;
}) {
  const [total, setTotal] = useState(option.min_to ?? 0);
  const [issue, setIssue] = useState<string | null>(null);
  const sized = option.kind === "bet_to" || option.kind === "raise_to";
  const activate = () => {
    try { setIssue(null); send(toCliLine(option, sized ? total : undefined)); }
    catch (error: unknown) { setIssue(error instanceof Error ? error.message : "金额无效"); }
  };
  return <Stack spacing={1}>
    {sized && <TextField size="small" type="number" label="本街总投入" value={total}
      onChange={(event) => setTotal(Number(event.target.value))} disabled={disabled}
      slotProps={{ htmlInput: { min: option.min_to ?? 0, max: option.max_to ?? undefined, step: 1 } }} />}
    <Button variant={option.kind === "fold" ? "outlined" : "contained"} color={option.kind === "fold" ? "inherit" : "primary"}
      disabled={disabled} onClick={activate}>
      {labels[option.kind]}{option.pay !== null ? ` ${option.pay}` : ""}
    </Button>
    {issue !== null && <Alert severity="error">{issue}</Alert>}
  </Stack>;
}

export function ActionBar({ options, disabled, send }: {
  readonly options: readonly ActionOption[]; readonly disabled: boolean; readonly send: (line: string) => void;
}) {
  return <Paper className="action-bar" variant="outlined">
    <Typography sx={{ fontWeight: 700, mb: 2 }}>操作</Typography>
    {options.length === 0 ? <Typography color="text.secondary">轮到你时，服务器允许的操作会显示在这里。</Typography>
      : <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
        {options.map((option) => <ActionControl key={`${option.kind}-${option.min_to}-${option.max_to}`} option={option} disabled={disabled} send={send} />)}
      </Stack>}
  </Paper>;
}
