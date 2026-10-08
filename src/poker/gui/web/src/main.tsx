import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";
import { reportDiagnostic } from "./diagnostics";

window.addEventListener("error", (event) => reportDiagnostic("gui.render_failed", event.message));
window.addEventListener("unhandledrejection", (event: PromiseRejectionEvent) => {
  reportDiagnostic("gui.render_failed", event.reason instanceof Error ? event.reason.message : "Unhandled browser rejection");
});

const theme = createTheme({
  palette: { mode: "dark", primary: { main: "#68d7ac" }, background: { default: "#0c181b", paper: "#15262a" } },
  typography: { fontFamily: '"Segoe UI", "Microsoft YaHei", sans-serif' },
  shape: { borderRadius: 12 },
});

const root = document.getElementById("root");
if (root === null) throw new Error("Missing root element");
createRoot(root).render(<StrictMode><ThemeProvider theme={theme}><CssBaseline /><App /></ThemeProvider></StrictMode>);
