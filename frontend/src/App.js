import React, { useState } from "react";
import { useDetectorSocket } from "./useDetectorSocket";
import StatusBar        from "./components/StatusBar";
import CameraFeed       from "./components/CameraFeed";
import AlertPanel       from "./components/AlertPanel";
import StatsPanel       from "./components/StatsPanel";
import ConfidenceChart  from "./components/ConfidenceChart";
import SettingsPanel    from "./components/SettingsPanel";
import VideoUpload      from "./components/VideoUpload";
import VideoProgressBar from "./components/VideoProgressBar";

const tabBtn = (active) => ({
  padding: "8px 22px", fontSize: 13, fontWeight: active ? 600 : 400,
  cursor: "pointer", borderRadius: 10, border: "none",
  background: active
    ? "linear-gradient(135deg, #3b5bdb, #5b4bdb)"
    : "rgba(255,255,255,0.04)",
  color: active ? "#fff" : "#6b7280",
  transition: "all .2s ease",
  letterSpacing: 0.3,
  boxShadow: active ? "0 2px 12px rgba(59,91,219,0.3)" : "none",
});

export default function App() {
  const {
    connected, engineOnline, alerts, latestFrame,
    cameraAlert, videoProgress, videoActive,
  } = useDetectorSocket();

  // Default to screen capture mode
  const [inputMode, setInputMode] = useState("screen");

  const switchToScreen = async () => {
    await fetch("http://localhost:3001/use-screen", { method: "POST" });
    setInputMode("screen");
  };

  return (
    <div style={styles.appContainer}>
      <style>{globalCSS}</style>

      <StatusBar
        connected={connected}
        engineOnline={engineOnline}
        alertCount={alerts.filter(a => a.type !== "camera_alert").length}
      />

      <div style={styles.mainContent}>

        <StatsPanel alerts={alerts} />

        {/* Input mode toggle */}
        <div style={styles.tabContainer}>
          <button style={tabBtn(inputMode === "screen")}
            onClick={() => { setInputMode("screen"); switchToScreen(); }}>
            🖥️ Screen Capture
          </button>
          <button style={tabBtn(inputMode === "video")}
            onClick={() => setInputMode("video")}>
            📁 Upload Video File
          </button>
        </div>

        {/* Video upload panel */}
        {inputMode === "video" && !videoActive && (
          <VideoUpload
            onVideoStart={(filename) => console.log("Processing:", filename)}
            onVideoStop={() => console.log("Video stopped")}
          />
        )}

        {/* Video progress bar */}
        <VideoProgressBar videoProgress={videoProgress} videoActive={videoActive} />

        <SettingsPanel />

        <ConfidenceChart alerts={alerts} />

        <div style={styles.grid}>
          <CameraFeed
            latestFrame={latestFrame}
            cameraAlert={cameraAlert}
            engineOnline={engineOnline}
            mode={inputMode}
            alerts={alerts}
          />
          <AlertPanel alerts={alerts} />
        </div>

      </div>
    </div>
  );
}

/* ── Global CSS injected into the page ─────────────────────────────────── */

const globalCSS = `
  @keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  @keyframes subtlePulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.7; }
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
`;

/* ── Layout Styles ─────────────────────────────────────────────────────── */

const styles = {
  appContainer: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #0f1117 0%, #12141f 50%, #0f1117 100%)",
    backgroundSize: "100% 200%",
  },
  mainContent: {
    padding: "20px 28px 40px",
    maxWidth: 1400,
    marginLeft: "auto",
    marginRight: "auto",
  },
  tabContainer: {
    display: "flex", gap: 8, marginBottom: 18,
    background: "rgba(26,29,39,0.8)", padding: 6, borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.06)", width: "fit-content",
    backdropFilter: "blur(8px)",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 400px",
    gap: 18,
  },
};
