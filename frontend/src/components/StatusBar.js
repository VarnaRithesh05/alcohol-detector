import React from "react";

const dotStyle = (on) => ({
  display: "inline-block", width: 8, height: 8, borderRadius: "50%",
  background: on ? "#22c55e" : "#ef4444", marginRight: 6,
  boxShadow: on ? "0 0 8px rgba(34,197,94,0.5)" : "0 0 8px rgba(239,68,68,0.5)",
  animation: on ? "subtlePulse 2s ease-in-out infinite" : "none",
});

export default function StatusBar({ connected, engineOnline, alertCount }) {
  return (
    <div style={styles.bar}>
      <style>{`
        @keyframes subtlePulse {
          0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(34,197,94,0.5); }
          50%      { opacity: 0.6; box-shadow: 0 0 16px rgba(34,197,94,0.8); }
        }
      `}</style>
      <div style={styles.brand}>
        <span style={styles.shieldIcon}>🛡️</span>
        <span style={styles.brandText}>Alcohol Detector</span>
        <span style={styles.badge}>AI</span>
      </div>
      <div style={styles.statusGroup}>
        <span style={styles.statusItem}>
          <span style={dotStyle(connected)} />
          {connected ? "Dashboard connected" : "Disconnected"}
        </span>
        <span style={styles.divider} />
        <span style={styles.statusItem}>
          <span style={dotStyle(engineOnline)} />
          {engineOnline ? "Detection engine online" : "Engine offline"}
        </span>
      </div>
      <div style={styles.alertCount}>
        {alertCount > 0 && (
          <span style={styles.alertBadge}>{alertCount}</span>
        )}
        <span>Total alerts: <strong style={{ color: "#e8e6df" }}>{alertCount}</strong></span>
      </div>
    </div>
  );
}

const styles = {
  bar: {
    display: "flex", alignItems: "center", gap: 24,
    padding: "12px 28px",
    background: "linear-gradient(90deg, rgba(26,29,39,0.95), rgba(18,20,31,0.95))",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    backdropFilter: "blur(12px)",
    fontSize: 13,
    position: "sticky", top: 0, zIndex: 100,
  },
  brand: {
    display: "flex", alignItems: "center", gap: 8,
  },
  shieldIcon: {
    fontSize: 20,
  },
  brandText: {
    fontWeight: 700, fontSize: 16, color: "#e8e6df",
    letterSpacing: 0.3,
  },
  badge: {
    fontSize: 9, fontWeight: 800, color: "#3b5bdb",
    background: "rgba(59,91,219,0.15)",
    border: "1px solid rgba(59,91,219,0.3)",
    padding: "2px 6px", borderRadius: 4,
    letterSpacing: 0.8,
  },
  statusGroup: {
    display: "flex", alignItems: "center", gap: 16,
  },
  statusItem: {
    color: "#9ca3af", display: "flex", alignItems: "center",
  },
  divider: {
    width: 1, height: 16, background: "rgba(255,255,255,0.08)",
  },
  alertCount: {
    marginLeft: "auto", color: "#9ca3af",
    display: "flex", alignItems: "center", gap: 8,
  },
  alertBadge: {
    background: "rgba(239,68,68,0.15)",
    color: "#ef4444",
    fontSize: 11, fontWeight: 700,
    padding: "2px 8px", borderRadius: 10,
    border: "1px solid rgba(239,68,68,0.3)",
  },
};
