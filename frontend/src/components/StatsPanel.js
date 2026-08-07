import React, { useMemo } from "react";

function StatCard({ label, value, sub, color, icon }) {
  return (
    <div style={styles.card}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)";
        e.currentTarget.style.transform = "translateY(-2px)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)";
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      <div style={styles.cardHeader}>
        <span style={styles.cardIcon}>{icon}</span>
        <span style={styles.label}>{label}</span>
      </div>
      <div style={{ ...styles.value, color: color || "#e8e6df" }}>
        {value}
      </div>
      {sub && <div style={styles.sub}>{sub}</div>}
    </div>
  );
}

export default function StatsPanel({ alerts }) {
  const stats = useMemo(() => {
    const bottles = alerts.filter(a => a.type !== "camera_alert");
    const avgConf = bottles.length
      ? (bottles.reduce((s, a) => s + a.confidence, 0) / bottles.length * 100).toFixed(0)
      : 0;
    const last = bottles[0]
      ? new Date(bottles[0].timestamp).toLocaleTimeString()
      : "—";
    const lastMin = bottles.filter(a => {
      return Date.now() - new Date(a.timestamp).getTime() < 60000;
    }).length;
    return { total: bottles.length, avgConf, last, lastMin };
  }, [alerts]);

  return (
    <div style={styles.grid}>
      <StatCard icon="🔍" label="TOTAL DETECTIONS" value={stats.total} color="#ef4444" />
      <StatCard icon="⏱️" label="LAST 60 SECONDS"  value={stats.lastMin} color="#f59e0b" />
      <StatCard icon="📊" label="AVG CONFIDENCE"   value={`${stats.avgConf}%`} />
      <StatCard icon="🕐" label="LAST ALERT"       value={stats.last} sub="time of last event" />
    </div>
  );
}

const styles = {
  grid: {
    display: "grid", gridTemplateColumns: "repeat(4, minmax(0,1fr))",
    gap: 14, marginBottom: 18,
  },
  card: {
    background: "rgba(18,20,31,0.8)",
    borderRadius: 14,
    padding: "16px 18px",
    border: "1px solid rgba(255,255,255,0.06)",
    backdropFilter: "blur(8px)",
    transition: "all 0.25s ease",
    cursor: "default",
  },
  cardHeader: {
    display: "flex", alignItems: "center", gap: 6, marginBottom: 10,
  },
  cardIcon: {
    fontSize: 14,
  },
  label: {
    fontSize: 10, color: "#6b7280",
    letterSpacing: 0.6, fontWeight: 600,
  },
  value: {
    fontSize: 26, fontWeight: 700, letterSpacing: -0.5,
  },
  sub: {
    fontSize: 11, color: "#6b7280", marginTop: 6,
  },
};
