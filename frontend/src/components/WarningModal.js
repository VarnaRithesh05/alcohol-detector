import React, { useEffect, useRef, useCallback } from "react";

/**
 * WarningModal — Full-screen blocking alert that fires when alcohol is detected.
 * Shows detection details + safety precautions.
 * User MUST click "I Acknowledge" to dismiss.
 * Plays an audio alarm via Web Audio API (no external files needed).
 */

const PRECAUTIONS = [
  {
    icon: "🚗",
    title: "Do Not Drink & Drive",
    desc: "Operating a vehicle under the influence of alcohol is illegal and endangers lives. Always designate a sober driver.",
  },
  {
    icon: "🔞",
    title: "Keep Away from Minors",
    desc: "Alcohol must be stored securely and kept out of reach of children and underage individuals at all times.",
  },
  {
    icon: "⚠️",
    title: "Health Warning",
    desc: "Excessive alcohol consumption is injurious to health. It can cause liver damage, addiction, and other serious conditions.",
  },
  {
    icon: "🏢",
    title: "Workplace Policy",
    desc: "If alcohol is detected in a restricted or workplace environment, report it immediately to the designated authority.",
  },
  {
    icon: "🔒",
    title: "Secure Storage Required",
    desc: "Ensure all alcohol is stored in authorized, locked areas with proper access control to prevent unauthorized use.",
  },
];

function playAlarmSound() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();

    // Three-tone alarm pattern
    const tones = [
      { freq: 880, start: 0, dur: 0.15 },
      { freq: 1100, start: 0.2, dur: 0.15 },
      { freq: 880, start: 0.4, dur: 0.15 },
      { freq: 1100, start: 0.6, dur: 0.15 },
      { freq: 1320, start: 0.8, dur: 0.3 },
    ];

    tones.forEach(({ freq, start, dur }) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.12, ctx.currentTime + start);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(ctx.currentTime + start);
      osc.stop(ctx.currentTime + start + dur + 0.01);
    });

    // Auto-close context after the sound finishes
    setTimeout(() => ctx.close(), 2000);
  } catch (e) {
    // Audio not available — silently fail
  }
}

export default function WarningModal({ detection, onDismiss }) {
  const alarmPlayed = useRef(false);

  useEffect(() => {
    if (detection && !alarmPlayed.current) {
      alarmPlayed.current = true;
      playAlarmSound();
    }
    if (!detection) {
      alarmPlayed.current = false;
    }
  }, [detection]);

  // Prevent background scrolling while modal is open
  useEffect(() => {
    if (detection) {
      document.body.style.overflow = "hidden";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [detection]);

  const handleDismiss = useCallback(() => {
    onDismiss?.();
  }, [onDismiss]);

  if (!detection) return null;

  const conf = detection.confidence
    ? (detection.confidence * 100).toFixed(0)
    : "—";
  const time = detection.timestamp
    ? new Date(detection.timestamp).toLocaleTimeString()
    : new Date().toLocaleTimeString();
  const cls = detection.class || "bottle";

  return (
    <div style={styles.overlay} id="warning-modal-overlay">
      <style>{animationCSS}</style>

      <div style={styles.modal} id="warning-modal">
        {/* Pulsing red border ring */}
        <div style={styles.borderPulse} />

        {/* Header */}
        <div style={styles.header}>
          <div style={styles.iconContainer}>
            <span style={styles.alertIcon}>🚨</span>
          </div>
          <h1 style={styles.title}>ALCOHOL DETECTED</h1>
          <p style={styles.subtitle}>
            Immediate attention required — the detection engine has identified alcohol in the video feed
          </p>
        </div>

        {/* Detection details strip */}
        <div style={styles.detailStrip}>
          <div style={styles.detailItem}>
            <span style={styles.detailLabel}>Object</span>
            <span style={styles.detailValue}>{cls}</span>
          </div>
          <div style={styles.detailDivider} />
          <div style={styles.detailItem}>
            <span style={styles.detailLabel}>Confidence</span>
            <span style={{ ...styles.detailValue, color: "#ef4444" }}>{conf}%</span>
          </div>
          <div style={styles.detailDivider} />
          <div style={styles.detailItem}>
            <span style={styles.detailLabel}>Detected At</span>
            <span style={styles.detailValue}>{time}</span>
          </div>
          {detection.brandMatched && (
            <>
              <div style={styles.detailDivider} />
              <div style={styles.detailItem}>
                <span style={styles.detailLabel}>Brand</span>
                <span style={{ ...styles.detailValue, color: "#f59e0b" }}>Known Alcohol Brand</span>
              </div>
            </>
          )}
        </div>

        {/* Precautions */}
        <div style={styles.precautionHeader}>
          <span style={styles.shieldIcon}>🛡️</span>
          <span>SAFETY PRECAUTIONS</span>
        </div>
        <div style={styles.precautionGrid}>
          {PRECAUTIONS.map((p, i) => (
            <div key={i} style={styles.precautionCard}>
              <div style={styles.precautionIcon}>{p.icon}</div>
              <div>
                <div style={styles.precautionTitle}>{p.title}</div>
                <div style={styles.precautionDesc}>{p.desc}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Legal disclaimer */}
        <div style={styles.disclaimer}>
          This is an automated detection alert. All detections are logged for compliance
          and safety auditing purposes. Ensure all actions comply with local regulations.
        </div>

        {/* Dismiss button */}
        <button
          id="warning-dismiss-btn"
          style={styles.dismissBtn}
          onClick={handleDismiss}
          onMouseEnter={(e) => {
            e.target.style.background = "#dc2626";
            e.target.style.transform = "scale(1.02)";
            e.target.style.boxShadow = "0 8px 40px rgba(239,68,68,0.4)";
          }}
          onMouseLeave={(e) => {
            e.target.style.background = "#ef4444";
            e.target.style.transform = "scale(1)";
            e.target.style.boxShadow = "0 4px 24px rgba(239,68,68,0.3)";
          }}
        >
          ✓ I Acknowledge — Dismiss Warning
        </button>
      </div>
    </div>
  );
}

/* ── CSS Animations ────────────────────────────────────────────────────── */

const animationCSS = `
  @keyframes warningFadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes warningScaleIn {
    from { opacity: 0; transform: scale(0.92) translateY(20px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
  }
  @keyframes warningPulse {
    0%, 100% { box-shadow: inset 0 0 0 2px rgba(239,68,68,0.6), 0 0 30px rgba(239,68,68,0.1); }
    50%      { box-shadow: inset 0 0 0 3px rgba(239,68,68,1),   0 0 60px rgba(239,68,68,0.25); }
  }
  @keyframes warningIconPulse {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.15); }
  }
  @keyframes warningHeaderGlow {
    0%, 100% { text-shadow: 0 0 20px rgba(239,68,68,0.3); }
    50%      { text-shadow: 0 0 40px rgba(239,68,68,0.6); }
  }
`;

/* ── Inline Styles ─────────────────────────────────────────────────────── */

const styles = {
  overlay: {
    position: "fixed",
    inset: 0,
    zIndex: 10000,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(0,0,0,0.75)",
    backdropFilter: "blur(12px)",
    WebkitBackdropFilter: "blur(12px)",
    animation: "warningFadeIn 0.3s ease",
    padding: 20,
  },
  modal: {
    position: "relative",
    width: "100%",
    maxWidth: 680,
    maxHeight: "92vh",
    overflowY: "auto",
    background: "linear-gradient(165deg, #1a1020 0%, #0f1117 40%, #110d15 100%)",
    borderRadius: 20,
    padding: "0 0 28px",
    animation: "warningScaleIn 0.35s cubic-bezier(0.16,1,0.3,1)",
  },
  borderPulse: {
    position: "absolute",
    inset: 0,
    borderRadius: 20,
    pointerEvents: "none",
    animation: "warningPulse 2s ease-in-out infinite",
  },
  header: {
    textAlign: "center",
    padding: "32px 28px 20px",
    background: "linear-gradient(180deg, rgba(239,68,68,0.08) 0%, transparent 100%)",
    borderRadius: "20px 20px 0 0",
  },
  iconContainer: {
    width: 64,
    height: 64,
    borderRadius: "50%",
    background: "rgba(239,68,68,0.12)",
    border: "2px solid rgba(239,68,68,0.3)",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 16,
    animation: "warningIconPulse 2s ease-in-out infinite",
  },
  alertIcon: {
    fontSize: 28,
  },
  title: {
    fontSize: 26,
    fontWeight: 800,
    color: "#ef4444",
    letterSpacing: 2,
    margin: "0 0 8px",
    fontFamily: "'Inter', -apple-system, sans-serif",
    animation: "warningHeaderGlow 2s ease-in-out infinite",
  },
  subtitle: {
    fontSize: 14,
    color: "#9ca3af",
    margin: 0,
    lineHeight: 1.5,
    maxWidth: 480,
    marginLeft: "auto",
    marginRight: "auto",
  },
  detailStrip: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexWrap: "wrap",
    gap: 0,
    margin: "0 24px",
    padding: "14px 20px",
    background: "rgba(239,68,68,0.06)",
    border: "1px solid rgba(239,68,68,0.15)",
    borderRadius: 12,
  },
  detailItem: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "0 16px",
  },
  detailLabel: {
    fontSize: 10,
    fontWeight: 600,
    color: "#6b7280",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 4,
  },
  detailValue: {
    fontSize: 15,
    fontWeight: 700,
    color: "#e8e6df",
  },
  detailDivider: {
    width: 1,
    height: 28,
    background: "rgba(239,68,68,0.2)",
  },
  precautionHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "20px 28px 12px",
    fontSize: 12,
    fontWeight: 700,
    color: "#9ca3af",
    letterSpacing: 1,
  },
  shieldIcon: {
    fontSize: 16,
  },
  precautionGrid: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    padding: "0 24px",
  },
  precautionCard: {
    display: "flex",
    alignItems: "flex-start",
    gap: 14,
    padding: "14px 16px",
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 12,
    transition: "background 0.2s",
  },
  precautionIcon: {
    fontSize: 22,
    flexShrink: 0,
    width: 36,
    height: 36,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(255,255,255,0.04)",
    borderRadius: 8,
  },
  precautionTitle: {
    fontSize: 14,
    fontWeight: 700,
    color: "#e8e6df",
    marginBottom: 4,
    fontFamily: "'Inter', -apple-system, sans-serif",
  },
  precautionDesc: {
    fontSize: 12,
    color: "#9ca3af",
    lineHeight: 1.5,
  },
  disclaimer: {
    margin: "16px 28px 0",
    padding: "10px 14px",
    background: "rgba(245,158,11,0.06)",
    border: "1px solid rgba(245,158,11,0.15)",
    borderRadius: 8,
    fontSize: 11,
    color: "#9ca3af",
    lineHeight: 1.5,
    textAlign: "center",
  },
  dismissBtn: {
    display: "block",
    margin: "20px auto 0",
    padding: "14px 40px",
    fontSize: 15,
    fontWeight: 700,
    color: "#fff",
    background: "#ef4444",
    border: "none",
    borderRadius: 12,
    cursor: "pointer",
    letterSpacing: 0.3,
    boxShadow: "0 4px 24px rgba(239,68,68,0.3)",
    transition: "all 0.2s ease",
    fontFamily: "'Inter', -apple-system, sans-serif",
  },
};
