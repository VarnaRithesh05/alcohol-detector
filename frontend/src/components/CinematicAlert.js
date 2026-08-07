import React, { useEffect, useState, useRef, useCallback } from "react";

/**
 * CinematicAlert — Movie-style bottom-of-video overlay.
 * Slides up from the bottom of the video frame when alcohol is detected,
 * similar to content advisories / lower-third banners in films.
 * Auto-dismisses after a set duration, plays an alarm tone.
 */

const DISPLAY_DURATION = 7000; // ms before auto-dismiss
const PRECAUTIONS = [
  "Alcohol consumption is injurious to health",
  "Do not drink and drive",
  "Keep alcohol away from minors",
  "Sale of alcohol to persons under 21 is prohibited",
];

function playAlarmTone() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();

    // Two-tone subtle alert beep (cinematic, not jarring)
    const tones = [
      { freq: 660, start: 0, dur: 0.12 },
      { freq: 880, start: 0.15, dur: 0.18 },
      { freq: 660, start: 0.38, dur: 0.12 },
      { freq: 880, start: 0.53, dur: 0.18 },
    ];

    tones.forEach(({ freq, start, dur }) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.1, ctx.currentTime + start);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(ctx.currentTime + start);
      osc.stop(ctx.currentTime + start + dur + 0.01);
    });

    setTimeout(() => ctx.close(), 2000);
  } catch (e) { /* Audio unavailable */ }
}

export default function CinematicAlert({ alerts }) {
  const [visible, setVisible] = useState(false);
  const [animState, setAnimState] = useState("idle"); // idle | entering | showing | exiting
  const [current, setCurrent] = useState(null);
  const [precautionIdx, setPrecautionIdx] = useState(0);
  const prevCount = useRef(0);
  const timerRef = useRef(null);
  const precautionTimer = useRef(null);

  // Cycle through precaution lines
  useEffect(() => {
    if (animState === "showing") {
      precautionTimer.current = setInterval(() => {
        setPrecautionIdx((i) => (i + 1) % PRECAUTIONS.length);
      }, 2800);
    }
    return () => clearInterval(precautionTimer.current);
  }, [animState]);

  const dismiss = useCallback(() => {
    setAnimState("exiting");
    setTimeout(() => {
      setVisible(false);
      setAnimState("idle");
      setCurrent(null);
    }, 500);
  }, []);

  // Trigger when new alert arrives
  useEffect(() => {
    const bottles = alerts.filter((a) => a.type !== "camera_alert");
    if (bottles.length > prevCount.current && bottles.length > 0) {
      const det = bottles[0];
      setCurrent(det);
      setVisible(true);
      setPrecautionIdx(0);
      setAnimState("entering");
      playAlarmTone();

      // transition to "showing" after entrance animation
      setTimeout(() => setAnimState("showing"), 600);

      // Auto-dismiss
      clearTimeout(timerRef.current);
      timerRef.current = setTimeout(dismiss, DISPLAY_DURATION);
    }
    prevCount.current = bottles.length;
  }, [alerts, dismiss]);

  if (!visible || !current) return null;

  const conf = current.confidence
    ? (current.confidence * 100).toFixed(0)
    : "—";
  const cls = current.class || "bottle";

  return (
    <div style={{
      ...s.wrapper,
      animation: animState === "entering"
        ? "cineBannerSlideUp 0.5s cubic-bezier(0.16,1,0.3,1) forwards"
        : animState === "exiting"
        ? "cineBannerSlideDown 0.45s ease-in forwards"
        : undefined,
    }} id="cinematic-alert">
      <style>{animCSS}</style>

      {/* Red accent line at top of banner */}
      <div style={s.accentLine}>
        <div style={s.accentGlow} />
      </div>

      <div style={s.content}>
        {/* Left side — icon + main info */}
        <div style={s.leftSection}>
          <div style={s.iconBadge}>
            <span style={s.iconEmoji}>⚠</span>
          </div>
          <div style={s.textBlock}>
            <div style={s.headline}>
              <span style={s.warningLabel}>ALCOHOL DETECTED</span>
              <span style={s.detailChip}>{cls} • {conf}%</span>
            </div>
            <div style={s.precautionLine} key={precautionIdx}>
              {PRECAUTIONS[precautionIdx]}
            </div>
          </div>
        </div>

        {/* Right side — timestamp + dismiss */}
        <div style={s.rightSection}>
          <span style={s.timestamp}>
            {new Date(current.timestamp).toLocaleTimeString()}
          </span>
          <button style={s.dismissBtn} onClick={dismiss} title="Dismiss">
            ✕
          </button>
        </div>
      </div>

      {/* Progress bar (auto-dismiss countdown) */}
      <div style={s.progressTrack}>
        <div style={{
          ...s.progressBar,
          animation: `cineShrink ${DISPLAY_DURATION}ms linear forwards`,
        }} />
      </div>
    </div>
  );
}

/* ── Animations ──────────────────────────────────────────────────────────── */

const animCSS = `
  @keyframes cineBannerSlideUp {
    from { transform: translateY(100%); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
  }
  @keyframes cineBannerSlideDown {
    from { transform: translateY(0);    opacity: 1; }
    to   { transform: translateY(100%); opacity: 0; }
  }
  @keyframes cineShrink {
    from { width: 100%; }
    to   { width: 0%; }
  }
  @keyframes cineGlow {
    0%, 100% { opacity: 0.6; }
    50%      { opacity: 1; }
  }
  @keyframes cinePrecautionFade {
    0%   { opacity: 0; transform: translateY(6px); }
    15%  { opacity: 1; transform: translateY(0); }
    85%  { opacity: 1; transform: translateY(0); }
    100% { opacity: 0; transform: translateY(-6px); }
  }
  @keyframes cinePulseIcon {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.1); }
  }
`;

/* ── Styles ───────────────────────────────────────────────────────────────── */

const s = {
  wrapper: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    zIndex: 100,
    pointerEvents: "auto",
    overflow: "hidden",
  },
  accentLine: {
    height: 3,
    background: "linear-gradient(90deg, transparent 0%, #ef4444 15%, #f59e0b 50%, #ef4444 85%, transparent 100%)",
    position: "relative",
  },
  accentGlow: {
    position: "absolute",
    inset: 0,
    background: "linear-gradient(90deg, transparent, rgba(239,68,68,0.6), transparent)",
    animation: "cineGlow 2s ease-in-out infinite",
    filter: "blur(4px)",
  },
  content: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 20px 10px",
    background: "linear-gradient(180deg, rgba(10,8,16,0.92) 0%, rgba(14,10,20,0.96) 100%)",
    backdropFilter: "blur(20px)",
    WebkitBackdropFilter: "blur(20px)",
    gap: 16,
  },
  leftSection: {
    display: "flex",
    alignItems: "center",
    gap: 14,
    flex: 1,
    minWidth: 0,
  },
  iconBadge: {
    width: 40,
    height: 40,
    borderRadius: 10,
    background: "rgba(239,68,68,0.15)",
    border: "1px solid rgba(239,68,68,0.3)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    animation: "cinePulseIcon 2s ease-in-out infinite",
  },
  iconEmoji: {
    fontSize: 20,
    filter: "saturate(1.5)",
  },
  textBlock: {
    flex: 1,
    minWidth: 0,
  },
  headline: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    marginBottom: 4,
    flexWrap: "wrap",
  },
  warningLabel: {
    fontSize: 13,
    fontWeight: 800,
    color: "#ef4444",
    letterSpacing: 1.5,
    fontFamily: "'Inter', -apple-system, sans-serif",
    textShadow: "0 0 20px rgba(239,68,68,0.4)",
  },
  detailChip: {
    fontSize: 11,
    fontWeight: 600,
    color: "#d1d5db",
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.1)",
    padding: "2px 10px",
    borderRadius: 20,
    whiteSpace: "nowrap",
  },
  precautionLine: {
    fontSize: 12,
    color: "#f59e0b",
    fontWeight: 500,
    letterSpacing: 0.3,
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
    animation: "cinePrecautionFade 2.8s ease-in-out",
    fontStyle: "italic",
  },
  rightSection: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    flexShrink: 0,
  },
  timestamp: {
    fontSize: 11,
    color: "#6b7280",
    fontWeight: 500,
    whiteSpace: "nowrap",
    fontVariantNumeric: "tabular-nums",
  },
  dismissBtn: {
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.1)",
    color: "#9ca3af",
    width: 28,
    height: 28,
    borderRadius: 8,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 13,
    fontWeight: 600,
    transition: "all 0.2s",
    padding: 0,
    lineHeight: 1,
  },
  progressTrack: {
    height: 2,
    background: "rgba(255,255,255,0.04)",
  },
  progressBar: {
    height: "100%",
    background: "linear-gradient(90deg, #ef4444, #f59e0b)",
    borderRadius: 1,
  },
};
