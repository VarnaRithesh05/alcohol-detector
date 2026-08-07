import React, { useEffect, useRef } from "react";

const W = 600, H = 130, PAD = { top:14, right:16, bottom:30, left:38 };
const WINDOW = 60;

function buildPoints(alerts) {
  return alerts.filter(a => a.type !== "camera_alert").slice(0, WINDOW).reverse();
}

export default function ConfidenceChart({ alerts }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    const pts = buildPoints(alerts);
    const inner = { w: W - PAD.left - PAD.right, h: H - PAD.top - PAD.bottom };

    ctx.clearRect(0, 0, W, H);

    // Background with subtle gradient
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, "#14161f");
    bg.addColorStop(1, "#10121a");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    [0.25, 0.5, 0.75, 1].forEach(v => {
      const y = PAD.top + inner.h * (1 - v);
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + inner.w, y); ctx.stroke();
      ctx.fillStyle = "#4b5563";
      ctx.font = "10px Inter, -apple-system, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(`${Math.round(v * 100)}%`, PAD.left - 4, y + 3);
    });

    if (pts.length === 0) {
      ctx.fillStyle = "#4b5563";
      ctx.font = "12px Inter, -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No detections yet", W / 2, H / 2);
      return;
    }

    const xStep = inner.w / Math.max(pts.length - 1, 1);

    // Fill under line (gradient)
    ctx.beginPath();
    pts.forEach((pt, i) => {
      const x = PAD.left + i * xStep;
      const y = PAD.top + inner.h * (1 - pt.confidence);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo(PAD.left + (pts.length - 1) * xStep, PAD.top + inner.h);
    ctx.lineTo(PAD.left, PAD.top + inner.h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + inner.h);
    grad.addColorStop(0, "rgba(239,68,68,0.15)");
    grad.addColorStop(1, "rgba(239,68,68,0.01)");
    ctx.fillStyle = grad;
    ctx.fill();

    // Plot line
    ctx.beginPath();
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    pts.forEach((pt, i) => {
      const x = PAD.left + i * xStep;
      const y = PAD.top + inner.h * (1 - pt.confidence);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Threshold line at 75%
    const ty = PAD.top + inner.h * 0.25;
    ctx.beginPath();
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.moveTo(PAD.left, ty);
    ctx.lineTo(PAD.left + inner.w, ty);
    ctx.stroke();
    ctx.setLineDash([]);

    // Dots with glow
    pts.forEach((pt, i) => {
      const x = PAD.left + i * xStep;
      const y = PAD.top + inner.h * (1 - pt.confidence);
      // Glow
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(239,68,68,0.2)";
      ctx.fill();
      // Dot
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#ef4444";
      ctx.fill();
    });

    // X-axis time labels
    ctx.fillStyle = "#4b5563";
    ctx.font = "10px Inter, -apple-system, sans-serif";
    ctx.textAlign = "center";
    const step = Math.max(1, Math.floor(pts.length / 5));
    pts.forEach((pt, i) => {
      if (i % step !== 0) return;
      const x = PAD.left + i * xStep;
      const t = new Date(pt.timestamp).toLocaleTimeString([], { hour:"2-digit", minute:"2-digit", second:"2-digit" });
      ctx.fillText(t, x, H - 6);
    });
  }, [alerts]);

  return (
    <div style={{ background:"rgba(26,29,39,0.8)", borderRadius:14,
      border:"1px solid rgba(255,255,255,0.06)", overflow:"hidden", marginBottom:18,
      backdropFilter:"blur(8px)" }}>
      <div style={{ padding:"12px 18px", borderBottom:"1px solid rgba(255,255,255,0.06)",
                    display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:14 }}>📈</span>
          <span style={{ fontSize:12, fontWeight:700, color:"#9ca3af", letterSpacing:0.8 }}>
            CONFIDENCE OVER TIME
          </span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:14, fontSize:11, color:"#6b7280" }}>
          <span style={{ display:"flex", alignItems:"center", gap:4 }}>
            <span style={{ width:12, height:2, background:"#ef4444", display:"inline-block" }} />
            Detection
          </span>
          <span style={{ display:"flex", alignItems:"center", gap:4 }}>
            <span style={{ width:12, height:0, borderTop:"2px dashed #f59e0b", display:"inline-block" }} />
            75% threshold
          </span>
        </div>
      </div>
      <canvas ref={canvasRef}
        style={{ width:"100%", height:H, display:"block" }}
        role="img" aria-label="Confidence score over time for bottle detections" />
    </div>
  );
}
