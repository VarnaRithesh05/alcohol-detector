import React, { useEffect, useState } from "react";

export default function AlertToast({ alerts }) {
  const [visible, setVisible] = useState(false);
  const [current, setCurrent] = useState(null);
  const prevCount = React.useRef(0);

  useEffect(() => {
    const bottles = alerts.filter(a => a.type !== "camera_alert");
    if (bottles.length > prevCount.current && bottles.length > 0) {
      setCurrent(bottles[0]);
      setVisible(true);
      const t = setTimeout(() => setVisible(false), 5000);
      prevCount.current = bottles.length;
      return () => clearTimeout(t);
    }
    prevCount.current = bottles.length;
  }, [alerts]);

  if (!visible || !current) return null;

  return (
    <div style={{
      position: "fixed", top: 80, right: 24, zIndex: 900,
      background: "linear-gradient(135deg, rgba(26,29,39,0.95), rgba(18,16,24,0.95))",
      border: "1px solid rgba(239,68,68,0.4)",
      borderLeft: "4px solid #ef4444",
      borderRadius: 14, padding: "16px 20px", minWidth: 300, maxWidth: 360,
      boxShadow: "0 12px 48px rgba(0,0,0,0.6), 0 0 20px rgba(239,68,68,0.1)",
      animation: "toastSlide .3s cubic-bezier(0.16,1,0.3,1)",
      backdropFilter: "blur(16px)",
    }}>
      <style>{`
        @keyframes toastSlide {
          from { transform:translateX(120%) scale(0.95); opacity:0; }
          to   { transform:translateX(0) scale(1); opacity:1; }
        }
        @keyframes toastShrink { from{width:100%} to{width:0%} }
      `}</style>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
        <div>
          <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:6 }}>
            <span style={{ fontSize:14 }}>🚨</span>
            <span style={{ fontSize:12, fontWeight:800, color:"#ef4444",
                           letterSpacing:0.8 }}>ALCOHOL DETECTED</span>
          </div>
          <div style={{ fontSize:13, color:"#d1d5db" }}>
            {current.class} — {(current.confidence * 100).toFixed(0)}% confidence
          </div>
          <div style={{ fontSize:11, color:"#6b7280", marginTop:4 }}>
            {new Date(current.timestamp).toLocaleTimeString()}
          </div>
        </div>
        <button onClick={() => setVisible(false)}
          style={{ background:"none", border:"none", color:"#6b7280",
                   cursor:"pointer", fontSize:18, padding:0, lineHeight:1 }}>
          ×
        </button>
      </div>
      <div style={{ marginTop:12, height:2, background:"rgba(255,255,255,0.06)", borderRadius:1 }}>
        <div style={{
          height:"100%", background:"linear-gradient(90deg,#ef4444,#f59e0b)",
          borderRadius:1, animation:"toastShrink 5s linear forwards",
        }} />
      </div>
    </div>
  );
}
