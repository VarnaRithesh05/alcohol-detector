import React from "react";

export default function VideoProgressBar({ videoProgress, videoActive }) {
  if (!videoActive || !videoProgress) return null;

  const { percent, filename } = videoProgress;
  const done = percent >= 100;

  return (
    <div style={{
      background: "rgba(26,29,39,0.8)", borderRadius: 12,
      border: `1px solid ${done ? "rgba(34,197,94,0.3)" : "rgba(59,91,219,0.3)"}`,
      padding: "14px 18px", marginBottom: 18,
      display: "flex", alignItems: "center", gap: 14,
      backdropFilter: "blur(8px)",
      boxShadow: done ? "0 0 20px rgba(34,197,94,0.08)" : "0 0 20px rgba(59,91,219,0.08)",
      transition: "all 0.4s",
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ display:"flex", justifyContent:"space-between",
                      fontSize:12, color:"#9ca3af", marginBottom:8 }}>
          <span>
            {done ? "✅ Processing complete" : "⏳ Processing video"} —{" "}
            <strong style={{ color:"#e8e6df" }}>{filename}</strong>
          </span>
          <span style={{ fontWeight:700, color: done ? "#22c55e" : "#3b5bdb" }}>
            {percent}%
          </span>
        </div>
        <div style={{ height:5, background:"rgba(255,255,255,0.06)", borderRadius:3, overflow:"hidden" }}>
          <div style={{
            width: `${percent}%`, height:"100%", borderRadius:3,
            background: done
              ? "linear-gradient(90deg, #22c55e, #16a34a)"
              : "linear-gradient(90deg, #3b5bdb, #5b4bdb)",
            transition:"width .4s ease, background .3s",
          }} />
        </div>
      </div>
      {!done && (
        <div style={{
          width:18, height:18, borderRadius:"50%",
          border:"2px solid rgba(255,255,255,0.08)", borderTopColor:"#3b5bdb",
          flexShrink:0, animation:"spin .7s linear infinite",
        }}>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}
    </div>
  );
}
