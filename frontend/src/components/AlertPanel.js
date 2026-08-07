import React from "react";

function ConfBar({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 90 ? "#ef4444" : pct >= 75 ? "#f59e0b" : "#22c55e";
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, marginTop:6 }}>
      <div style={{ flex:1, height:4, background:"rgba(255,255,255,0.06)", borderRadius:2 }}>
        <div style={{ width:`${pct}%`, height:"100%", background:color,
                      borderRadius:2, transition:"width .3s" }} />
      </div>
      <span style={{ fontSize:11, color, fontWeight:700, minWidth:32 }}>{pct}%</span>
    </div>
  );
}

function AlertRow({ event }) {
  const isCamera = event.type === "camera_alert";
  const time = new Date(event.timestamp).toLocaleTimeString();

  return (
    <div style={{
      padding:"14px 18px", borderBottom:"1px solid rgba(255,255,255,0.04)",
      borderLeft:`3px solid ${isCamera ? "#f59e0b" : "#ef4444"}`,
      transition:"background 0.2s",
    }}
    onMouseEnter={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.02)"}
    onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
    >
      <div style={{ display:"flex", justifyContent:"space-between",
                    alignItems:"flex-start", gap:8 }}>
        <div>
          <div style={{ display:"flex", alignItems:"center", gap:6 }}>
            <span style={{ fontSize:12 }}>{isCamera ? "📷" : "🍾"}</span>
            <span style={{
              fontSize:11, fontWeight:800, letterSpacing:0.5,
              color: isCamera ? "#f59e0b" : "#ef4444",
            }}>
              {isCamera ? "CAMERA ALERT" : "BOTTLE DETECTED"}
            </span>
          </div>
          {!isCamera && (
            <div style={{ fontSize:13, color:"#d1d5db", marginTop:4 }}>
              {event.class} · {event.brandMatched ? "known alcohol brand" : "unmatched brand"}
            </div>
          )}
          {isCamera && (
            <div style={{ fontSize:12, color:"#9ca3af", marginTop:4 }}>
              {event.reason} — luminance: {event.luminance}
            </div>
          )}
        </div>
        <span style={{ fontSize:11, color:"#6b7280", flexShrink:0 }}>{time}</span>
      </div>
      {!isCamera && <ConfBar value={event.confidence} />}
    </div>
  );
}

export default function AlertPanel({ alerts }) {
  return (
    <div style={{
      background:"rgba(26,29,39,0.8)", borderRadius:14,
      border:"1px solid rgba(255,255,255,0.06)", overflow:"hidden",
      display:"flex", flexDirection:"column", backdropFilter:"blur(8px)",
    }}>
      <div style={{ padding:"12px 18px", borderBottom:"1px solid rgba(255,255,255,0.06)",
                    display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:14 }}>📋</span>
          <span style={{ fontSize:12, fontWeight:700, color:"#9ca3af", letterSpacing:0.8 }}>
            ALERT LOG
          </span>
        </div>
        <span style={{ fontSize:11, color:"#6b7280",
          background:"rgba(255,255,255,0.04)", padding:"2px 8px", borderRadius:6 }}>
          {alerts.length} events
        </span>
      </div>

      <div style={{ overflowY:"auto", maxHeight:480 }}>
        {alerts.length === 0 ? (
          <div style={{ padding:40, textAlign:"center", color:"#4b5563", fontSize:13 }}>
            <div style={{ fontSize:28, marginBottom:8 }}>🔍</div>
            No alerts yet — monitoring active
          </div>
        ) : (
          alerts.map(a => <AlertRow key={a.id} event={a} />)
        )}
      </div>
    </div>
  );
}
