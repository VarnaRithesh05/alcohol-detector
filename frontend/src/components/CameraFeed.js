import React from "react";
import CinematicAlert from "./CinematicAlert";

export default function CameraFeed({ latestFrame, cameraAlert, engineOnline, mode, alerts }) {
  const hasDetections = latestFrame?.detections?.length > 0;

  return (
    <div style={{
      background: "rgba(26,29,39,0.8)", borderRadius: 14,
      border: `1px solid ${hasDetections ? "rgba(239,68,68,0.3)" : "rgba(255,255,255,0.06)"}`,
      overflow: "hidden", backdropFilter: "blur(8px)",
      boxShadow: hasDetections ? "0 0 30px rgba(239,68,68,0.08)" : "none",
      transition: "border-color 0.4s, box-shadow 0.4s",
    }}>
      <style>{`
        @keyframes bboxPulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
        @keyframes scanMove { 0%{top:0} 100%{top:100%} }
      `}</style>

      <div style={{ padding:"12px 18px", borderBottom:"1px solid rgba(255,255,255,0.06)",
                    display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:14 }}>{mode === "video" ? "📹" : mode === "screen" ? "🖥️" : "🎥"}</span>
          <span style={{ fontSize:12, fontWeight:700, color:"#9ca3af", letterSpacing:0.8 }}>
            {mode === "video" ? "VIDEO FEED" : mode === "screen" ? "SCREEN CAPTURE" : "LIVE FEED"}
          </span>
        </div>
        {hasDetections && (
          <span style={{ fontSize:11, fontWeight:700, color:"#ef4444",
            background:"rgba(239,68,68,0.1)", border:"1px solid rgba(239,68,68,0.3)",
            padding:"3px 10px", borderRadius:6 }}>● DETECTING</span>
        )}
      </div>

      {/* Video frame container — cinematic alert is overlaid at bottom */}
      <div style={{ position:"relative", background:"#080a10", minHeight:300 }}>
        {latestFrame?.image ? (
          <img src={`data:image/jpeg;base64,${latestFrame.image}`}
            alt="Live camera feed" style={{ width:"100%", display:"block" }} />
        ) : (
          <div style={{ display:"flex", alignItems:"center", justifyContent:"center",
                        height:300, flexDirection:"column", gap:10 }}>
            <div style={{ width:56, height:56, borderRadius:"50%",
              border:"2px solid rgba(255,255,255,0.08)", display:"flex",
              alignItems:"center", justifyContent:"center", fontSize:24, color:"#4b5563" }}>
              {mode === "video" ? "📁" : mode === "screen" ? "🖥️" : "◎"}
            </div>
            <div style={{ fontSize:14, color:"#6b7280", fontWeight:500 }}>
              {engineOnline
                ? (mode === "video" ? "Upload a video file to start detection" : "Waiting for frames...")
                : "Detection engine offline"}
            </div>
          </div>
        )}

        {latestFrame?.detections?.map((det, i) => (
          <div key={i} style={{
            position:"absolute",
            left:`${(det.bbox[0]/640)*100}%`, top:`${(det.bbox[1]/480)*100}%`,
            width:`${((det.bbox[2]-det.bbox[0])/640)*100}%`,
            height:`${((det.bbox[3]-det.bbox[1])/480)*100}%`,
            border:`2px solid ${det.brand_matched ? "#22c55e" : "#ef4444"}`,
            borderRadius:4, pointerEvents:"none",
            boxShadow:`0 0 12px ${det.brand_matched ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
            animation:"bboxPulse 1.5s ease-in-out infinite",
          }}>
            <span style={{
              position:"absolute", top:-22, left:0, fontSize:11, fontWeight:700,
              background: det.brand_matched ? "#22c55e" : "#ef4444",
              color:"#fff", padding:"2px 8px", borderRadius:4, whiteSpace:"nowrap",
            }}>{det.class} {(det.confidence*100).toFixed(0)}%</span>
          </div>
        ))}

        {hasDetections && (
          <div style={{ position:"absolute", left:0, right:0, height:2,
            background:"linear-gradient(90deg,transparent,rgba(239,68,68,0.6),transparent)",
            animation:"scanMove 2s linear infinite", pointerEvents:"none" }} />
        )}

        {cameraAlert && (
          <div style={{ position:"absolute", bottom:0, left:0, right:0,
            background:"rgba(239,68,68,0.9)", color:"#fff",
            padding:"10px 16px", fontSize:13, fontWeight:500 }}>
            ⚠️ Camera alert: {cameraAlert.reason} (lum: {cameraAlert.luminance})
          </div>
        )}

        {/* ── Cinematic movie-style alert overlay at bottom of video ── */}
        {!cameraAlert && <CinematicAlert alerts={alerts || []} />}
      </div>

      {latestFrame && (
        <div style={{ padding:"10px 18px", fontSize:11, color:"#6b7280",
                      borderTop:"1px solid rgba(255,255,255,0.06)",
                      display:"flex", justifyContent:"space-between" }}>
          <span>Last frame: {new Date(latestFrame.timestamp * 1000).toLocaleTimeString()}</span>
          <span>{latestFrame.detections?.length ?? 0} detection(s)</span>
        </div>
      )}
    </div>
  );
}
