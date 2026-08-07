import React, { useState } from "react";

const row = {
  display:"flex", alignItems:"center", justifyContent:"space-between",
  padding:"12px 0", borderBottom:"1px solid rgba(255,255,255,0.04)",
};
const labelStyle = { fontSize:13, color:"#9ca3af" };
const valStyle = { fontSize:13, fontWeight:700, color:"#e8e6df", minWidth:36, textAlign:"right" };

export default function SettingsPanel({ onSettingsChange }) {
  const [conf, setConf] = useState(75);
  const [debounce, setDebounce] = useState(3);
  const [skip, setSkip] = useState(1);
  const [open, setOpen] = useState(false);

  function apply() {
    onSettingsChange?.({ confidenceThresh: conf/100, debounceFrames: debounce, frameSkip: skip });
    setOpen(false);
  }

  return (
    <div style={{ background:"rgba(26,29,39,0.8)", borderRadius:14,
      border:"1px solid rgba(255,255,255,0.06)", marginBottom:18,
      overflow:"hidden", backdropFilter:"blur(8px)" }}>
      <div style={{ padding:"12px 18px", display:"flex", justifyContent:"space-between",
        alignItems:"center", cursor:"pointer", userSelect:"none" }}
        onClick={() => setOpen(o => !o)}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:14 }}>⚙️</span>
          <span style={{ fontSize:12, fontWeight:700, color:"#9ca3af", letterSpacing:0.8 }}>
            DETECTION SETTINGS
          </span>
        </div>
        <span style={{ color:"#6b7280", fontSize:12 }}>{open ? "▲ collapse" : "▼ expand"}</span>
      </div>

      {open && (
        <div style={{ padding:"0 18px 18px" }}>
          <div style={row}>
            <span style={labelStyle}>Confidence threshold</span>
            <div style={{ display:"flex", alignItems:"center", gap:10 }}>
              <input type="range" min={50} max={99} step={1} value={conf}
                onChange={e => setConf(Number(e.target.value))} style={{ width:120 }} />
              <span style={valStyle}>{conf}%</span>
            </div>
          </div>
          <div style={row}>
            <span style={labelStyle}>Debounce frames</span>
            <div style={{ display:"flex", alignItems:"center", gap:10 }}>
              <input type="range" min={1} max={10} step={1} value={debounce}
                onChange={e => setDebounce(Number(e.target.value))} style={{ width:120 }} />
              <span style={valStyle}>{debounce}</span>
            </div>
          </div>
          <div style={{ ...row, borderBottom:"none" }}>
            <span style={labelStyle}>Process every Nth frame</span>
            <div style={{ display:"flex", alignItems:"center", gap:10 }}>
              <input type="range" min={1} max={5} step={1} value={skip}
                onChange={e => setSkip(Number(e.target.value))} style={{ width:120 }} />
              <span style={valStyle}>{skip}</span>
            </div>
          </div>
          <div style={{ marginTop:16, display:"flex", gap:10, alignItems:"center" }}>
            <button onClick={apply} style={{
              padding:"9px 24px", background:"linear-gradient(135deg,#3b5bdb,#5b4bdb)",
              color:"#fff", border:"none", borderRadius:10, fontSize:13,
              fontWeight:600, cursor:"pointer", boxShadow:"0 2px 12px rgba(59,91,219,0.3)",
            }}>Apply</button>
            <p style={{ fontSize:11, color:"#6b7280", margin:0 }}>
              Note: changes take effect on engine restart
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
