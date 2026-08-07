import React, { useState, useRef, useCallback } from "react";

const STATUS = { idle:"idle", uploading:"uploading", processing:"processing", done:"done", error:"error" };

export default function VideoUpload({ onVideoStart, onVideoStop }) {
  const [status, setStatus] = useState(STATUS.idle);
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const reset = () => {
    setStatus(STATUS.idle); setProgress(0);
    setFileName(null); setErrorMsg(null);
    onVideoStop?.();
  };

  const uploadFile = useCallback(async (file) => {
    if (!file) return;
    const allowed = ["video/mp4","video/avi","video/x-msvideo","video/quicktime",
                     "video/x-matroska","video/webm","video/mpeg"];
    if (!allowed.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|webm|mpeg|mpg)$/i)) {
      setErrorMsg("Unsupported format. Use MP4, AVI, MOV, MKV, or WebM.");
      setStatus(STATUS.error); return;
    }
    setFileName(file.name); setStatus(STATUS.uploading);
    setProgress(0); setErrorMsg(null);

    const formData = new FormData();
    formData.append("video", file);
    try {
      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status === 200) {
          const res = JSON.parse(xhr.responseText);
          setStatus(STATUS.processing); onVideoStart?.(res.filename);
        } else {
          const res = JSON.parse(xhr.responseText);
          setErrorMsg(res.error || "Upload failed"); setStatus(STATUS.error);
        }
      };
      xhr.onerror = () => {
        setErrorMsg("Network error — is the backend running on port 3001?");
        setStatus(STATUS.error);
      };
      xhr.open("POST", "http://localhost:3001/upload-video");
      xhr.send(formData);
    } catch (err) { setErrorMsg(err.message); setStatus(STATUS.error); }
  }, [onVideoStart, onVideoStop]);

  const onDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = ()  => setDragging(false);
  const onDrop = (e) => { e.preventDefault(); setDragging(false); uploadFile(e.dataTransfer.files[0]); };
  const onFileChange = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); };

  return (
    <div style={{ background:"rgba(26,29,39,0.8)", borderRadius:14,
      border:"1px solid rgba(255,255,255,0.06)", overflow:"hidden", marginBottom:18,
      backdropFilter:"blur(8px)" }}>
      <div style={{ padding:"12px 18px", borderBottom:"1px solid rgba(255,255,255,0.06)",
                    display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ fontSize:14 }}>📁</span>
          <span style={{ fontSize:12, fontWeight:700, color:"#9ca3af", letterSpacing:0.8 }}>
            VIDEO FILE INPUT
          </span>
        </div>
        {(status === STATUS.processing || status === STATUS.done) && (
          <button onClick={reset} style={{ fontSize:12, color:"#6b7280", background:"rgba(255,255,255,0.04)",
            border:"1px solid rgba(255,255,255,0.08)", borderRadius:8, padding:"4px 12px", cursor:"pointer" }}>
            ✕ Clear
          </button>
        )}
      </div>

      {(status === STATUS.idle || status === STATUS.error) && (
        <div style={{
          margin:18, border:`2px dashed ${dragging ? "#3b5bdb" : "rgba(255,255,255,0.08)"}`,
          borderRadius:14, padding:"32px 24px", textAlign:"center", cursor:"pointer",
          transition:"all .2s", background: dragging ? "rgba(59,91,219,0.06)" : "transparent",
        }}
        onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
        onClick={() => inputRef.current?.click()}>
          <input ref={inputRef} type="file"
            accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm"
            style={{ display:"none" }} onChange={onFileChange} />
          <div style={{ fontSize:36, marginBottom:12, color:"#4b5563" }}>📤</div>
          <div style={{ fontSize:15, color:"#9ca3af", fontWeight:500, marginBottom:6 }}>
            Drag & drop a video file here
          </div>
          <div style={{ fontSize:12, color:"#6b7280", marginBottom:16 }}>
            MP4, AVI, MOV, MKV, WebM supported · Max 2GB
          </div>
          <button style={{ padding:"9px 24px", background:"linear-gradient(135deg,#3b5bdb,#5b4bdb)",
            color:"#fff", border:"none", borderRadius:10, fontSize:13, fontWeight:600,
            cursor:"pointer", boxShadow:"0 2px 12px rgba(59,91,219,0.3)" }}>
            Browse file
          </button>
          {status === STATUS.error && errorMsg && (
            <div style={{ marginTop:14, fontSize:12, color:"#ef4444",
              background:"rgba(239,68,68,0.08)", padding:"8px 12px", borderRadius:8 }}>
              ⚠️ {errorMsg}
            </div>
          )}
        </div>
      )}

      {status === STATUS.uploading && (
        <div style={{ padding:22 }}>
          <div style={{ fontSize:13, color:"#9ca3af", marginBottom:12 }}>
            Uploading <strong style={{ color:"#e8e6df" }}>{fileName}</strong>...
          </div>
          <div style={{ height:6, background:"rgba(255,255,255,0.06)", borderRadius:3, overflow:"hidden" }}>
            <div style={{ width:`${progress}%`, height:"100%",
              background:"linear-gradient(90deg,#3b5bdb,#5b4bdb)", borderRadius:3,
              transition:"width .2s" }} />
          </div>
          <div style={{ fontSize:12, color:"#6b7280", marginTop:8, textAlign:"right" }}>{progress}%</div>
        </div>
      )}

      {status === STATUS.processing && (
        <div style={{ padding:22 }}>
          <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:10 }}>
            <Spinner />
            <div>
              <div style={{ fontSize:13, color:"#9ca3af" }}>
                Processing <strong style={{ color:"#e8e6df" }}>{fileName}</strong>
              </div>
              <div style={{ fontSize:11, color:"#6b7280", marginTop:2 }}>
                Detection engine is scanning — alerts appear in real time
              </div>
            </div>
          </div>
          <div style={{ height:4, background:"rgba(255,255,255,0.06)", borderRadius:2, overflow:"hidden" }}>
            <div style={{ height:"100%", background:"#22c55e", borderRadius:2,
              animation:"indeterminate 1.4s ease infinite" }} />
          </div>
          <style>{`@keyframes indeterminate {
            0%{width:0%;margin-left:0} 50%{width:60%;margin-left:20%} 100%{width:0%;margin-left:100%}
          }`}</style>
        </div>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <div style={{ width:20, height:20, borderRadius:"50%", border:"2px solid rgba(255,255,255,0.08)",
      borderTopColor:"#22c55e", flexShrink:0, animation:"spin .7s linear infinite" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
