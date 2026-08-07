import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const SERVER = "http://localhost:3001";

export function useDetectorSocket() {
  const socketRef                  = useRef(null);
  const [connected, setConnected]         = useState(false);
  const [engineOnline, setEngine]         = useState(false);
  const [alerts, setAlerts]               = useState([]);
  const [latestFrame, setFrame]           = useState(null);
  const [cameraAlert, setCameraAlert]     = useState(null);
  const [videoProgress, setVideoProgress] = useState(null);  // { percent, filename }
  const [videoActive, setVideoActive]     = useState(false);

  useEffect(() => {
    const socket = io(SERVER, { transports: ["websocket"] });
    socketRef.current = socket;

    socket.on("connect",    () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    socket.on("engine_status",  ({ online }) => setEngine(online));
    socket.on("alert_history",  (history)   => setAlerts(history));

    socket.on("alert", (event) => {
      setAlerts(prev => [event, ...prev].slice(0, 100));
      if (Notification.permission === "granted") {
        new Notification("Alcohol bottle detected!", {
          body: `Confidence: ${(event.confidence * 100).toFixed(0)}% — ${new Date(event.timestamp).toLocaleTimeString()}`,
        });
      }
    });

    socket.on("frame_update",  (data)  => setFrame(data));

    socket.on("camera_alert",  (event) => {
      setCameraAlert(event);
      setTimeout(() => setCameraAlert(null), 5000);
    });

    socket.on("video_processing_start", (data) => {
      setVideoActive(true);
      setVideoProgress({ percent: 0, filename: data.filename });
    });

    socket.on("video_progress", (data) => {
      setVideoProgress(prev => ({ ...prev, percent: data.percent }));
    });

    socket.on("video_done", () => {
      setVideoProgress(prev => prev ? { ...prev, percent: 100 } : null);
      setTimeout(() => { setVideoActive(false); setVideoProgress(null); }, 2000);
    });

    socket.on("video_processing_stop", () => {
      setVideoActive(false);
      setVideoProgress(null);
    });

    if (Notification.permission === "default") Notification.requestPermission();

    return () => socket.disconnect();
  }, []);

  return { connected, engineOnline, alerts, latestFrame, cameraAlert, videoProgress, videoActive };
}
