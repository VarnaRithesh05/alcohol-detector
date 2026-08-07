/**
 * Alcohol Detection — Node.js Backend
 * Bridges the Python detection engine to the React dashboard via Socket.io
 * Handles video file uploads and relays them to the Python engine
 */

const express    = require("express");
const http       = require("http");
const { Server } = require("socket.io");
const cors       = require("cors");
const { v4: uuidv4 } = require("uuid");
const multer     = require("multer");
const path       = require("path");
const fs         = require("fs");

const app    = express();
const server = http.createServer(app);
const PORT   = 3001;

// ── Upload directory ──────────────────────────────────────────────────────
const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename:    (_, file, cb) => {
    const ext  = path.extname(file.originalname);
    cb(null, `video_${Date.now()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 2 * 1024 * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const ok = /\.(mp4|avi|mov|mkv|webm|mpeg|mpg)$/i.test(file.originalname);
    ok ? cb(null, true) : cb(new Error("Unsupported video format"));
  },
});

// ── CORS ──────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
  },
  maxHttpBufferSize: 5e6,
});

// ── State ─────────────────────────────────────────────────────────────────
const MAX_LOG = 100;
let alertLog     = [];
let latestFrame  = null;
let engineOnline = false;
let engineSocket = null;

function addAlert(event) {
  alertLog.unshift(event);
  if (alertLog.length > MAX_LOG) alertLog.pop();
}

// ── REST endpoints ────────────────────────────────────────────────────────
app.get("/health", (_, res) => {
  res.json({ status: "ok", engineOnline, alertCount: alertLog.length });
});

app.get("/alerts", (_, res) => res.json(alertLog));

app.post("/upload-video", upload.single("video"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No video file received" });

  const filePath = req.file.path;
  const filename = req.file.filename;
  console.log(`[UPLOAD] Received: ${filename} (${(req.file.size / 1024 / 1024).toFixed(1)} MB)`);

  if (engineSocket) {
    engineSocket.emit("process_video", { path: filePath, filename });
    console.log(`[UPLOAD] Sent process_video to engine`);
  } else {
    console.warn("[UPLOAD] No Python engine connected");
  }

  io.emit("video_processing_start", { filename, size: req.file.size });
  res.json({ ok: true, filename, path: filePath });
});

app.post("/use-webcam", (_, res) => {
  if (engineSocket) engineSocket.emit("use_webcam");
  io.emit("video_processing_stop");
  res.json({ ok: true });
});

app.post("/use-screen", (_, res) => {
  if (engineSocket) engineSocket.emit("use_screen");
  io.emit("video_processing_stop");
  res.json({ ok: true });
});

app.use((err, req, res, next) => {
  if (err) return res.status(400).json({ error: err.message });
  next(err);
});

// ── Socket.io ─────────────────────────────────────────────────────────────
io.on("connection", (socket) => {
  const isEngine = socket.handshake.query.client === "python";
  console.log(`[SOCKET] ${isEngine ? "Python engine" : "Dashboard"} connected — ${socket.id}`);

  if (isEngine) {
    engineSocket = socket;
    engineOnline = true;
    io.emit("engine_status", { online: true });

    socket.on("bottle_detected", (data) => {
      const event = {
        id: uuidv4(), timestamp: new Date().toISOString(),
        confidence: data.confidence, class: data.class,
        bbox: data.bbox, brandMatched: data.brand_matched,
      };
      addAlert(event);
      console.log(`[ALERT] conf=${data.confidence?.toFixed(2)}`);
      io.emit("alert", event);
    });

    socket.on("camera_alert", (data) => {
      const event = { id: uuidv4(), timestamp: new Date().toISOString(),
                      type: "camera_alert", reason: data.reason, luminance: data.luminance };
      addAlert(event);
      io.emit("camera_alert", event);
    });

    socket.on("frame_update",    (d) => { latestFrame = d; io.emit("frame_update", d); });
    socket.on("video_progress",  (d) => io.emit("video_progress", d));
    socket.on("video_done",      (d) => {
      io.emit("video_done", d);
      if (d.path && fs.existsSync(d.path)) fs.unlink(d.path, () => {});
    });

    socket.on("disconnect", () => {
      engineSocket = null;
      engineOnline = false;
      io.emit("engine_status", { online: false });
      console.log("[SOCKET] Python engine disconnected");
    });

  } else {
    socket.emit("engine_status",  { online: engineOnline });
    socket.emit("alert_history",  alertLog.slice(0, 20));
    if (latestFrame) socket.emit("frame_update", latestFrame);
    socket.on("disconnect", () => console.log(`[SOCKET] Dashboard left — ${socket.id}`));
  }
});

// ── Start ─────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n[SERVER] Running on http://localhost:${PORT}`);
  console.log(`[SERVER] Uploads: ${UPLOAD_DIR}\n`);
});
