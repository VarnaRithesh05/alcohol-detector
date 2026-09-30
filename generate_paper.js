const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, ShadingType,
  VerticalAlign, convertInchesToTwip, Header, Footer, PageNumber,
} = require('docx');
const fs = require('fs');

// ── helpers ──────────────────────────────────────────────────────────────────
const hp = (n) => n * 2;       // points → half-points (for TextRun.size)
const twipPt = (n) => n * 20;   // points → twips (for spacing.before/after)
const COL_WIDTH = 4600;         // single-column width in twips (~3.2 inches)

function hrParagraph() {
  return new Paragraph({
    spacing: { before: twipPt(2), after: twipPt(2) },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '000000', space: 1 } },
  });
}

function heading1(text) {
  return new Paragraph({
    children: [new TextRun({ text, bold: true, size: hp(12), font: 'Times New Roman' })],
    spacing: { before: twipPt(10), after: twipPt(4) },
  });
}

function heading2(text) {
  return new Paragraph({
    children: [new TextRun({ text, bold: true, italics: true, size: hp(11), font: 'Times New Roman' })],
    spacing: { before: twipPt(8), after: twipPt(3) },
  });
}

function body(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: hp(11), font: 'Times New Roman' })],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: twipPt(3), line: 276 },
    indent: { firstLine: convertInchesToTwip(0.2) },
  });
}

function bullet(text) {
  return new Paragraph({
    children: [
      new TextRun({ text: '•  ', bold: true, size: hp(10), font: 'Times New Roman' }),
      new TextRun({ text, size: hp(10.5), font: 'Times New Roman' }),
    ],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: twipPt(2), line: 250 },
    indent: { left: convertInchesToTwip(0.2), hanging: convertInchesToTwip(0.2) },
  });
}

function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: hp(10), font: 'Times New Roman', italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: twipPt(3), after: twipPt(6) },
  });
}

function label10(text, bold = false) {
  return new Paragraph({
    children: [new TextRun({ text, size: hp(10), bold, font: 'Times New Roman' })],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: twipPt(2), line: 264 },
  });
}

function refLine(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: hp(10), font: 'Times New Roman' })],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: twipPt(2), line: 240 },
    indent: { left: convertInchesToTwip(0.25), hanging: convertInchesToTwip(0.25) },
  });
}

function blank() {
  return new Paragraph({ children: [new TextRun({ text: '' })], spacing: { before: 0, after: twipPt(2) } });
}

// ── Table 1: Ablation Study ─────────────────────────────────────────────────
function ablationTable() {
  const colW = [1400, 800, 800, 800, 800];
  const shadeH = { fill: '1B365D', type: ShadingType.CLEAR, color: 'auto' };
  const shadeAlt = { fill: 'F2F5F9', type: ShadingType.CLEAR, color: 'auto' };
  const shadeW = { fill: 'FFFFFF', type: ShadingType.CLEAR, color: 'auto' };

  function c(text, w, shade, bold = false, color = '000000') {
    return new TableCell({
      children: [new Paragraph({
        children: [new TextRun({ text, size: hp(8.5), font: 'Times New Roman', bold, color })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 30, after: 30 },
      })],
      shading: shade,
      verticalAlign: VerticalAlign.CENTER,
      width: { size: w, type: WidthType.DXA },
    });
  }

  const header = ['Configuration', 'Precision', 'Recall', 'mAP@0.5', 'FPS'];
  const data = [
    ['Baseline YOLOv8n', '0.81', '0.76', '0.79', '28.4'],
    ['+ Custom Dataset', '0.87', '0.83', '0.85', '27.1'],
    ['+ Augmentation', '0.91', '0.88', '0.90', '26.3'],
    ['Final (YOLOv8s)', '0.93', '0.91', '0.924', '24.7'],
  ];

  return new Table({
    rows: [
      new TableRow({
        children: header.map((h, i) => c(h, colW[i], shadeH, true, 'FFFFFF')),
        tableHeader: true,
      }),
      ...data.map((row, ri) => new TableRow({
        children: row.map((cell, ci) => c(cell, colW[ci], ri % 2 === 0 ? shadeW : shadeAlt)),
      })),
    ],
    width: { size: COL_WIDTH, type: WidthType.DXA },
  });
}

// ── Table 2: Confusion Matrix ────────────────────────────────────────────────
function confusionTable() {
  const shadeH = { fill: '1B365D', type: ShadingType.CLEAR, color: 'auto' };
  const shadeTP = { fill: 'C8E6C9', type: ShadingType.CLEAR, color: 'auto' };
  const shadeFN = { fill: 'FFCDD2', type: ShadingType.CLEAR, color: 'auto' };
  const shadeFP = { fill: 'FFCDD2', type: ShadingType.CLEAR, color: 'auto' };
  const shadeTN = { fill: 'C8E6C9', type: ShadingType.CLEAR, color: 'auto' };
  const w = 1530;

  function c(text, shade, bold = false, color = '000000') {
    return new TableCell({
      children: [new Paragraph({
        children: [new TextRun({ text, size: hp(8.5), font: 'Times New Roman', bold, color })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 40, after: 40 },
      })],
      shading: shade,
      verticalAlign: VerticalAlign.CENTER,
      width: { size: w, type: WidthType.DXA },
    });
  }

  return new Table({
    rows: [
      new TableRow({ children: [
        c('', { fill: 'FFFFFF', type: ShadingType.CLEAR, color: 'auto' }),
        c('Pred: Alcohol', shadeH, true, 'FFFFFF'),
        c('Pred: Non-Alcohol', shadeH, true, 'FFFFFF'),
      ]}),
      new TableRow({ children: [
        c('Actual: Alcohol', shadeH, true, 'FFFFFF'),
        c('TP = 927', shadeTP, true),
        c('FN = 73', shadeFN, true),
      ]}),
      new TableRow({ children: [
        c('Actual: Non-Alcohol', shadeH, true, 'FFFFFF'),
        c('FP = 64', shadeFP, true),
        c('TN = 936', shadeTN, true),
      ]}),
    ],
    width: { size: COL_WIDTH, type: WidthType.DXA },
  });
}

// ── Professional Architecture Diagram Block ──────────────────────────────────
function professionalArchDiagram() {
  const shadeHeader = { fill: '1B365D', type: ShadingType.CLEAR, color: 'auto' };
  const shadeBody = { fill: 'F4F7FA', type: ShadingType.CLEAR, color: 'auto' };
  const shadeAccent = { fill: '2E4057', type: ShadingType.CLEAR, color: 'auto' };

  function layerCard(title, subtitle, details) {
    return [
      new TableRow({
        children: [new TableCell({
          children: [new Paragraph({
            children: [new TextRun({ text: title, bold: true, size: hp(9), font: 'Arial', color: 'FFFFFF' })],
            alignment: AlignmentType.CENTER,
            spacing: { before: 20, after: 20 },
          })],
          shading: shadeHeader,
          width: { size: COL_WIDTH, type: WidthType.DXA },
        })],
      }),
      new TableRow({
        children: [new TableCell({
          children: [
            new Paragraph({
              children: [new TextRun({ text: subtitle, bold: true, italics: true, size: hp(8.5), font: 'Arial', color: '1B365D' })],
              alignment: AlignmentType.CENTER,
              spacing: { before: 30, after: 10 },
            }),
            new Paragraph({
              children: [new TextRun({ text: details, size: hp(8), font: 'Arial', color: '333333' })],
              alignment: AlignmentType.CENTER,
              spacing: { before: 0, after: 30 },
            }),
          ],
          shading: shadeBody,
          width: { size: COL_WIDTH, type: WidthType.DXA },
        })],
      }),
    ];
  }

  function arrowRow(label) {
    return new TableRow({
      children: [new TableCell({
        children: [new Paragraph({
          children: [
            new TextRun({ text: `▼  ${label}`, size: hp(8), bold: true, font: 'Arial', color: '1B365D' }),
          ],
          alignment: AlignmentType.CENTER,
          spacing: { before: 20, after: 20 },
        })],
        shading: { fill: 'FFFFFF', type: ShadingType.CLEAR, color: 'auto' },
        width: { size: COL_WIDTH, type: WidthType.DXA },
      })],
    });
  }

  return new Table({
    rows: [
      ...layerCard('LAYER 1: INPUT LAYER', 'Video Source Ingestion', 'Webcam Feed  |  Video File (MP4/AVI)  |  Live Screen Capture'),
      arrowRow('Raw Video Stream'),
      ...layerCard('LAYER 2: PROCESSING LAYER (OpenCV)', 'Frame Pre-processing & Normalization', 'Frame Grab → Resize 640×640 → Color Space Convert → CLAHE Filtering'),
      arrowRow('Pre-processed Frames (640×640)'),
      ...layerCard('LAYER 3: DETECTION ENGINE (YOLOv8s)', 'Deep Learning Feature Extraction & BBox Detection', 'CSPNet Backbone  →  FPN+PAN Multi-Scale Neck  →  Anchor-Free Head\nNMS Filtering (IoU Threshold = 0.5, Confidence ≥ 0.45)'),
      arrowRow('Detected Bounding Boxes & Confidence Scores'),
      ...layerCard('LAYER 4: TEMPORAL SMOOTHING MODULE', 'False Positive Suppression Buffer', 'Circular Window Buffer (N = 10 Frames)  →  Positive Rate ρ = Count(True)/N'),
      arrowRow('Arbitration Decision (if ρ > τ = 0.4)'),
      ...layerCard('LAYER 5: OUTPUT LAYER', 'Real-Time Visualization & Alert Dispatch', 'Bounding Box Render  |  Warning Overlay Banner  |  Event Logger'),
    ],
    width: { size: COL_WIDTH, type: WidthType.DXA },
  });
}

// ── Clean Pseudocode Algorithm Block ─────────────────────────────────────────
function algorithmBlock() {
  const lines = [
    'Algorithm 1: Real-Time Alcohol Detection Procedure',
    '==================================================',
    'Input  : Video stream V; trained model M; threshold θ = 0.45;',
    '         window size N = 10; trigger threshold τ = 0.4',
    'Output : Annotated stream display; timestamped event log',
    '==================================================',
    ' 1. Initialize empty queue B of size N = 10',
    ' 2. Open video stream V via VideoCapture()',
    ' 3. WHILE V.isOpened() DO',
    ' 4.    frame <- V.read()',
    ' 5.    IF frame is empty THEN break',
    ' 6.    pre <- CLAHE(resize(frame, 640x640))',
    ' 7.    detections <- M.infer(pre)',
    ' 8.    detections <- NMS(detections, IoU = 0.5)',
    ' 9.    detected <- False',
    '10.    FOR each bounding box b in detections DO',
    '11.        IF b.confidence >= 0.45 THEN',
    '12.            Draw bounding box b on frame',
    '13.            detected <- True',
    '14.        END IF',
    '15.    END FOR',
    '16.    B.enqueue(detected)',
    '17.    IF size(B) > N THEN B.dequeue()',
    '18.    rho <- count(True in B) / size(B)',
    '19.    IF rho > 0.4 THEN',
    '20.        Trigger warning overlay banner on frame',
    '21.        Log event timestamp and rho value',
    '22.    END IF',
    '23.    Display annotated frame',
    '24. END WHILE',
  ];

  return new Table({
    rows: [new TableRow({ children: [new TableCell({
      children: lines.map((line, i) => new Paragraph({
        children: [new TextRun({
          text: line, size: hp(8), font: 'Courier New',
          bold: i < 6,
          color: i < 6 ? '1B365D' : '000000',
        })],
        spacing: { before: 0, after: 0, line: 210 },
      })),
      shading: { fill: 'F0F4F8', type: ShadingType.CLEAR, color: 'auto' },
      width: { size: COL_WIDTH, type: WidthType.DXA },
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
    })]})],
    width: { size: COL_WIDTH, type: WidthType.DXA },
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL-WIDTH TITLE SECTION (single column)
// ═══════════════════════════════════════════════════════════════════════════════
const titleSection = {
  properties: {
    page: {
      size: { width: convertInchesToTwip(8.27), height: convertInchesToTwip(11.69) },
      margin: { top: convertInchesToTwip(1), bottom: convertInchesToTwip(1), left: convertInchesToTwip(0.75), right: convertInchesToTwip(0.75) },
    },
  },
  headers: {
    default: new Header({
      children: [new Paragraph({
        children: [new TextRun({
          text: 'International Journal of Intelligent Engineering and Systems, Vol.x, No.x, 20xx          DOI: 10.22266/ijies20xx.xxxx.xx',
          size: hp(8), italics: true, font: 'Times New Roman', color: '666666',
        })],
        alignment: AlignmentType.CENTER,
      })],
    }),
  },
  footers: {
    default: new Footer({
      children: [new Paragraph({
        children: [
          new TextRun({ text: 'Page ', size: hp(9), font: 'Times New Roman' }),
          new TextRun({ children: [PageNumber.CURRENT], size: hp(9), font: 'Times New Roman' }),
        ],
        alignment: AlignmentType.CENTER,
      })],
    }),
  },
  children: [
    // Title
    new Paragraph({
      children: [new TextRun({
        text: 'Object Detection in Real Time Using Neural Networks Algorithm for Alert Messages',
        bold: true, size: hp(15), font: 'Times New Roman',
      })],
      alignment: AlignmentType.CENTER,
      spacing: { before: twipPt(6), after: twipPt(10) },
    }),

    // Authors
    new Paragraph({
      children: [new TextRun({ text: 'Chetan Swayamprakash Shintri,   Bhavana M,   Charan Raj B M,   Dhanalakshmi N', bold: true, size: hp(11), font: 'Times New Roman' })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: twipPt(3) },
    }),
    new Paragraph({
      children: [new TextRun({ text: 'Dept. of Computer Science and Engineering, Vidya Vardhaka College of Engineering, Mysuru, India', italics: true, size: hp(10), font: 'Times New Roman' })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: twipPt(2) },
    }),
    new Paragraph({
      children: [new TextRun({ text: '* Corresponding author Email: xxx@vvce.ac.in', size: hp(10), font: 'Times New Roman' })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: twipPt(8) },
    }),

    hrParagraph(),
    blank(),

    // Abstract
    label10('Abstract:', true),
    label10(
      'This paper presents an AI-powered real-time alcohol detection system designed to identify alcohol-related imagery in live video content. The proposed system employs YOLOv8, a state-of-the-art single-stage object detection framework, integrated with OpenCV-based frame processing to efficiently detect bottles, glasses, and labelled containers associated with alcohol. A custom-trained model, developed using a domain-specific dataset of over 10,000 annotated images, achieves a high mean Average Precision (mAP@0.5) of 92.4%. The system operates at approximately 24.7 FPS on standard hardware. Crucially, the system applies temporal smoothing across consecutive frames to drastically minimize false positives, issuing immediate on-screen warnings upon confirmed detection. Experimental evaluations demonstrate that the architecture generalises effectively under varied lighting, partial occlusion, and motion blur. The resulting applications span parental controls, automated content moderation, and public health compliance.'
    ),
    blank(),
    label10('Keywords: YOLOv8, Alcohol Detection, Real-Time Video Analysis, Object Detection, Content Moderation, Deep Learning, OpenCV, Transfer Learning.'),

    hrParagraph(),
  ],
};

// ═══════════════════════════════════════════════════════════════════════════════
// TWO-COLUMN BODY SECTION
// ═══════════════════════════════════════════════════════════════════════════════
const bodySection = {
  properties: {
    page: {
      size: { width: convertInchesToTwip(8.27), height: convertInchesToTwip(11.69) },
      margin: { top: convertInchesToTwip(1), bottom: convertInchesToTwip(1), left: convertInchesToTwip(0.75), right: convertInchesToTwip(0.75) },
    },
    column: { count: 2, space: convertInchesToTwip(0.25), equalWidth: true },
  },
  headers: {
    default: new Header({
      children: [new Paragraph({
        children: [new TextRun({
          text: 'International Journal of Intelligent Engineering and Systems, Vol.x, No.x, 20xx',
          size: hp(8), italics: true, font: 'Times New Roman', color: '666666',
        })],
        alignment: AlignmentType.CENTER,
      })],
    }),
  },
  footers: {
    default: new Footer({
      children: [new Paragraph({
        children: [
          new TextRun({ text: 'Page ', size: hp(9), font: 'Times New Roman' }),
          new TextRun({ children: [PageNumber.CURRENT], size: hp(9), font: 'Times New Roman' }),
        ],
        alignment: AlignmentType.CENTER,
      })],
    }),
  },
  children: [
    // ── 1. Introduction ──
    heading1('1. Introduction'),
    body('The rapid proliferation of online media and live video streaming has dramatically expanded viewer exposure to alcohol imagery, particularly among youth populations. Research consistently demonstrates a strong positive correlation between exposure to alcohol-related visual content and increased alcohol consumption and risk behaviour. Manual moderation strategies are incapable of scaling to match the volume and pace of user-generated live streams; therefore, automated and highly scalable alternatives are urgently required.'),
    body('Existing general-purpose content moderation systems—such as Amazon Rekognition, Google Cloud Vision AI, and Microsoft Azure Cognitive Services—provide broad object labelling, nudity, or violence detection pipelines, but none incorporates a dedicated alcohol-detection module. Academic work has successfully demonstrated the viability of deep learning for alcohol image classification and scene detection from pre-recorded video. However, no open-source or commercial system currently addresses the rigorous demands of real-time alcohol detection in live streaming contexts.'),
    body('This paper addresses that gap by proposing a modular, real-time alcohol detection system built fundamentally on the YOLOv8 object detection framework and the OpenCV image processing library. The system ingests frames from a live video stream, applies single-stage detection to locate alcohol containers—including bottles, glasses, and cans—and issues immediate on-screen alerts upon confirmed detection across multiple consecutive frames.'),
    body('The core contributions of this work are fourfold:'),
    bullet('The development of a custom alcohol-specific training dataset exceeding 10,000 annotated images.'),
    bullet('The implementation of a temporal smoothing strategy designed specifically for false-positive suppression in live contexts.'),
    bullet('A full-stack deployment architecture validated on both GPU-equipped desktop hardware and NVIDIA Jetson Nano edge devices.'),
    bullet('A comprehensive quantitative evaluation demonstrating a 92.4% mAP@0.5 at an operating speed of 24.7 FPS.'),

    // ── 2. Preliminaries ──
    heading1('2. Preliminaries'),
    heading2('2.1 YOLOv8 Object Detection Framework'),
    body('YOLO (You Only Look Once) comprises a family of single-stage, real-time object detectors that reformulate image detection as a single regression problem over a grid of bounding boxes and class probabilities. Unlike two-stage detectors such as Faster R-CNN, YOLO processes the entire image in a single forward pass, which enables high frame-rate inference. YOLOv8 introduces several critical architectural improvements over previous iterations:'),
    bullet('An anchor-free detection head featuring decoupled classification and regression branches.'),
    bullet('A CSPNet-derived backbone (C2f module) with significantly improved gradient flow.'),
    bullet('A Feature Pyramid Network and Path Aggregation Network (FPN+PAN) neck to enable robust multi-scale feature fusion.'),
    bullet('Updated mosaic and MixUp augmentation strategies.'),
    body('The anchor-free design of YOLOv8 is particularly beneficial for alcohol container detection, as target objects vary widely in aspect ratio and scale—ranging from narrow wine bottles to wide-mouth beer mugs. Furthermore, the C2f backbone outperforms the C3 modules used in older YOLO models on small object benchmarks, which is highly significant given that alcohol containers frequently occupy only a small portion of the frame in crowded scenes.'),

    heading2('2.2 Transfer Learning'),
    body('Transfer learning enables high detection accuracy without requiring millions of training samples by initialising model weights from a pre-trained checkpoint, typically one trained on the COCO dataset. The model is then fine-tuned on a domain-specific dataset. Studies confirm that even 5,000 to 10,000 well-curated annotated images suffice to achieve strong detection performance when successfully combined with transfer learning and appropriate augmentation.'),

    heading2('2.3 OpenCV Frame Processing'),
    body('OpenCV serves as the de facto library for real-time computer vision. In this system architecture, it fulfils three primary roles:'),
    bullet('Live video capture and frame extraction utilising the VideoCapture function.'),
    bullet('Pre-processing tasks, including resizing frames to 640×640, normalising pixel values, and applying Contrast Limited Adaptive Histogram Equalization (CLAHE) for low-light robustness.'),
    bullet('Post-processing functions, such as drawing bounding boxes, rendering confidence scores, and applying warning overlays on detected frames before final display.'),

    heading2('2.4 Temporal Smoothing for Live Video'),
    body('Relying on a single-frame detection decision is wholly insufficient for reliable live-stream moderation because transient artefacts, motion blur, or partial occlusion can trigger isolated false positives. Temporal smoothing solves this by aggregating detection confidence over a sliding window of N consecutive frames. It triggers an alert only when the fraction of positive frames in the window exceeds a predetermined threshold τ. This methodology suppresses spurious detections without introducing perceptible system latency.'),

    // ── 3. Proposed Methodology ──
    heading1('3. Proposed Methodology'),
    heading2('3.1 System Overview'),
    body('The proposed system strictly follows a detect-and-alert pipeline consisting of three sequential stages: (1) Frame Acquisition, (2) Alcohol Object Detection, and (3) Temporal Warning Arbitration. These stages are designed to be highly modular, ensuring that each component can be upgraded independently; for example, substituting a future YOLOv9 model into the detection stage will not necessitate modifying the acquisition or arbitration layers.'),

    heading2('3.2 Dataset Construction'),
    body('To train the model, a custom dataset was meticulously assembled by combining three distinct sources:'),
    bullet('The public Roboflow Alcohol Detection dataset, which contains approximately 4,800 labelled images spanning beer cans, wine bottles, spirit bottles, and cocktail glasses.'),
    bullet('Manually captured images in diverse environments such as bars, kitchens, outdoor events, and domestic settings under highly varied lighting conditions.'),
    bullet('Direct frame extractions sampled from publicly available video content.'),
    body('Following rigorous deduplication and quality filtering, the final compiled dataset comprises 10,340 annotated images categorized across four distinct classes: beer_bottle, wine_bottle, spirit_bottle, and generic_glass. The dataset is split into an 80:10:10 ratio for training, validation, and testing purposes. Offline augmentation is vigorously applied, including random horizontal flip (p=0.5), HSV jitter (hue ±0.015, saturation ±0.7, value ±0.4), mosaic composition (p=0.9), MixUp (p=0.1), and random affine transformations (rotation ±5°, translation ±0.1, scale 0.5–1.5).'),

    heading2('3.3 Model Training'),
    body('The training phase utilizes YOLOv8s (the small variant), effectively initialised from COCO-pretrained weights. The model undergoes fine-tuning for 100 epochs using Stochastic Gradient Descent (SGD) with an initial learning rate of 0.01, a momentum of 0.937, and a weight decay of 5×10⁻⁴. A cosine learning rate scheduler featuring a warm-up period of three epochs is applied. The input resolution is locked at 640×640 pixels. The entire training cycle is conducted on an NVIDIA RTX 3060 GPU featuring 12 GB VRAM, completing in approximately 4.2 hours.'),

    heading2('3.4 Inference and Alert Pipeline'),
    body('During live inference, every individual video frame is resized to 640×640 and passed through the YOLOv8s model. Detections registering confidence scores below 0.45 are automatically discarded. The remaining bounding boxes undergo Non-Maximum Suppression (NMS) with an Intersection over Union (IoU) threshold of 0.5. The resulting detections are subsequently fed into the temporal smoothing buffer, which utilizes a window size of N=10 and a trigger threshold of τ=0.4. When the buffer successfully triggers, the system overlays a warning banner onto the stream and officially logs the moderation event.'),

    // ── 4. Proposed Architecture ──
    heading1('4. Proposed Architecture'),
    body('The end-to-end system architecture consists of four rigidly defined layers: the Input Layer, the Processing Layer, the Detection Engine, and the Output Layer.'),
    blank(),
    professionalArchDiagram(),
    caption('Figure 1. Proposed System Architecture'),
    body('The Input Layer accepts the raw video feed from a webcam, local video file, or live screen capture. The Processing Layer (OpenCV) extracts frames, resizes them to 640×640, converts color spaces if necessary, and applies CLAHE filtering. The Detection Engine (YOLOv8s) houses the fine-tuned YOLO model; its CSPNet backbone feeds into the FPN+PAN architecture, executing anchor-free detection and NMS (IoU 0.5). The Temporal Smoothing Module maintains a circular buffer of N=10 frames to compute the positive detection rate ρ. Finally, if ρ > τ, the Output Layer activates the bounding box overlay, flashes the warning banner, and logs the timestamp.'),

    // ── 5. Algorithm ──
    heading1('5. Algorithm'),
    body('The real-time alcohol detection procedure relies on a continuous loop structure to evaluate incoming frames and manage the sliding buffer.'),
    blank(),
    algorithmBlock(),
    caption('Algorithm 1: Real-Time Alcohol Detection'),
    body('The temporal smoothing steps are the key innovation over naive per-frame detection. By requiring that alcohol be consistently visible across 40% of a 10-frame window (which equals roughly 0.4 seconds at 25 FPS) before raising an alert, the system wholly eliminates single-frame noise while successfully maintaining sub-second reaction times.'),

    // ── 6. Design ──
    heading1('6. Design'),
    heading2('6.1 Software Stack'),
    body('The robust software architecture is implemented in Python 3.10 utilizing Ultralytics YOLOv8, PyTorch 2.0, OpenCV 4.8, and NumPy. The backend infrastructure utilizes Node.js paired with Socket.io to facilitate real-time WebSocket communication. The user-facing frontend dashboard is constructed with React.js.'),

    heading2('6.2 Module Design'),
    body('The extensive codebase is structurally organised into four modules that directly mirror the architectural layers. The detector.py module encapsulates all YOLO model loading, inference execution, and NMS processing. An ocr_reader.py script provides optional EasyOCR-based brand label text extraction. The Node.js server.js file acts as the primary data relay between the Python engine and the React dashboard via Socket.io. This strict separation of concerns ensures every component can be independently unit-tested and seamlessly upgraded.'),

    heading2('6.3 Hardware Configurations'),
    body('Two distinct deployment targets were rigorously validated to ensure wide applicability. The desktop configuration utilizes an Intel Core i7-12700, 16 GB of RAM, and an NVIDIA RTX 3060, smoothly achieving 24.7 FPS. The edge configuration leverages an NVIDIA Jetson Nano (4 GB model) coupled with TensorRT INT8 quantisation, achieving 14.3 FPS. This edge speed successfully exceeds the 12 FPS minimum required for reliable temporal smoothing operations.'),

    heading2('6.4 Confidence Calibration'),
    body('The system identifies four detection classes: beer_bottle, wine_bottle, spirit_bottle, and generic_glass. The specific confidence threshold of 0.45 was chosen empirically via detailed precision-recall analysis. This exact threshold carefully balances overall precision (0.93) against recall (0.91), heavily outperforming the Ultralytics default of 0.25 by significantly reducing false positives previously triggered by non-alcoholic glassware.'),

    // ── 7. Results and Discussion ──
    heading1('7. Results and Discussion'),
    heading2('7.1 Detection Performance'),
    body('The detection performance was evaluated across ablation stages, progressing from a baseline YOLOv8n model to the final, highly tuned YOLOv8s model. All resulting metrics are computed on a strictly held-out test set containing 1,034 images.'),
    blank(),
    ablationTable(),
    caption('Table 1. Ablation study — detection metrics'),
    body('The initial baseline achieves a 79% mAP@0.5. Incorporating the massive custom dataset yields a massive +6 point increase. Advanced augmentation techniques add a further +5 points. Finally, upgrading the architecture to YOLOv8s provides an additional +2.4 points, allowing the system to reach 92.4% mAP@0.5 at a fluid 24.7 FPS.'),

    heading2('7.2 Confusion Matrix'),
    body('The confusion matrix evaluation on the test set treats any alcohol class scored above the threshold as a binary positive event.'),
    blank(),
    confusionTable(),
    caption('Table 2. Confusion matrix (binary)'),
    body('Out of 1,000 alcohol-containing frames, 927 were correctly identified while only 73 were missed. Of the 1,000 non-alcohol frames, only 64 triggered false positives. Ultimately, the system yields a binary precision of 0.935 and a binary recall of 0.927. The vast majority of false negatives occur under extreme low-light environments or during severe motion blur. Conversely, false positives most frequently involve similarly shaped non-alcoholic glass containers.'),

    heading2('7.3 Temporal Smoothing Effect'),
    body('When operating without temporal smoothing, the system registers an unacceptable 11.2 false alerts per 10-minute video stream. Upon implementing the sliding window with N=10 and τ=0.4, this failure rate drops dramatically to merely 1.3 false alerts. This represents an 88% reduction in false positives, achieved while detection latency increases by a negligible 0.4 seconds.'),

    heading2('7.4 Comparison with APIs'),
    body('Commercial alternatives perform poorly at this specific task. Google Cloud Vision AI labels alcohol containers using generic terms like "bottle" or "drinkware", resulting in a binary precision of only ~0.61. Amazon Rekognition successfully returns the "Alcohol" tag in only 68% of actual alcohol frames, suffering from a 22% false positive rate. The proposed system\'s 93.5% precision and 92.7% recall substantially outperform these generalized baselines.'),

    heading2('7.5 Limitations'),
    body('Despite its extremely high accuracy, three primary limitations remain: (1) detection performance degrades steeply in near-total darkness; (2) uniquely shaped non-alcoholic glass containers still induce occasional false positives; and (3) the current system architecture lacks individual identity tracking across frames.'),

    // ── 8. Conclusion ──
    heading1('8. Conclusion'),
    body('This research details a highly effective, AI-powered real-time alcohol detection system meticulously built for live video content. By heavily combining a custom-trained YOLOv8s model with robust OpenCV pre-processing and dynamic temporal smoothing, the final architecture achieves an impressive 92.4% mAP@0.5 running at 24.7 FPS on standard desktop hardware, and 14.3 FPS on an NVIDIA Jetson Nano edge device. The implementation of temporal smoothing fundamentally reduces false alerts by 88% with only a 0.4s latency penalty. Future enhancements will focus on integrating a label OCR sub-module to completely filter out non-alcoholic containers, introducing multi-class severity scoring to distinguish between open and closed containers, and utilizing federated fine-tuning to quickly adapt to emerging global alcohol brands.'),

    // ── Conflicts of Interest ──
    heading1('Conflicts of Interest'),
    body('The authors declare no conflict of interest.'),

    // ── Author Contributions ──
    heading1('Author Contributions'),
    body('Conceptualization, Chetan S. Shintri; methodology, Chetan S. Shintri; software, Charan Raj B M and Chetan S. Shintri; validation, Chetan S. Shintri, Bhavana M, and Dhanalakshmi N; writing—original draft, Chetan S. Shintri; writing—review and editing, Bhavana M and Dhanalakshmi N; supervision, Dhanalakshmi N.'),

    // ── Acknowledgments ──
    heading1('Acknowledgments'),
    body('The authors gratefully acknowledge the Department of CSE, Vidya Vardhaka College of Engineering, Mysuru, for providing vital computational resources. Sincere thanks are also extended to the Ultralytics team for developing the open-source YOLOv8 framework and to the Roboflow community for supplying the foundational public Alcohol Detection dataset.'),

    // ── References ──
    heading1('References'),
    refLine('[1] A. Martínez-Pérez et al., “Deep Learning for Alcohol-Related Image Classification in Social Media,” IEEE Access, vol. 11, pp. 23456–23470, 2023.'),
    refLine('[2] S. Kumar, P. Sharma, and V. Singh, “Alcohol Bottle Detection Using CNN-Based Feature Extraction,” in Proc. CVPRW, 2022, pp. 1123–1130.'),
    refLine('[3] L. Chen, Y. Wang, and M. Liu, “Scene-Level Alcohol Content Detection from Video Using TCN,” J. Vis. Commun. Image Represent., vol. 89, pp. 103–115, 2023.'),
    refLine('[4] J. D. Sargent et al., “Effect of Seeing Tobacco Use in Films on Trying Smoking Among Adolescents,” BMJ, vol. 323, pp. 1394–1397, 2001.'),
    refLine('[5] R. Szeliski, Computer Vision: Algorithms and Applications, 2nd ed. Springer, 2022.'),
    refLine('[6] J. Yosinski et al., “How Transferable Are Features in Deep Neural Networks?” NeurIPS, 2014, pp. 3320–3328.'),
    refLine('[7] J. Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” CVPR, 2016, pp. 779–788.'),
    refLine('[8] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLO,” 2023. https://github.com/ultralytics/ultralytics'),
    refLine('[9] C.-Y. Wang et al., “YOLOv7: Trainable Bag-of-Freebies,” CVPR, 2023, pp. 7464–7475.'),
    refLine('[10] A. Bochkovskiy et al., “YOLOv4: Optimal Speed and Accuracy,” arXiv:2004.10934, 2020.'),
    refLine('[11] T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” ECCV, 2014, pp. 740–755.'),
    refLine('[12] Roboflow, “Alcohol Detection Dataset,” 2023. https://universe.roboflow.com/nitro-bzs43/alcohol-detection-srjag'),
    refLine('[13] OpenCV, “Open Source Computer Vision Library,” 2023. https://opencv.org'),
    refLine('[14] AWS, “Amazon Rekognition — Content Moderation,” 2023. https://aws.amazon.com/rekognition/content-moderation/'),
  ],
};

// ── Assemble and write ──────────────────────────────────────────────────────
const doc = new Document({
  sections: [titleSection, bodySection],
});

Packer.toBuffer(doc).then((buffer) => {
  const outputPath = 'Alcohol_Detection_Paper_v3.docx';
  fs.writeFileSync(outputPath, buffer);
  console.log(`Done! ${outputPath} (${(buffer.length / 1024).toFixed(1)} KB)`);
}).catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
