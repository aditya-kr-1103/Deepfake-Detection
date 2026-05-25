import { Activity, AlertCircle, CheckCircle2, Image, Loader2, Upload, Video } from "lucide-react";
import { ChangeEvent, FormEvent, useMemo, useState } from "react";

type MediaKind = "image" | "video";

type DetectionResponse = {
  prediction: string;
  score: number;
  confidence: number;
  media_type: MediaKind;
  frames_used?: number | null;
  message?: string | null;
};

type UploadState = {
  file: File | null;
  previewUrl: string | null;
  loading: boolean;
  result: DetectionResponse | null;
  error: string | null;
};

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);

const initialState: UploadState = {
  file: null,
  previewUrl: null,
  loading: false,
  result: null,
  error: null,
};

const MODES = {
  image: {
    label: "Photo",
    accept: "image/*",
    endpoint: "/detect-image",
    icon: Image,
  },
  video: {
    label: "Video",
    accept: "video/*",
    endpoint: "/detect-video",
    icon: Video,
  },
} satisfies Record<
  MediaKind,
  {
    label: string;
    accept: string;
    endpoint: string;
    icon: typeof Image;
  }
>;

export default function App() {
  const [activeKind, setActiveKind] = useState<MediaKind>("image");
  const [uploadState, setUploadState] = useState<UploadState>(initialState);
  const activeMode = MODES[activeKind];
  const ActiveIcon = activeMode.icon;

  const fileSize = useMemo(() => {
    if (!uploadState.file) return "";
    return `${(uploadState.file.size / (1024 * 1024)).toFixed(2)} MB`;
  }, [uploadState.file]);

  function changeMode(kind: MediaKind) {
    setUploadState((previous) => {
      if (previous.previewUrl) URL.revokeObjectURL(previous.previewUrl);
      return initialState;
    });
    setActiveKind(kind);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setUploadState((previous) => {
      if (previous.previewUrl) URL.revokeObjectURL(previous.previewUrl);
      return {
        ...initialState,
        file,
        previewUrl: file ? URL.createObjectURL(file) : null,
      };
    });
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!uploadState.file) return;

    setUploadState((previous) => ({ ...previous, loading: true, error: null, result: null }));

    const body = new FormData();
    body.append("file", uploadState.file);

    try {
      const response = await fetch(`${API_BASE_URL}${activeMode.endpoint}`, {
        method: "POST",
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Detection failed.");
      }
      setUploadState((previous) => ({
        ...previous,
        loading: false,
        result: payload as DetectionResponse,
      }));
    } catch (error) {
      setUploadState((previous) => ({
        ...previous,
        loading: false,
        error: error instanceof Error ? error.message : "Detection failed.",
      }));
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Beyond the Pixel</p>
          <h1>Deepfake Detection Console</h1>
        </div>
        <div className="status-pill">
          <Activity size={16} aria-hidden="true" />
          <span>FastAPI: {API_BASE_URL.replace(/^https?:\/\//, "")}</span>
        </div>
      </header>

      <section className="checker-layout" aria-label="Deepfake checker">
        <form className="panel upload-panel" onSubmit={onSubmit}>
          <div className="panel-header">
            <div className="panel-title">
              <ActiveIcon size={20} aria-hidden="true" />
              <h2>{activeMode.label} Upload</h2>
            </div>
            <div className="mode-toggle" aria-label="Media type">
              {(Object.keys(MODES) as MediaKind[]).map((kind) => {
                const ModeIcon = MODES[kind].icon;
                return (
                  <button
                    className={kind === activeKind ? "active" : ""}
                    key={kind}
                    type="button"
                    onClick={() => changeMode(kind)}
                  >
                    <ModeIcon size={16} aria-hidden="true" />
                    <span>{MODES[kind].label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <label className="drop-zone large">
            <Upload size={28} aria-hidden="true" />
            <span>{uploadState.file ? uploadState.file.name : `Upload ${activeMode.label}`}</span>
            <input
              key={activeKind}
              type="file"
              accept={activeMode.accept}
              onChange={onFileChange}
            />
          </label>

          <div className="file-row">
            <span>{uploadState.file ? fileSize : "No file selected"}</span>
            <button
              type="submit"
              disabled={!uploadState.file || uploadState.loading}
              title={`Check ${activeMode.label.toLowerCase()}`}
            >
              {uploadState.loading ? (
                <Loader2 className="spin" size={18} aria-hidden="true" />
              ) : (
                <Activity size={18} aria-hidden="true" />
              )}
              <span>Check Real/Fake</span>
            </button>
          </div>
        </form>

        <section className="panel analysis-panel" aria-label="Analysis result">
          <div className="panel-header">
            <div className="panel-title">
              <CheckCircle2 size={20} aria-hidden="true" />
              <h2>Result</h2>
            </div>
            <span className="thresholds">0.40 / 0.60</span>
          </div>

          <Preview kind={activeKind} url={uploadState.previewUrl} />

          {uploadState.error && (
            <div className="notice error" role="alert">
              <AlertCircle size={18} aria-hidden="true" />
              <span>{uploadState.error}</span>
            </div>
          )}

          {uploadState.result ? (
            <ResultPanel result={uploadState.result} />
          ) : (
            <div className="result empty-result">
              <p>Awaiting check</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function Preview({ kind, url }: { kind: MediaKind; url: string | null }) {
  if (!url) {
    return <div className="preview empty">Awaiting media</div>;
  }

  if (kind === "image") {
    return <img className="preview media-preview" src={url} alt="Selected upload preview" />;
  }

  return <video className="preview media-preview" src={url} controls />;
}

function ResultPanel({ result }: { result: DetectionResponse }) {
  const scorePercent = Math.round(result.score * 100);
  const confidencePercent = Math.round(result.confidence * 100);
  const tone = result.prediction.includes("FAKE")
    ? "fake"
    : result.prediction.includes("REAL")
      ? "real"
      : "suspicious";
  const verdict = tone === "fake" ? "FAKE" : tone === "real" ? "REAL" : "SUSPICIOUS";

  return (
    <section className={`result ${tone}`} aria-label="Detection result">
      <div className="result-line">
        <div>
          <p className="result-label">Verdict</p>
          <strong>{verdict}</strong>
          <span>{result.prediction}</span>
        </div>
        <CheckCircle2 size={22} aria-hidden="true" />
      </div>

      <MetricBar label="Deepfake score" value={scorePercent} />
      <MetricBar label="Confidence" value={confidencePercent} />

      {typeof result.frames_used === "number" && (
        <p className="frames">Frames checked: {result.frames_used}</p>
      )}
    </section>
  );
}

function MetricBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric">
      <div className="metric-label">
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div className="bar" aria-hidden="true">
        <span style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}
