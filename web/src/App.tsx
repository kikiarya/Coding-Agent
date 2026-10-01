import { useEffect, useMemo, useState, type FormEvent } from "react";

type HarnessPolicy = "adaptive_v1" | "static_v1";
type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

type Run = {
  run_id: string;
  status: RunStatus;
  task_id: string;
  repo_path: string;
  issue: string;
  harness_policy: HarnessPolicy;
  run_dir?: string | null;
  error?: string | null;
  result?: Record<string, unknown> | null;
};

type AgentEvent = {
  seq: number;
  type: string;
  timestamp: string;
  payload: Record<string, unknown>;
};

type Artifacts = {
  available: string[];
  run: Record<string, unknown>;
  steps: Record<string, unknown>[];
  episodes: Record<string, unknown>[];
  checkpoints: Record<string, unknown>[];
  evaluator: Record<string, unknown>;
  diff: string;
};

type Trace = {
  timeline: {
    step_index: number;
    episode_id?: string;
    action?: Record<string, unknown>;
    tool_result?: Record<string, unknown>;
    test_state?: Record<string, unknown> | null;
    latency_ms?: number;
    tool_latency_ms?: number;
    input_tokens?: number;
    output_tokens?: number;
    labels?: string[];
  }[];
  summary: Record<string, unknown>;
};

const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail ?? `Request failed with ${response.status}`);
  }
  return body as T;
};

const pretty = (value: unknown) => JSON.stringify(value, null, 2);

function StatusPill({ status }: { status: RunStatus }) {
  return <span className={`status status-${status}`}>{status}</span>;
}

function RunLauncher({ onStarted }: { onStarted: (run: Run) => void }) {
  const [repoPath, setRepoPath] = useState("");
  const [issue, setIssue] = useState("");
  const [taskId, setTaskId] = useState("web_task");
  const [policy, setPolicy] = useState<HarnessPolicy>("adaptive_v1");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const run = await api<Run>("/runs", {
        method: "POST",
        body: JSON.stringify({
          repo_path: repoPath,
          issue,
          task_id: taskId,
          harness_policy: policy,
        }),
      });
      onStarted(run);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to start run");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="launcher card" onSubmit={submit}>
      <div className="eyebrow">RUN LAUNCHER</div>
      <h1>让 Agent 在受控仓库里工作</h1>
      <p className="muted">
        仅接受干净的 Git worktree。浏览器不会接触 API Key，也不能提交任意命令。
      </p>
      <label>
        Git worktree 路径
        <input value={repoPath} onChange={(e) => setRepoPath(e.target.value)} placeholder="E:\\workspace\\my-repo" required />
      </label>
      <label>
        Issue / 任务描述
        <textarea value={issue} onChange={(e) => setIssue(e.target.value)} placeholder="修复 xxx，并运行相关测试" rows={5} required />
      </label>
      <div className="form-row">
        <label>
          Task ID
          <input value={taskId} onChange={(e) => setTaskId(e.target.value)} required />
        </label>
        <label>
          Harness Policy
          <select value={policy} onChange={(e) => setPolicy(e.target.value as HarnessPolicy)}>
            <option value="adaptive_v1">adaptive_v1</option>
            <option value="static_v1">static_v1</option>
          </select>
        </label>
      </div>
      {error && <div className="error-box">{error}</div>}
      <button className="primary" disabled={busy} type="submit">
        {busy ? "正在提交…" : "启动受控 Run"}
      </button>
    </form>
  );
}

function RunConsole({ run, onReset }: { run: Run; onReset: () => void }) {
  const [current, setCurrent] = useState(run);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [artifacts, setArtifacts] = useState<Artifacts | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState("");

  const terminal = useMemo(
    () => ["completed", "failed", "cancelled"].includes(current.status),
    [current.status],
  );

  useEffect(() => {
    let source: EventSource | undefined;
    let active = true;
    const poll = async () => {
      try {
        const next = await api<Run>(`/runs/${run.run_id}`);
        if (active) setCurrent(next);
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "Unable to read run");
      }
    };
    source = new EventSource(`/runs/${run.run_id}/events`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as AgentEvent;
      setEvents((previous) => previous.some((item) => item.seq === event.seq) ? previous : [...previous, event]);
      if (event.type === "web_run_finished" || event.type === "web_run_failed") void poll();
    };
    source.onerror = () => {
      // The status poll remains authoritative if the browser reconnects.
      void poll();
    };
    const timer = window.setInterval(() => {
      void poll();
      if (terminal) window.clearInterval(timer);
    }, 2000);
    void poll();
    return () => {
      active = false;
      source?.close();
      window.clearInterval(timer);
    };
  }, [run.run_id, terminal]);

  useEffect(() => {
    if (!terminal) return;
    void Promise.all([
      api<Artifacts>(`/runs/${run.run_id}/artifacts`),
      api<Trace>(`/runs/${run.run_id}/trace`),
    ]).then(([nextArtifacts, nextTrace]) => {
      setArtifacts(nextArtifacts);
      setTrace(nextTrace);
    }).catch((reason) => setError(String(reason)));
  }, [run.run_id, terminal]);

  async function cancel() {
    try {
      setCurrent(await api<Run>(`/runs/${run.run_id}/cancel`, { method: "POST" }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to cancel run");
    }
  }

  async function resume() {
    try {
      setCurrent(await api<Run>(`/runs/${run.run_id}/resume`, { method: "POST" }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to resume run");
    }
  }

  return (
    <main className="console">
      <header className="topbar">
        <div>
          <div className="eyebrow">CAPO CONTROL PLANE · {current.run_id}</div>
          <h1>{current.task_id}</h1>
        </div>
        <div className="top-actions">
          <StatusPill status={current.status} />
          {!terminal && <button onClick={cancel}>请求停止</button>}
          {terminal && ["failed", "cancelled"].includes(current.status) && <button onClick={resume}>从 checkpoint 恢复</button>}
          <button onClick={onReset}>新建 Run</button>
        </div>
      </header>
      <section className="summary-grid">
        <div className="card summary-card"><span>Harness</span><strong>{current.harness_policy}</strong></div>
        <div className="card summary-card"><span>Repository</span><strong title={current.repo_path}>{current.repo_path}</strong></div>
        <div className="card summary-card"><span>Artifacts</span><strong>{artifacts?.available.join(", ") || "运行中"}</strong></div>
      </section>
      {error && <div className="error-box">{error}</div>}
      {current.error && <div className="error-box">{current.error}</div>}
      <section className="workspace-grid">
        <div className="card timeline-card">
          <div className="section-heading"><h2>Live Timeline</h2><span>{events.length} events</span></div>
          <div className="timeline">
            {events.length === 0 && <p className="muted">等待 Agent 事件…</p>}
            {events.map((event) => (
              <article className="event" key={event.seq}>
                <div className="event-marker" />
                <div className="event-body">
                  <div className="event-title"><strong>{event.type}</strong><time>{new Date(event.timestamp).toLocaleTimeString()}</time></div>
                  <pre>{pretty(event.payload)}</pre>
                </div>
              </article>
            ))}
          </div>
        </div>
        <div className="right-column">
          <div className="card">
            <div className="section-heading"><h2>Trajectory</h2><span>{trace?.timeline.length ?? 0} steps</span></div>
            <div className="trace-list">
              {trace?.timeline.map((step) => (
                <article className="trace-step" key={`${step.episode_id ?? "episode"}-${step.step_index}`}>
                  <div><strong>Step {step.step_index + 1}</strong> · {String(step.action?.type ?? "unknown")}</div>
                  <span className="muted">
                    {step.tool_result?.ok === false ? "FAIL" : String(step.test_state?.status ?? "executed")}
                    {` · ${step.tool_latency_ms ?? 0}ms tool · ${step.latency_ms ?? 0}ms model`}
                  </span>
                </article>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="section-heading"><h2>Issue</h2><span>{current.task_id}</span></div>
            <p className="issue">{current.issue}</p>
          </div>
          <div className="card">
            <div className="section-heading"><h2>Review Panel</h2><span>{terminal ? "final" : "live"}</span></div>
            {artifacts?.episodes?.at(-1) ? <pre>{pretty(artifacts.episodes.at(-1))}</pre> : <p className="muted">Run 完成后显示 Episode、测试和评测结果。</p>}
          </div>
          <div className="card diff-card">
            <div className="section-heading"><h2>Diff</h2><span>artifact-backed</span></div>
            <pre className="diff">{artifacts?.diff || "暂无 inspect_diff 产物。"}</pre>
          </div>
        </div>
      </section>
    </main>
  );
}

export function App() {
  const [run, setRun] = useState<Run | null>(null);
  return run ? <RunConsole run={run} onReset={() => setRun(null)} /> : <div className="shell"><RunLauncher onStarted={setRun} /></div>;
}
