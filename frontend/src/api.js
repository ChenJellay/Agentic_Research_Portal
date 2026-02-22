const API_BASE = '/api';

export async function runQuery(question) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function searchOnly(query, topK = 5) {
  const res = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}&top_k=${topK}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listThreads() {
  const res = await fetch(`${API_BASE}/threads`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getThread(threadId) {
  const res = await fetch(`${API_BASE}/threads/${threadId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function generateEvidenceTable(threadId) {
  const res = await fetch(`${API_BASE}/artifacts/evidence-table`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function generateAnnotatedBib(threadId) {
  const res = await fetch(`${API_BASE}/artifacts/annotated-bib`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function generateSynthesisMemo(threadId) {
  const res = await fetch(`${API_BASE}/artifacts/synthesis-memo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function exportUrl(artifactType, format, threadId) {
  return `${API_BASE}/export/${format}?artifact_type=${artifactType}&thread_id=${threadId}`;
}

/**
 * Fetch export as blob and trigger download. Use instead of window.open to avoid
 * proxy errors with file downloads in dev (Vite proxy).
 */
export async function exportArtifact(artifactType, format, threadId) {
  const url = exportUrl(artifactType, format, threadId);
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  const blob = await res.blob();
  const disposition = res.headers.get('Content-Disposition');
  let filename = `export.${format === 'md' ? 'md' : format === 'csv' ? 'csv' : 'pdf'}`;
  const match = disposition?.match(/filename="?([^";\n]+)"?/);
  if (match) filename = match[1];
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

export async function runEvaluation() {
  const res = await fetch(`${API_BASE}/evaluation/run`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getEvaluationResults() {
  const res = await fetch(`${API_BASE}/evaluation/latest`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
