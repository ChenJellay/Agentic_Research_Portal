import { useState, useEffect } from 'react'
import { listThreads, getThread, generateEvidenceTable, generateAnnotatedBib, generateSynthesisMemo, exportArtifact } from '../api'

function ArtifactsPage() {
  const [threads, setThreads] = useState([])
  const [selectedThreadId, setSelectedThreadId] = useState('')
  const [artifactType, setArtifactType] = useState('evidence-table')
  const [artifact, setArtifact] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    listThreads()
      .then((data) => setThreads(data.threads || []))
      .catch((err) => setError(err.message))
  }, [])

  const handleGenerate = async () => {
    if (!selectedThreadId) {
      setError('Select a thread first')
      return
    }
    setLoading(true)
    setError(null)
    setArtifact(null)
    try {
      let data
      if (artifactType === 'evidence-table') {
        data = await generateEvidenceTable(selectedThreadId)
      } else if (artifactType === 'annotated-bib') {
        data = await generateAnnotatedBib(selectedThreadId)
      } else {
        data = await generateSynthesisMemo(selectedThreadId)
      }
      setArtifact(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleExport = async (format) => {
    if (!selectedThreadId) return
    setError(null)
    try {
      await exportArtifact(artifactType, format, selectedThreadId)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <>
      <h1 className="page-title">Generate Artifacts</h1>
      <div className="artifact-form">
        <select
          value={selectedThreadId}
          onChange={(e) => setSelectedThreadId(e.target.value)}
        >
          <option value="">Select a thread…</option>
          {threads.map((t) => (
            <option key={t.thread_id} value={t.thread_id}>
              {t.query?.slice(0, 60)}…
            </option>
          ))}
        </select>
        <select
          value={artifactType}
          onChange={(e) => setArtifactType(e.target.value)}
        >
          <option value="evidence-table">Evidence Table</option>
          <option value="annotated-bib">Annotated Bibliography</option>
          <option value="synthesis-memo">Synthesis Memo</option>
        </select>
        <button
          onClick={handleGenerate}
          disabled={loading || !selectedThreadId}
        >
          {loading ? 'Generating…' : 'Generate'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {artifact && (
        <>
          <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem' }}>
            <button onClick={() => handleExport('md')}>Export Markdown</button>
            {artifactType !== 'synthesis-memo' && (
              <button onClick={() => handleExport('csv')}>Export CSV</button>
            )}
            <button onClick={() => handleExport('pdf')}>Export PDF</button>
          </div>
          <div className="answer-box">
            {artifact.markdown || artifact.content || JSON.stringify(artifact, null, 2)}
          </div>
        </>
      )}
    </>
  )
}

export default ArtifactsPage
