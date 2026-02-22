import { useState, useEffect } from 'react'
import { listThreads, getThread } from '../api'

function HistoryPage() {
  const [threads, setThreads] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    listThreads()
      .then((data) => setThreads(data.threads || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  const loadThread = async (threadId) => {
    setError(null)
    try {
      const data = await getThread(threadId)
      setSelected(data)
    } catch (err) {
      setError(err.message)
    }
  }

  if (loading) return <div className="loading">Loading history…</div>
  if (error) return <div className="error">{error}</div>

  return (
    <>
      <h1 className="page-title">Research History</h1>
      {threads.length === 0 ? (
        <p>No research threads yet. Ask a question on the Ask page to get started.</p>
      ) : (
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          <ul className="thread-list" style={{ flex: '1', minWidth: '280px' }}>
            {threads.map((t) => (
              <li
                key={t.thread_id}
                className="thread-item"
                onClick={() => loadThread(t.thread_id)}
              >
                <div className="thread-query">{t.query}</div>
                <div className="thread-date">{t.created_at}</div>
              </li>
            ))}
          </ul>
          {selected && (
            <div className="thread-detail" style={{ flex: '1', minWidth: '300px' }}>
              <h3>Query</h3>
              <p>{selected.query}</p>
              <h3>Answer</h3>
              <div className="answer-box">{selected.answer}</div>
              {selected.suggested_queries?.length > 0 && (
                <>
                  <h3>Suggested next steps</h3>
                  <ul>{selected.suggested_queries.map((q, i) => <li key={i}>{q}</li>)}</ul>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </>
  )
}

export default HistoryPage
