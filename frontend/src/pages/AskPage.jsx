import { useState } from 'react'
import { runQuery } from '../api'

function AskPage() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [expandedSource, setExpandedSource] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await runQuery(question)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const toggleSource = (idx) => {
    setExpandedSource(expandedSource === idx ? null : idx)
  }

  return (
    <>
      <h1 className="page-title">Ask a Research Question</h1>
      <form className="search-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How does AI affect code review?"
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Searching…' : 'Ask'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="answer-box">{result.answer}</div>

          {result.suggested_queries && result.suggested_queries.length > 0 && (
            <div className="suggested-queries">
              <h4>Suggested next steps</h4>
              <ul>{result.suggested_queries.map((q, i) => <li key={i}>{q}</li>)}</ul>
            </div>
          )}

          <div className="sources-section">
            <h3>Sources & Citations</h3>
            {result.retrieved_chunks.map((chunk, idx) => (
              <div
                key={chunk.chunk_id}
                className={`source-card ${expandedSource === idx ? 'expanded' : ''}`}
                onClick={() => toggleSource(idx)}
              >
                <div className="source-header">
                  <span className="source-id">{chunk.source_id} / {chunk.chunk_id}</span>
                  {chunk.section && <span>{chunk.section}</span>}
                </div>
                <div className="source-text">{chunk.text}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}

export default AskPage
