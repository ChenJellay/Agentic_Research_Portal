import { useState, useEffect } from 'react'
import { runEvaluation, getEvaluationResults } from '../api'

function EvaluationPage() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingLatest, setLoadingLatest] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getEvaluationResults()
      .then((data) => setResults(data))
      .catch(() => setResults(null))
      .finally(() => setLoadingLatest(false))
  }, [])

  const handleRun = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await runEvaluation()
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loadingLatest && !results) return <div className="loading">Loading…</div>

  return (
    <>
      <h1 className="page-title">Evaluation</h1>
      <button
        onClick={handleRun}
        disabled={loading}
        style={{ marginBottom: '1rem' }}
      >
        {loading ? 'Running…' : 'Run Evaluation'}
      </button>

      {error && <div className="error">{error}</div>}

      {results?.summary && (
        <>
          <div className="eval-metrics">
            <div className="eval-metric">
              <div className="label">Total Queries</div>
              <div className="value">{results.summary.total_queries}</div>
            </div>
            <div className="eval-metric">
              <div className="label">Evaluated</div>
              <div className="value">{results.summary.evaluated}</div>
            </div>
            <div className="eval-metric">
              <div className="label">Avg Groundedness</div>
              <div className="value">{results.summary.avg_groundedness?.toFixed(3)}</div>
            </div>
            <div className="eval-metric">
              <div className="label">Avg Citation Precision</div>
              <div className="value">{results.summary.avg_citation_precision?.toFixed(3)}</div>
            </div>
            <div className="eval-metric">
              <div className="label">Failure Cases</div>
              <div className="value">{results.summary.failure_cases_count}</div>
            </div>
          </div>

          {results.summary.failure_cases?.length > 0 && (
            <div>
              <h3>Representative Failure Cases</h3>
              {results.summary.failure_cases.map((fc, i) => (
                <div key={i} className="failure-case">
                  <h4>{fc.query_id}: {fc.query?.slice(0, 80)}…</h4>
                  {fc.error && <p>{fc.error}</p>}
                  {fc.groundedness_score != null && (
                    <p>Groundedness: {fc.groundedness_score}, Citation Precision: {fc.citation_precision}</p>
                  )}
                  {fc.ungrounded_samples?.length > 0 && (
                    <p>Ungrounded: {fc.ungrounded_samples.join('; ')}</p>
                  )}
                  {fc.answer_snippet && <p>{fc.answer_snippet}…</p>}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {!results?.summary && !loading && (
        <p>No evaluation results yet. Click "Run Evaluation" to run the 22-query suite.</p>
      )}
    </>
  )
}

export default EvaluationPage
