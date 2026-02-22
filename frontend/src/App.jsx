import { Routes, Route, Link } from 'react-router-dom'
import AskPage from './pages/AskPage'
import HistoryPage from './pages/HistoryPage'
import ArtifactsPage from './pages/ArtifactsPage'
import EvaluationPage from './pages/EvaluationPage'
import './App.css'

function App() {
  return (
    <div className="app">
      <nav className="nav">
        <Link to="/" className="nav-brand">Personal Research Portal</Link>
        <div className="nav-links">
          <Link to="/">Ask</Link>
          <Link to="/history">History</Link>
          <Link to="/artifacts">Artifacts</Link>
          <Link to="/evaluation">Evaluation</Link>
        </div>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<AskPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/artifacts" element={<ArtifactsPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
