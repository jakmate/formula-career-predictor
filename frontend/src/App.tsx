import './App.css'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './components/Dashboard'

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Navigate to="/predictions" replace />} />
          <Route path="/predictions" element={<Dashboard />} />
          <Route path="/schedule" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App