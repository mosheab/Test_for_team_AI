import React, { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [matches, setMatches] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const ask = async (e) => {
    e.preventDefault()
    setLoading(true); setError(''); setAnswer(''); setMatches([])
    try {
      const res = await fetch(`${API_BASE}/chat/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setAnswer(data.answer)
      setMatches(data.matches || [])
    } catch (err) { setError(String(err)) }
    finally { setLoading(false) }
  }
  return (
    <div style={{ maxWidth: 800, margin: '40px auto', fontFamily: 'system-ui, Arial' }}>
      <h1>Video Highlight Chat</h1>
      <form onSubmit={ask} style={{ display: 'flex', gap: 8 }}>
        <input value={query} onChange={(e)=>setQuery(e.target.value)}
               placeholder="Ask about the videos (e.g., What occurred in the videos?)"
               style={{ flex: 1, padding: 10, fontSize: 16 }} />
        <button disabled={loading || !query.trim()} style={{ padding: '10px 16px' }}>
          {loading ? 'Searching…' : 'Ask'}
        </button>
      </form>
      {error && <p style={{ color: 'crimson' }}>{error}</p>}
      {answer && <div style={{ whiteSpace:'pre-wrap', background:'#f7f7f7', padding:12, borderRadius:8, marginTop:16 }}>{answer}</div>}
      {matches.length>0 && (
        <div style={{ marginTop: 16 }}>
          <h3>Matches</h3>
          <ul>{matches.map((m)=>(
            <li key={m.id} style={{ marginBottom:8 }}>
              <code>{m.filename}</code> [{sec(m.start_sec)}–{sec(m.end_sec)}] — {m.summary || m.title}
            </li>
          ))}</ul>
        </div>
      )}
    </div>
  )
}
function sec(s){
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = (s%60).toFixed(3).padStart(6,'0')
  return h ? `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}:${ss}` : `${m.toString().padStart(2,'0')}:${ss}`
}
