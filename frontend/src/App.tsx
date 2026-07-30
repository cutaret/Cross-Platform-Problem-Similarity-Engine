import { useState } from 'react';
import { Search, Sparkles, BrainCircuit, LayoutList, Network } from 'lucide-react';
import ResultCard from './components/ResultCard';
import SimilarityGraph from './components/SimilarityGraph';
import './index.css';

// Types
export interface ScoreBreakdown {
  primary_technique: number;
  secondary_overlap: number;
  composition_sim: number;
  core_insight_sim: number;
  constraint_match: number;
}

export interface MatchResult {
  problem_id: number;
  title: string;
  url: string;
  platform: string;
  primary_technique: string;
  secondary_techniques: string[];
  core_insight: string;
  total_score: number;
  score_breakdown: ScoreBreakdown;
  relationship_reason: string;
}

interface QuerySchema {
  primary_technique: string;
  core_insight: string;
  constraint_fingerprint: string;
}

type ViewMode = 'list' | 'graph';

export default function App() {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [querySchema, setQuerySchema] = useState<QuerySchema | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('list');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setError(null);
    setMatches([]);
    setQuerySchema(null);

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_n: 10 }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to analyze problem');
      }

      const data = await response.json();
      setMatches(data.matches);
      setQuerySchema(data.query_schema);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>CP Problem Finder</h1>
        <p>Paste a Codeforces URL or raw problem text to find conceptually similar problems.</p>
      </header>

      <form className="search-container" onSubmit={handleSearch}>
        <input
          type="text"
          className="search-input"
          placeholder="e.g., https://codeforces.com/problemset/problem/1900/A"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isSearching}
        />
        <button type="submit" className="search-button" disabled={isSearching || !query.trim()}>
          {isSearching ? <Sparkles size={20} className="animate-pulse" /> : <Search size={20} />}
          {isSearching ? 'Analyzing...' : 'Find Matches'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {isSearching && (
        <div className="loader-container">
          <div className="spinner"></div>
          <div className="loader-text">AI is decomposing the problem...</div>
        </div>
      )}

      {!isSearching && querySchema && (
        <div className="query-schema-box">
          <h3>
            <BrainCircuit size={18} />
            Extracted Algorithm Signature
          </h3>
          <div className="tag-container" style={{ marginBottom: '0.75rem' }}>
            <span className="tag primary">{querySchema.primary_technique}</span>
            <span className="tag">{querySchema.constraint_fingerprint}</span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            {querySchema.core_insight}
          </p>
        </div>
      )}

      {!isSearching && matches.length > 0 && (
        <div className="results-section">
          <div className="results-header">
            <h2>Top Matches</h2>
            <div className="view-toggle">
              <button
                className={`toggle-btn ${viewMode === 'list' ? 'active' : ''}`}
                onClick={() => setViewMode('list')}
                title="List View"
              >
                <LayoutList size={18} />
              </button>
              <button
                className={`toggle-btn ${viewMode === 'graph' ? 'active' : ''}`}
                onClick={() => setViewMode('graph')}
                title="Graph View"
              >
                <Network size={18} />
              </button>
            </div>
          </div>

          {viewMode === 'graph' && (
            <SimilarityGraph
              queryTitle={querySchema?.primary_technique || 'Your Problem'}
              matches={matches}
            />
          )}

          {viewMode === 'list' && (
            <div className="results-grid">
              {matches.map((match) => (
                <ResultCard key={match.problem_id} match={match} />
              ))}
            </div>
          )}
        </div>
      )}

      {!isSearching && matches.length === 0 && querySchema && (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
          <h3 style={{ color: 'var(--text-primary)', marginBottom: '1rem' }}>No matches found</h3>
          <p style={{ color: 'var(--text-secondary)' }}>
            The database might be empty or this problem is entirely unique. Try running the backfill command to index more problems.
          </p>
        </div>
      )}
    </div>
  );
}
