import { ExternalLink, Target, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import type { MatchResult } from '../App';

interface ResultCardProps {
  match: MatchResult;
}

const SCORE_LABELS: Record<string, { label: string; color: string }> = {
  primary_technique: { label: 'Primary Technique', color: '#58a6ff' },
  secondary_overlap: { label: 'Secondary Tags', color: '#a371f7' },
  core_insight_sim: { label: 'Core Insight', color: '#3fb950' },
  composition_sim: { label: 'Problem Structure', color: '#f0883e' },
  constraint_match: { label: 'Constraints', color: '#ff7b72' },
};

export default function ResultCard({ match }: ResultCardProps) {
  const [expanded, setExpanded] = useState(false);
  const scorePercentage = Math.round(match.total_score * 100);

  return (
    <div className="glass-panel result-card">
      <a href={match.url} target="_blank" rel="noopener noreferrer" className="result-card-link">
        <div className="result-card-header">
          <div>
            <h3 className="result-title">{match.title}</h3>
            <span className="result-platform">{match.platform}</span>
          </div>
          <div className="result-score" title="Similarity Score">
            {scorePercentage}% Match
          </div>
        </div>

        <div className="tag-container">
          <span className="tag primary" title="Primary Technique">
            <Target size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-1px' }} />
            {match.primary_technique}
          </span>
          {match.secondary_techniques.map((tag, idx) => (
            <span key={idx} className="tag">{tag}</span>
          ))}
        </div>

        {/* Relationship reason — the "why" */}
        <div className="relationship-reason">
          {match.relationship_reason}
        </div>

        <div className="insight-box">
          <strong>Core Insight:</strong> {match.core_insight}
        </div>
      </a>

      {/* Expandable score breakdown */}
      <button
        className="expand-button"
        onClick={(e) => {
          e.preventDefault();
          setExpanded(!expanded);
        }}
      >
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        {expanded ? 'Hide' : 'Show'} Score Breakdown
      </button>

      {expanded && match.score_breakdown && (
        <div className="score-breakdown">
          {Object.entries(SCORE_LABELS).map(([key, { label, color }]) => {
            const value = (match.score_breakdown as any)[key] ?? 0;
            const percentage = Math.round(value * 100);
            return (
              <div key={key} className="score-row">
                <span className="score-label">{label}</span>
                <div className="score-bar-track">
                  <div
                    className="score-bar-fill"
                    style={{ width: `${percentage}%`, backgroundColor: color }}
                  />
                </div>
                <span className="score-value" style={{ color }}>
                  {percentage}%
                </span>
              </div>
            );
          })}
        </div>
      )}

      <a
        href={match.url}
        target="_blank"
        rel="noopener noreferrer"
        className="view-problem-link"
      >
        View Problem <ExternalLink size={14} />
      </a>
    </div>
  );
}
