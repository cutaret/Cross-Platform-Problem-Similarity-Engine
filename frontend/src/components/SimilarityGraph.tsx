import { useRef, useEffect, useCallback, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import type { MatchResult } from '../App';

interface SimilarityGraphProps {
  queryTitle: string;
  matches: MatchResult[];
}

// Vibrant color palette for nodes based on technique
const TECHNIQUE_COLORS: Record<string, string> = {
  'greedy': '#3fb950',
  'dp': '#a371f7',
  'dp-bitmask': '#bc8cff',
  'binary-search': '#58a6ff',
  'graphs': '#f0883e',
  'math': '#ff7b72',
  'strings': '#ffd700',
  'sorting': '#79c0ff',
  'constructive': '#56d364',
  'two-pointers': '#db61a2',
  'segment-tree': '#f78166',
  'dfs-bfs': '#ff9b54',
  'divide-conquer': '#d2a8ff',
  'number-theory': '#ff6b6b',
};

function getTechniqueColor(technique: string): string {
  // Check for exact match
  if (TECHNIQUE_COLORS[technique]) return TECHNIQUE_COLORS[technique];
  // Check for partial match
  for (const [key, color] of Object.entries(TECHNIQUE_COLORS)) {
    if (technique.includes(key) || key.includes(technique)) return color;
  }
  // Deterministic hash-based color for unknown techniques
  let hash = 0;
  for (let i = 0; i < technique.length; i++) {
    hash = technique.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 60%)`;
}

interface GraphData {
  nodes: Array<{
    id: string;
    label: string;
    score: number;
    isQuery: boolean;
    url?: string;
    technique?: string;
    insight?: string;
    color: string;
    val: number; // node size
  }>;
  links: Array<{
    source: string;
    target: string;
    reason: string;
    score: number;
    color: string;
  }>;
}

export default function SimilarityGraph({ queryTitle, matches }: SimilarityGraphProps) {
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Track container size
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: Math.max(600, window.innerHeight * 0.65),
        });
      }
    });
    observer.observe(containerRef.current);
    // Initial size
    setDimensions({
      width: containerRef.current.clientWidth,
      height: Math.max(600, window.innerHeight * 0.65),
    });
    return () => observer.disconnect();
  }, []);

  // Build graph data
  const graphData: GraphData = useMemo(() => {
    const nodes = [
      {
        id: 'query',
        label: queryTitle || 'Your Problem',
        score: 1,
        isQuery: true,
        color: '#58a6ff',
        val: 30,
        technique: 'query',
      },
      ...matches.map((m) => ({
        id: `p-${m.problem_id}`,
        label: m.title.length > 25 ? m.title.slice(0, 23) + '…' : m.title,
        score: m.total_score,
        isQuery: false,
        url: m.url,
        technique: m.primary_technique,
        insight: m.core_insight,
        color: getTechniqueColor(m.primary_technique),
        val: 8 + m.total_score * 20,
      })),
    ];

    const links = matches.map((m) => ({
      source: 'query',
      target: `p-${m.problem_id}`,
      reason: m.relationship_reason,
      score: m.total_score,
      color: `rgba(${hexToRgb(getTechniqueColor(m.primary_technique))}, ${0.3 + m.total_score * 0.5})`,
    }));

    return { nodes, links };
  }, [queryTitle, matches]);

  // Auto-rotate camera on load
  useEffect(() => {
    if (!graphRef.current) return;
    const fg = graphRef.current;

    // Zoom to fit after simulation settles
    setTimeout(() => {
      fg.zoomToFit(400, 60);
    }, 800);

    // Set initial camera position — closer
    fg.cameraPosition({ x: 0, y: 0, z: 160 });

    // Start auto-rotation
    let angle = 0;
    const rotateInterval = setInterval(() => {
      angle += 0.003;
      fg.cameraPosition({
        x: 160 * Math.sin(angle),
        z: 160 * Math.cos(angle),
      });
    }, 30);

    // Stop auto-rotate on user interaction
    const stopRotate = () => clearInterval(rotateInterval);
    const canvas = fg.renderer()?.domElement;
    if (canvas) {
      canvas.addEventListener('mousedown', stopRotate, { once: true });
      canvas.addEventListener('touchstart', stopRotate, { once: true });
    }

    return () => clearInterval(rotateInterval);
  }, [graphData]);

  // Custom node rendering with glow
  const nodeThreeObject = useCallback((node: any) => {
    const group = new THREE.Group();

    // Main sphere
    const radius = node.isQuery ? 8 : 3 + node.score * 6;
    const geometry = new THREE.SphereGeometry(radius, 32, 32);
    const material = new THREE.MeshPhongMaterial({
      color: new THREE.Color(node.color),
      emissive: new THREE.Color(node.color),
      emissiveIntensity: node.isQuery ? 0.6 : 0.3,
      shininess: 80,
      transparent: true,
      opacity: 0.9,
    });
    const sphere = new THREE.Mesh(geometry, material);
    group.add(sphere);

    // Outer glow ring
    const glowRadius = radius * (node.isQuery ? 2.2 : 1.6);
    const glowGeom = new THREE.SphereGeometry(glowRadius, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(node.color),
      transparent: true,
      opacity: node.isQuery ? 0.12 : 0.06,
    });
    const glow = new THREE.Mesh(glowGeom, glowMat);
    group.add(glow);

    // Text label (using sprite)
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    const fontSize = node.isQuery ? 48 : 32;
    canvas.width = 512;
    canvas.height = 128;

    ctx.fillStyle = 'transparent';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.font = `${node.isQuery ? 'bold ' : ''}${fontSize}px Inter, sans-serif`;
    ctx.fillStyle = '#f0f6fc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Draw label
    ctx.fillText(node.label, 256, 40);

    // Draw score badge for non-query nodes
    if (!node.isQuery) {
      const scoreText = `${Math.round(node.score * 100)}%`;
      ctx.font = `bold ${fontSize - 4}px Inter, sans-serif`;
      ctx.fillStyle = node.color;
      ctx.fillText(scoreText, 256, 90);
    }

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(40, 10, 1);
    sprite.position.set(0, -(radius + 8), 0);
    group.add(sprite);

    return group;
  }, []);

  // Link styling
  const linkWidth = useCallback((link: any) => 0.5 + link.score * 3, []);
  const linkOpacity = useCallback((link: any) => 0.2 + link.score * 0.6, []);

  // Node click handler
  const handleNodeClick = useCallback((node: any) => {
    if (!node.isQuery && node.url) {
      window.open(node.url, '_blank');
    }
  }, []);

  // Link particles for animation
  const linkDirectionalParticles = useCallback((link: any) => {
    return Math.ceil(link.score * 4);
  }, []);

  const linkDirectionalParticleSpeed = useCallback((link: any) => {
    return 0.002 + link.score * 0.006;
  }, []);

  return (
    <div className="graph-container-3d glass-panel" ref={containerRef}>
      <ForceGraph3D
        ref={graphRef}
        graphData={graphData}
        nodeThreeObject={nodeThreeObject}
        nodeLabel={(node: any) =>
          node.isQuery
            ? `<div class="graph3d-tooltip"><strong>Your Query</strong><br/>${node.label}</div>`
            : `<div class="graph3d-tooltip"><strong>${node.label}</strong><br/>Technique: ${node.technique}<br/>Match: ${Math.round(node.score * 100)}%<br/><em>${node.insight?.slice(0, 80)}...</em><br/><span style="color:#58a6ff">Click to open →</span></div>`
        }
        linkWidth={linkWidth as any}
        linkOpacity={linkOpacity as any}
        linkColor={((link: any) => link.color) as any}
        linkDirectionalParticles={linkDirectionalParticles as any}
        linkDirectionalParticleSpeed={linkDirectionalParticleSpeed as any}
        linkDirectionalParticleColor={((link: any) => link.color) as any}
        linkDirectionalParticleWidth={2}
        onNodeClick={handleNodeClick}
        backgroundColor="rgba(0,0,0,0)"
        showNavInfo={false}
        width={dimensions.width}
        height={dimensions.height}
        d3VelocityDecay={0.3}
        d3AlphaDecay={0.02}
      />
      <div className="graph-3d-legend">
        <span className="legend-item">
          <span className="legend-dot" style={{ background: '#58a6ff' }}></span>
          Your Query
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ background: '#3fb950' }}></span>
          High Match
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ background: '#f0883e' }}></span>
          Medium Match
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ background: '#8b949e' }}></span>
          Low Match
        </span>
      </div>
    </div>
  );
}

// Helper to convert hex to rgb string
function hexToRgb(hex: string): string {
  // Handle hsl colors
  if (hex.startsWith('hsl')) return '136, 136, 136';

  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return '136, 136, 136';
  return `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}`;
}
