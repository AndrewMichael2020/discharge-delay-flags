const data = window.DEMO_DATA;
let activeScenario = data?.demo_summary?.default_scenario || 'balanced';

const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Number(n || 0).toLocaleString();
const fmtPct = (n) => `${(Number(n || 0) * 100).toFixed(1)}%`;
const fmtHours = (n) => `${Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}h`;
const titleCase = (s) => String(s || '').replaceAll('_', ' ').replace(/\b\w/g, (m) => m.toUpperCase());

function scenario() {
  return data.scenarios.find((s) => s.scenario_mode === activeScenario) || data.scenarios[0];
}

function renderTabs() {
  const tabs = $('scenarioTabs');
  tabs.innerHTML = '';
  data.scenarios.forEach((s) => {
    const button = document.createElement('button');
    button.className = `scenario-tab${s.scenario_mode === activeScenario ? ' active' : ''}`;
    button.type = 'button';
    button.textContent = s.label;
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-selected', s.scenario_mode === activeScenario ? 'true' : 'false');
    button.addEventListener('click', () => {
      activeScenario = s.scenario_mode;
      renderAll();
    });
    tabs.appendChild(button);
  });
}

function renderKpis() {
  const s = scenario();
  const cards = [
    ['Encounters', fmtInt(s.admissions), 'Synthetic admissions in the case window'],
    ['OOB signal rate', fmtPct(s.oob_signal_rate), 'Share of encounters crossing a delay threshold'],
    ['Signals / shift', Number(s.avg_signal_groups_per_shift).toFixed(2), 'Average grouped signals for huddle review'],
    ['Raw signals', fmtInt(s.raw_oob_delay_signals), 'Patient-level evidence before grouping'],
    ['Top delay reason', s.top_management_delay_reason, 'Highest-ranked management reason'],
    ['Model ROC-AUC', Number(s.roc_auc || 0).toFixed(3), 'Local statistical risk baseline'],
  ];
  $('kpiGrid').innerHTML = cards.map(([label, value, note]) => `
    <article class="kpi-card">
      <span class="kpi-value">${value}</span>
      <span class="kpi-label">${label}</span>
      <p>${note}</p>
    </article>
  `).join('');
}

function renderReasonChart() {
  const rows = (data.delay_reasons[activeScenario] || []).slice(0, 10);
  const max = Math.max(...rows.map((r) => Number(r.recoverable_bed_hours || 0)), 1);
  $('reasonChart').innerHTML = rows.map((r) => {
    const pct = Math.max(4, Number(r.recoverable_bed_hours || 0) / max * 100);
    return `
      <div class="bar-row" title="${r.flagged_cases} cases, ${titleCase(r.signal_family)}">
        <div class="bar-label">${r.delay_reason}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <div class="bar-value">${fmtHours(r.recoverable_bed_hours)}</div>
      </div>
    `;
  }).join('');
}

function renderControlChart() {
  const rows = data.control_chart[activeScenario] || [];
  const svg = $('controlChart');
  const width = 720;
  const height = 340;
  const pad = { top: 24, right: 26, bottom: 42, left: 48 };
  const maxY = Math.max(...rows.map((r) => Number(r.oob_rate || 0)), ...rows.map((r) => Number(r.upper_control_limit || 0)), 0.1);
  const x = (i) => pad.left + (i / Math.max(rows.length - 1, 1)) * (width - pad.left - pad.right);
  const y = (v) => height - pad.bottom - (Number(v || 0) / maxY) * (height - pad.top - pad.bottom);
  const ratePath = rows.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(r.oob_rate)}`).join(' ');
  const uclPath = rows.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(r.upper_control_limit)}`).join(' ');
  const centerPath = rows.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(r.centerline_oob_rate)}`).join(' ');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="22" fill="rgba(255,255,255,0.42)"></rect>
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="rgba(23,33,29,.18)"></line>
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="rgba(23,33,29,.18)"></line>
    <path d="${uclPath}" fill="none" stroke="#d95d39" stroke-width="2" stroke-dasharray="6 6"></path>
    <path d="${centerPath}" fill="none" stroke="#1f6f57" stroke-width="2" stroke-dasharray="4 5" opacity=".8"></path>
    <path d="${ratePath}" fill="none" stroke="#244f73" stroke-width="3"></path>
    ${rows.map((r, i) => `<circle cx="${x(i)}" cy="${y(r.oob_rate)}" r="${r.control_chart_signal_flag ? 5 : 3}" fill="${r.control_chart_signal_flag ? '#d95d39' : '#244f73'}"><title>${r.date}: ${fmtPct(r.oob_rate)}</title></circle>`).join('')}
    <text x="${pad.left}" y="${height - 12}" fill="#5b675f" font-size="13">90-day case window</text>
    <text x="${width - pad.right}" y="${pad.top + 4}" text-anchor="end" fill="#5b675f" font-size="13">OOB signal rate</text>
  `;
}

function renderTable(id, rows, columns) {
  const table = $(id);
  table.innerHTML = `
    <thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join('')}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${columns.map((c) => `<td>${c.format ? c.format(r[c.key], r) : (r[c.key] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody>
  `;
}

function renderTables() {
  renderTable('worklistTable', data.management_worklist[activeScenario] || [], [
    { key: 'signal_group_rank', label: 'Rank' },
    { key: 'shift_date', label: 'Shift date' },
    { key: 'shift_name', label: 'Shift' },
    { key: 'delay_reason', label: 'Delay reason' },
    { key: 'flagged_cases', label: 'Cases' },
    { key: 'recoverable_bed_hours', label: 'Bed-hours', format: fmtHours },
    { key: 'recommended_owner', label: 'Suggested owner', format: (_, r) => titleCase(r.recommended_owner) },
  ]);
  renderTable('actionTable', data.action_examples[activeScenario] || [], [
    { key: 'executive_rank', label: 'Rank' },
    { key: 'priority', label: 'Priority', format: titleCase },
    { key: 'delay_reason', label: 'Delay reason' },
    { key: 'unit_id', label: 'Unit', format: titleCase },
    { key: 'evidence_summary', label: 'Evidence' },
    { key: 'recommended_action', label: 'Recommended next step' },
  ]);
}

function renderGlossary() {
  $('glossaryGrid').innerHTML = data.demo_summary.metric_glossary.map((g) => `
    <article class="glossary-card"><strong>${g.metric}</strong><span>${g.meaning}</span></article>
  `).join('');
}

function renderAll() {
  renderTabs();
  renderKpis();
  renderReasonChart();
  renderControlChart();
  renderTables();
  renderGlossary();
}

if (!data) {
  document.body.innerHTML = '<main class="panel"><h1>Demo data missing</h1><p>Run <code>python3 scripts/build_demo.py</code> to generate demo/data files.</p></main>';
} else {
  renderAll();
}
