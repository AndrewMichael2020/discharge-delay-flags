const data = window.DEMO_DATA;
let activeScenario = data?.demo_summary?.default_scenario || 'balanced';

const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Number(n || 0).toLocaleString();
const fmtPct = (n) => `${(Number(n || 0) * 100).toFixed(1)}%`;
const fmtHours = (n) => `${Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}h`;
const fmtNum = (n, digits = 2) => Number(n || 0).toFixed(digits);
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

function renderScenarioSummary() {
  const s = scenario();
  $('scenarioSummary').innerHTML = `
    <p class="eyebrow">Selected case</p>
    <h3>${s.label}</h3>
    <p>${s.description}</p>
    <div class="summary-list">
      <div class="summary-row"><span>Detected pattern</span><strong>${titleCase(s.detected_scenario)}</strong></div>
      <div class="summary-row"><span>Top family</span><strong>${titleCase(s.top_signal_family)}</strong></div>
      <div class="summary-row"><span>Family share</span><strong>${fmtPct(s.top_family_share)}</strong></div>
      <div class="summary-row"><span>Compression</span><strong>${fmtNum(s.compression_ratio, 2)}x</strong></div>
    </div>
  `;
  $('statusPill').innerHTML = `Dashboard status<strong>${s.management_signal_groups} grouped signals</strong>`;
}

function renderKpis() {
  const s = scenario();
  const cards = [
    ['Encounters', fmtInt(s.admissions), '90-day synthetic case window'],
    ['OOB rate', fmtPct(s.oob_signal_rate), 'Encounters crossing a delay threshold'],
    ['Signals / shift', fmtNum(s.avg_signal_groups_per_shift, 2), 'Grouped agenda items per shift'],
    ['Raw signals', fmtInt(s.raw_oob_delay_signals), 'Patient-level evidence before grouping'],
    ['Top reason', s.top_management_delay_reason, 'Highest-ranked management signal'],
    ['ROC-AUC', fmtNum(s.roc_auc, 3), 'Local risk baseline'],
  ];
  $('kpiGrid').innerHTML = cards.map(([label, value, note]) => `
    <article class="kpi-card">
      <span class="kpi-value" title="${value}">${value}</span>
      <span class="kpi-label">${label}</span>
      <p>${note}</p>
    </article>
  `).join('');
}

function renderReasonChart() {
  const rows = (data.delay_reasons[activeScenario] || []).slice(0, 9);
  const max = Math.max(...rows.map((r) => Number(r.recoverable_bed_hours || 0)), 1);
  $('reasonChart').innerHTML = rows.map((r) => {
    const pct = Math.max(3, Number(r.recoverable_bed_hours || 0) / max * 100);
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
  const width = 520;
  const height = 292;
  const pad = { top: 18, right: 18, bottom: 28, left: 38 };
  const maxY = Math.max(...rows.map((r) => Number(r.oob_rate || 0)), ...rows.map((r) => Number(r.upper_control_limit || 0)), 0.1);
  const x = (i) => pad.left + (i / Math.max(rows.length - 1, 1)) * (width - pad.left - pad.right);
  const y = (v) => height - pad.bottom - (Number(v || 0) / maxY) * (height - pad.top - pad.bottom);
  const path = (field) => rows.map((r, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(r[field])}`).join(' ');
  const gridLines = [0, 0.25, 0.5, 0.75, 1].map((g) => {
    const yy = pad.top + g * (height - pad.top - pad.bottom);
    return `<line x1="${pad.left}" y1="${yy}" x2="${width - pad.right}" y2="${yy}" stroke="#e8edf3" />`;
  }).join('');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="10" fill="#ffffff"></rect>
    ${gridLines}
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#cfd8e3"></line>
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#cfd8e3"></line>
    <path d="${path('upper_control_limit')}" fill="none" stroke="#c84e3a" stroke-width="2" stroke-dasharray="5 5"></path>
    <path d="${path('centerline_oob_rate')}" fill="none" stroke="#1f7a55" stroke-width="2" stroke-dasharray="3 4"></path>
    <path d="${path('oob_rate')}" fill="none" stroke="#1769aa" stroke-width="2.5"></path>
    ${rows.map((r, i) => `<circle cx="${x(i)}" cy="${y(r.oob_rate)}" r="${r.control_chart_signal_flag ? 4 : 2.2}" fill="${r.control_chart_signal_flag ? '#c84e3a' : '#1769aa'}"><title>${r.date}: ${fmtPct(r.oob_rate)}</title></circle>`).join('')}
    <text x="${pad.left}" y="${height - 8}" fill="#637083" font-size="10">90-day window</text>
    <text x="${width - pad.right}" y="${pad.top + 2}" text-anchor="end" fill="#637083" font-size="10">OOB rate</text>
  `;
}

function cell(value, className = '') {
  return `<td class="${className}" title="${String(value ?? '').replaceAll('"', '&quot;')}">${value ?? ''}</td>`;
}

function renderTable(id, rows, columns) {
  const table = $(id);
  table.innerHTML = `
    <thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join('')}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${columns.map((c) => {
      const value = c.format ? c.format(r[c.key], r) : (r[c.key] ?? '');
      return cell(value, c.truncate ? 'truncate' : '');
    }).join('')}</tr>`).join('')}</tbody>
  `;
}

function renderTables() {
  renderTable('worklistTable', (data.management_worklist[activeScenario] || []).slice(0, 8), [
    { key: 'signal_group_rank', label: 'Rank' },
    { key: 'shift_date', label: 'Date' },
    { key: 'shift_name', label: 'Shift', format: titleCase },
    { key: 'delay_reason', label: 'Delay reason', truncate: true },
    { key: 'flagged_cases', label: 'Cases' },
    { key: 'recoverable_bed_hours', label: 'Bed-hours', format: fmtHours },
    { key: 'recommended_owner', label: 'Owner', format: titleCase, truncate: true },
  ]);
  renderTable('actionTable', (data.action_examples[activeScenario] || []).slice(0, 8), [
    { key: 'executive_rank', label: 'Rank' },
    { key: 'priority', label: 'Priority', format: (v) => `<span class="priority-pill">${titleCase(v)}</span>` },
    { key: 'delay_reason', label: 'Delay reason', truncate: true },
    { key: 'unit_id', label: 'Unit', format: titleCase },
    { key: 'evidence_summary', label: 'Evidence', truncate: true },
    { key: 'recommended_action', label: 'Next step', truncate: true },
  ]);
}

function renderGlossary() {
  $('glossaryGrid').innerHTML = data.demo_summary.metric_glossary.map((g) => `
    <article class="glossary-card"><strong>${g.metric}</strong><span>${g.meaning}</span></article>
  `).join('');
}

function renderAll() {
  renderTabs();
  renderScenarioSummary();
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
