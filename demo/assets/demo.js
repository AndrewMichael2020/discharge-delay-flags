const data = window.DEMO_DATA;
let state = { scenario: data?.demo_summary?.default_scenario || 'balanced', filters: {} };
const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Number(n || 0).toLocaleString();
const fmtPct = (n) => `${(Number(n || 0) * 100).toFixed(1)}%`;
const fmtHours = (n) => `${Number(n || 0).toLocaleString(undefined,{maximumFractionDigits:0})}h`;
const fmtNum = (n,d=2) => Number(n || 0).toFixed(d);
const titleCase = (s) => String(s || '').replaceAll('_',' ').replace(/\b\w/g, m => m.toUpperCase());
const filterDefs = [
  ['filter_signal_family','Signal category'], ['filter_delay_reason','Specific delay reason'], ['filter_unit','Unit'], ['filter_service_line','Service line'],
  ['filter_shift','Shift'], ['filter_priority','Priority'], ['filter_owner','Owner']
];
function scenario(){ return data.scenarios.find(s => s.scenario_mode === state.scenario) || data.scenarios[0]; }
function labelFor(key, value){ const opt=(data.filters[state.scenario]?.[key]||[]).find(o=>o.value===value); return opt?.label || titleCase(value); }
function setFilter(key, value){ if(value) state.filters[key]=value; else delete state.filters[key]; renderAll(false); }
function signalDateSet(){
  return new Set((data.control_chart[state.scenario] || []).filter(r => r.control_chart_signal_flag).map(r => r.date));
}
function rowMatches(row){
  for(const [key] of filterDefs){
    if(!state.filters[key]) continue;
    const raw = row[key];
    const values = Array.isArray(raw) ? raw.map(String) : [String(raw ?? '')];
    if(!values.includes(String(state.filters[key]))) return false;
  }
  const rowDate = String(row.shift_date || row.date || '');
  if(state.filters.startDate && rowDate < state.filters.startDate) return false;
  if(state.filters.endDate && rowDate > state.filters.endDate) return false;
  if(state.filters.signalDaysOnly && !signalDateSet().has(rowDate)) return false;
  return true;
}
function currentRows(){
  return {
    worklist: (data.management_worklist[state.scenario]||[]).filter(rowMatches),
    actions: (data.action_examples[state.scenario]||[]).filter(rowMatches),
  };
}
function renderTabs(){
  $('scenarioTabs').innerHTML = data.scenarios.map(s => `<button class="scenario-tab ${s.scenario_mode===state.scenario?'active':''}" data-scenario="${s.scenario_mode}" type="button">${s.label}</button>`).join('');
  document.querySelectorAll('[data-scenario]').forEach(btn => btn.onclick = () => { state.scenario=btn.dataset.scenario; state.filters={}; renderAll(); });
}
function renderFilterControls(){
  const options = data.filters[state.scenario] || {};
  const selects = filterDefs.map(([key,label]) => `<div class="filter-field"><label for="f_${key}">${label}</label><select id="f_${key}" data-filter="${key}"><option value="">All</option>${(options[key]||[]).map(o=>`<option value="${o.value}" ${state.filters[key]===o.value?'selected':''}>${o.label}</option>`).join('')}</select></div>`).join('');
  const dates = options.date || [];
  const minDate = dates[0]?.value || ''; const maxDate = dates[dates.length-1]?.value || '';
  $('filterGrid').innerHTML = selects + `<div class="filter-field"><label for="startDate">Start date</label><input id="startDate" data-filter="startDate" type="date" min="${minDate}" max="${maxDate}" value="${state.filters.startDate||''}"></div><div class="filter-field"><label for="endDate">End date</label><input id="endDate" data-filter="endDate" type="date" min="${minDate}" max="${maxDate}" value="${state.filters.endDate||''}"></div><label class="check-row"><input id="signalDaysOnly" type="checkbox" ${state.filters.signalDaysOnly?'checked':''}> Control-chart signal days only</label>`;
  document.querySelectorAll('[data-filter]').forEach(el => el.onchange = e => setFilter(e.target.dataset.filter, e.target.value));
  $('signalDaysOnly').onchange = e => { if(e.target.checked) state.filters.signalDaysOnly='true'; else delete state.filters.signalDaysOnly; renderAll(false); };
}
function renderChips(){
  const chips=[];
  for(const [key,label] of filterDefs){ if(state.filters[key]) chips.push(`<span class="chip">${label}: ${labelFor(key,state.filters[key])}</span>`); }
  if(state.filters.startDate) chips.push(`<span class="chip">From ${state.filters.startDate}</span>`);
  if(state.filters.endDate) chips.push(`<span class="chip">To ${state.filters.endDate}</span>`);
  if(state.filters.signalDaysOnly) chips.push(`<span class="chip">Signal days only</span>`);
  $('activeChips').innerHTML = chips.length ? chips.join('') : '<span class="chip">All records</span>';
}
function renderSummary(){
  const s=scenario(); const rows=currentRows();
  $('scenarioSummary').innerHTML = `<p class="eyebrow">Selected case</p><h3>${s.label}</h3><p>${s.description}</p><div class="summary-list"><div class="summary-row"><span>Pattern</span><strong>${titleCase(s.detected_scenario)}</strong></div><div class="summary-row"><span>Top family</span><strong>${titleCase(s.top_signal_family)}</strong></div><div class="summary-row"><span>Filtered groups</span><strong>${fmtInt(rows.worklist.length)}</strong></div><div class="summary-row"><span>Filtered actions</span><strong>${fmtInt(rows.actions.length)}</strong></div></div>`;
  $('statusPill').innerHTML = `Filtered dashboard<strong>${fmtInt(rows.worklist.length)} groups / ${fmtInt(rows.actions.length)} actions</strong>`;
}
function renderKpis(){
  const s=scenario(); const rows=currentRows();
  const bedHours = rows.worklist.reduce((a,r)=>a+Number(r.recoverable_bed_hours||0),0);
  const cases = rows.worklist.reduce((a,r)=>a+Number(r.flagged_cases||0),0);
  const reasons = new Set(rows.worklist.map(r=>r.delay_reason)).size;
  const signalDays = filteredControlRows().filter(r=>r.control_chart_signal_flag).length;
  const cards = [['Encounters',fmtInt(s.admissions),'Scenario denominator'],['OOB rate',fmtPct(s.oob_signal_rate),'Scenario-level rate'],['Groups',fmtInt(rows.worklist.length),'After filters'],['Actions',fmtInt(rows.actions.length),'After filters'],['Cases',fmtInt(cases),'Flagged in groups'],['Bed-hours',fmtHours(bedHours),'Recoverable estimate'],['Reasons',fmtInt(reasons),'Visible reasons'],['Signal days',fmtInt(signalDays),'Control-chart days']];
  $('kpiGrid').innerHTML = cards.map(c=>`<article class="kpi-card"><span class="kpi-value" title="${c[1]}">${c[1]}</span><span class="kpi-label">${c[0]}</span><p>${c[2]}</p></article>`).join('');
}
function renderReasonChart(){
  const rows=currentRows().worklist;
  const by={}; rows.forEach(r=>{ const k=r.delay_reason||'Unattributed'; by[k] ||= {delay_reason:k, recoverable_bed_hours:0, flagged_cases:0}; by[k].recoverable_bed_hours += Number(r.recoverable_bed_hours||0); by[k].flagged_cases += Number(r.flagged_cases||0); });
  const list=Object.values(by).sort((a,b)=>b.recoverable_bed_hours-a.recoverable_bed_hours).slice(0,10);
  const max=Math.max(...list.map(r=>r.recoverable_bed_hours),1);
  $('reasonCount').textContent = `${fmtInt(list.length)} visible reasons`;
  $('reasonChart').innerHTML = list.length ? list.map(r=>{ const pct=Math.max(3,r.recoverable_bed_hours/max*100); return `<div class="bar-row"><div class="bar-label" title="${r.delay_reason}">${r.delay_reason}</div><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><div class="bar-value">${fmtHours(r.recoverable_bed_hours)}</div></div>`; }).join('') + `<div class="bar-axis"><span>0h</span><strong>Recoverable bed-hours</strong><span>${fmtHours(max)}</span></div>` : '<p>No records match the current filters.</p>';
}
function filteredControlRows(){
  let rows = (data.control_chart[state.scenario]||[]).filter(r => (!state.filters.startDate || r.date >= state.filters.startDate) && (!state.filters.endDate || r.date <= state.filters.endDate));
  if(state.filters.signalDaysOnly) rows = rows.filter(r => r.control_chart_signal_flag);
  return rows;
}
function renderControlChart(){
  const rows=filteredControlRows(); const svg=$('controlChart'); const w=560,h=300,p={t:18,r:20,b:30,l:42};
  const maxY=Math.max(...rows.map(r=>Number(r.oob_rate||0)),...rows.map(r=>Number(r.upper_control_limit||0)),0.1);
  const x=i=>p.l+(i/Math.max(rows.length-1,1))*(w-p.l-p.r); const y=v=>h-p.b-(Number(v||0)/maxY)*(h-p.t-p.b);
  const path=f=>rows.map((r,i)=>`${i?'L':'M'} ${x(i)} ${y(r[f])}`).join(' ');
  const ticks=[0,.25,.5,.75,1];
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  const yLabels = ticks.map(t => {
    const value = maxY * (1 - t);
    const yy = p.t + t * (h - p.t - p.b);
    return `<text x="${p.l-8}" y="${yy+3}" text-anchor="end" fill="#657386" font-size="9">${fmtPct(value)}</text>`;
  }).join('');
  svg.innerHTML = `<rect width="${w}" height="${h}" rx="7" fill="#fff"/>${ticks.map(t=>`<line x1="${p.l}" y1="${p.t+t*(h-p.t-p.b)}" x2="${w-p.r}" y2="${p.t+t*(h-p.t-p.b)}" stroke="#e8edf3"/>`).join('')}${yLabels}<line x1="${p.l}" y1="${h-p.b}" x2="${w-p.r}" y2="${h-p.b}" stroke="#cfd8e3"/><line x1="${p.l}" y1="${p.t}" x2="${p.l}" y2="${h-p.b}" stroke="#cfd8e3"/><path d="${path('upper_control_limit')}" fill="none" stroke="#c84e3a" stroke-width="2" stroke-dasharray="5 5"/><path d="${path('centerline_oob_rate')}" fill="none" stroke="#14866d" stroke-width="2" stroke-dasharray="3 4"/><path d="${path('oob_rate')}" fill="none" stroke="#1565a9" stroke-width="2.4"/>${rows.map((r,i)=>`<circle cx="${x(i)}" cy="${y(r.oob_rate)}" r="${r.control_chart_signal_flag?4:2.1}" fill="${r.control_chart_signal_flag?'#c84e3a':'#1565a9'}"><title>${r.date}: ${fmtPct(r.oob_rate)}</title></circle>`).join('')}<text x="${p.l}" y="${h-8}" fill="#657386" font-size="10">Date (${fmtInt(rows.length)} days)</text><text transform="translate(12 ${h/2}) rotate(-90)" text-anchor="middle" fill="#657386" font-size="10">Daily OOB signal rate</text><text x="${w-p.r}" y="${p.t+2}" text-anchor="end" fill="#657386" font-size="10">Observed vs threshold</text><g transform="translate(${p.l} ${p.t-6})"><circle r="3" fill="#1565a9"></circle><text x="8" y="3" font-size="9" fill="#657386">Observed</text><line x1="62" x2="78" y1="0" y2="0" stroke="#c84e3a" stroke-width="2" stroke-dasharray="4 4"></line><text x="84" y="3" font-size="9" fill="#657386">Review threshold</text><line x1="172" x2="188" y1="0" y2="0" stroke="#14866d" stroke-width="2" stroke-dasharray="3 3"></line><text x="194" y="3" font-size="9" fill="#657386">Expected baseline</text></g>`;
}
function cell(value, cls=''){ return `<td class="${cls}" title="${String(value??'').replaceAll('"','&quot;')}">${value??''}</td>`; }
function renderTable(id, rows, columns){ $(id).innerHTML = `<thead><tr>${columns.map(c=>`<th style="width:${c.w||'auto'}">${c.label}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>cell(c.format?c.format(r[c.key],r):(r[c.key]??''),c.truncate?'truncate':'')).join('')}</tr>`).join('')}</tbody>`; }
function renderTables(){
  const rows=currentRows(); $('worklistCount').textContent=`${fmtInt(rows.worklist.length)} groups`; $('actionCount').textContent=`${fmtInt(rows.actions.length)} actions`;
  renderTable('worklistTable', rows.worklist.slice(0,80), [{key:'signal_group_rank',label:'Rank',w:'48px'},{key:'shift_date',label:'Date',w:'78px'},{key:'shift_display',label:'Shift',w:'72px'},{key:'delay_reason',label:'Delay reason',truncate:true},{key:'flagged_cases',label:'Cases',w:'64px'},{key:'recoverable_bed_hours',label:'Bed-hours',format:fmtHours,w:'82px'},{key:'recommended_owner_display',label:'Owner',truncate:true}]);
  renderTable('actionTable', rows.actions.slice(0,80), [{key:'executive_rank',label:'Rank',w:'48px'},{key:'priority_display',label:'Priority',format:v=>`<span class="priority-pill">${v}</span>`,w:'76px'},{key:'delay_reason',label:'Delay reason',truncate:true},{key:'unit_display',label:'Unit',truncate:true,w:'110px'},{key:'evidence_summary',label:'Evidence',truncate:true},{key:'recommended_action',label:'Next step',truncate:true}]);
}
function renderAll(rebuildFilters=true){ renderTabs(); if(rebuildFilters) renderFilterControls(); renderChips(); renderSummary(); renderKpis(); renderReasonChart(); renderControlChart(); renderTables(); }
if(!data){ document.body.innerHTML='<main class="panel"><h1>Demo data missing</h1></main>'; } else { $('clearFilters').onclick=()=>{state.filters={};renderAll();}; renderAll(); }
