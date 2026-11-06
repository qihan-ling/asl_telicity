// stimuli.js — item list (NGT signs from Wagner et al. 2024) and Exp. 1A counterbalancing orders.
// To use other stimuli (e.g. ASL), replace ITEMS and drop the mp4 files into stimuli/,
// then rebuild data/references.json with precompute.html.

export const ITEMS = [
  { id: 'buy',     file: 'buy.mp4',     telicity: 'telic',  domain: 'social' },
  { id: 'decide',  file: 'decide.mp4',  telicity: 'telic',  domain: 'psych' },
  { id: 'die',     file: 'die.mp4',     telicity: 'telic',  domain: 'physical' },
  { id: 'discuss', file: 'discuss.mp4', telicity: 'atelic', domain: 'social' },
  { id: 'imagine', file: 'imagine.mp4', telicity: 'atelic', domain: 'psych' },
  { id: 'float',   file: 'float.mp4',   telicity: 'atelic', domain: 'physical' },
  { id: 'forget',  file: 'forget.mp4',  telicity: 'telic',  domain: 'psych' },
  { id: 'leave',   file: 'leave.mp4',   telicity: 'telic',  domain: 'physical' },
  { id: 'think',   file: 'think.mp4',   telicity: 'atelic', domain: 'psych' },
  { id: 'talk',    file: 'talk.mp4',    telicity: 'atelic', domain: 'social' },
  { id: 'run',     file: 'run.mp4',     telicity: 'atelic', domain: 'physical' },
  { id: 'sell',    file: 'sell.mp4',    telicity: 'telic',  domain: 'social' },
];

// Practice items (ASL, not scored; always shown with their correct meaning). Order is fixed.
export const PRACTICE = [
  { id: 'laugh',  file: 'practice/laugh.mp4',  telicity: 'atelic', meaning: 'laugh',  practice: true },
  { id: 'zip_up', file: 'practice/zip_up.mp4', telicity: 'telic',  meaning: 'zip up', practice: true },
];

// Item orders from "Counterbalancing Lists.docx", Experiment 1A.
export const LISTS = {
  list1: ['buy', 'decide', 'die', 'discuss', 'imagine', 'float', 'forget', 'leave', 'think', 'talk', 'run', 'sell'],
  list2: ['talk', 'run', 'sell', 'think', 'leave', 'forget', 'float', 'imagine', 'discuss', 'die', 'decide', 'buy'],
  list3: ['buy', 'decide', 'die', 'discuss', 'imagine', 'float', 'forget', 'leave', 'think', 'sell', 'run', 'talk'],
  list4: ['talk', 'run', 'sell', 'think', 'leave', 'forget', 'float', 'imagine', 'discuss', 'die', 'decide', 'buy'],
};

function shuffle(a) {
  const r = [...a];
  for (let i = r.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [r[i], r[j]] = [r[j], r[i]]; }
  return r;
}

/** Random draw of nTelic telic + nAtelic atelic items, presented in random order. */
export function drawBalanced(nTelic = 1, nAtelic = 1) {
  const telic = shuffle(ITEMS.filter((it) => it.telicity === 'telic')).slice(0, nTelic);
  const atelic = shuffle(ITEMS.filter((it) => it.telicity === 'atelic')).slice(0, nAtelic);
  return shuffle([...telic, ...atelic]);
}

export function orderItems(order, n, critical = { nTelic: 1, nAtelic: 1 }) {
  let items;
  if (order === 'balanced') items = drawBalanced(critical.nTelic, critical.nAtelic);
  else if (order === 'random') items = shuffle(ITEMS);
  else if (order.startsWith('ids:')) items = order.slice(4).split(',').map((id) => ITEMS.find((it) => it.id === id)).filter(Boolean);
  else if (LISTS[order]) items = LISTS[order].map((id) => ITEMS.find((it) => it.id === id));
  else items = [...ITEMS];
  return n ? items.slice(0, n) : items;
}

// ---- Meanings shown before each critical sign (data/sign_meanings.csv) ----
// Conditions (one per participant): 'correct' = the sign's own meaning,
// 'same' = a wrong meaning of the same telicity, 'opposite' = a wrong meaning of the opposite telicity.
// columns 'matched' uses same_telicity / opposite_telicity; 'random' uses random_same_telicity / random_opposite_telicity.
export const CONDITIONS = ['correct', 'same', 'opposite'];

export async function loadMeanings(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error('Cannot load ' + url);
  const lines = (await r.text()).replace(/\r/g, '').split('\n').map((l) => l.trim()).filter(Boolean);
  const head = lines[0].split(',').map((h) => h.trim());
  const table = {};
  for (const line of lines.slice(1)) {
    const cells = line.split(',').map((c) => c.trim());
    const row = Object.fromEntries(head.map((h, i) => [h, cells[i]]));
    if (row.sign) table[row.sign.toLowerCase()] = row;
  }
  return table;
}

export function pickCondition(forced) {
  if (CONDITIONS.includes(forced)) return forced;
  return CONDITIONS[Math.floor(Math.random() * CONDITIONS.length)];
}

/** The meaning to show for an item: { condition, shown, correct, column }. */
export function meaningFor(item, table, condition, columns = 'matched') {
  if (item.practice) return { condition: 'correct', shown: item.meaning, correct: item.meaning, column: 'practice' };
  const row = table[item.id];
  if (!row) throw new Error('No meanings for sign "' + item.id + '" in the meanings CSV');
  const col = condition === 'correct' ? 'sign'
    : (columns === 'random' ? 'random_' : '') + (condition === 'same' ? 'same_telicity' : 'opposite_telicity');
  return { condition, shown: row[col], correct: row.sign, column: col };
}
