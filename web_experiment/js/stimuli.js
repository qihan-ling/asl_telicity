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

export function orderItems(order, n) {
  let items;
  if (order === 'random') items = shuffle(ITEMS);
  else if (order.startsWith('ids:')) items = order.slice(4).split(',').map((id) => ITEMS.find((it) => it.id === id)).filter(Boolean);
  else if (LISTS[order]) items = LISTS[order].map((id) => ITEMS.find((it) => it.id === id));
  else items = [...ITEMS];
  return n ? items.slice(0, n) : items;
}
