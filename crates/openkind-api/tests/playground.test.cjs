// Run with node --test crates/openkind-api/tests/playground.test.cjs.
// Exercise the embedded script offline; browser interaction is checked separately.
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');
const html = readFileSync(require('node:path').join(__dirname, '../assets/playground.html'), 'utf8');
const source = html.split('<script>')[1].split('</script>')[0].replace(/\nboot\(\);/, '');

function harness() {
  const nodes = new Map();
  const storage = new Map();
  function node() {
    return { value: '', textContent: '', innerHTML: '', children: [], dataset: {},
      classList: { add() {}, remove() {}, toggle() {} },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      setAttribute() {}, addEventListener() {}, remove() {} };
  }
  const context = vm.createContext({
    document: { getElementById(id) { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); }, createElement: node, querySelectorAll: () => [] },
    localStorage: { getItem: k => storage.get(k), setItem: (k, v) => storage.set(k, v) },
    setTimeout: () => 0, clearTimeout() {}, AbortSignal, performance,
  });
  vm.runInContext(source, context);
  return { context, run: code => vm.runInContext(code, context), nodes, storage };
}

const choice = { state: 'Keep this text', model: 'offline-model', questions: {
  route: { type: 'choice', instructions: 'Pick a route', criteria: { a: 'First', __none__: 'None apply' } },
} };

test('raw/form round trip preserves unknown aliases, whitespace, and special map keys', () => {
  const h = harness();
  const request = JSON.parse(JSON.stringify(choice));
  request.questions.route.instructions = '  Pick a route  ';
  request.questions.route.criteria.a = '  First  ';
  Object.defineProperty(request.questions, '__proto__', { value: { type: 'noul', instructions: 'Keep this question' }, enumerable: true });
  Object.defineProperty(request.questions.route.criteria, '__proto__', { value: 'Keep this option', enumerable: true });
  h.context.request = request;
  h.run('validateFormRequest(request); loadRequestIntoForm(request)');
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(buildRequest())')), request);
});

test('structured instructions and unsupported fields stay in raw mode intact', () => {
  const h = harness();
  const request = structuredClone(choice);
  request.questions.route.instructions = { structured: ['do', 'not', 'flatten'] };
  h.context.request = request;
  h.run('adoptRequest(request)');
  assert.equal(h.run('S.mode'), 'raw');
  assert.deepEqual(JSON.parse(h.run('JSON.stringify(buildRequestBody())')), request);
  assert.equal(h.run('applyRawToForm()'), false);
  assert.deepEqual(JSON.parse(h.run('$("raw-box").value')), request);
});

test('unfinished raw drafts survive storage and restore without validation loss', () => {
  const h = harness();
  h.run('S.mode = "raw"; $("raw-box").value = "{unfinished"; persistSession(); S.mode = "form"; $("raw-box").value = ""; restoreSession()');
  assert.equal(h.run('S.mode'), 'raw');
  assert.equal(h.run('$("raw-box").value'), '{unfinished');
});

test('invalid form edits are not replaced by an empty raw request', () => {
  const h = harness();
  h.run('S.selectedModel = "mock"; S.questions = []; $("state-box").value = "Draft"; $("raw-box").value = "previous raw"; switchMode("raw")');
  assert.equal(h.run('S.mode'), 'form');
  assert.equal(h.run('$("raw-box").value'), 'previous raw');
  assert.match(h.run('$("request-error").textContent'), /at least one question/);
});

test('a failed model refresh clears stale choices and disables evaluation', async () => {
  const h = harness();
  h.context.fetch = async () => ({ ok: false, status: 401 });
  h.run('S.models = [{ name: "mock" }]; S.selectedModel = "mock"; S.bench.chosen.add("mock")');
  await h.run('refreshModels()');
  assert.equal(h.run('S.models.length'), 0);
  assert.equal(h.run('S.bench.chosen.size'), 0);
  assert.equal(h.nodes.get('run-btn').disabled, true);
  assert.match(h.nodes.get('model-message').textContent, /API key required/);
});

test('successful timed requests define benchmark throughput despite failures', () => {
  const h = harness();
  h.run('S.bench.runs = { mock: { latencies: [100,100], done: 3, errors: 1, tokensOut: 4 } }; renderBenchSummary(["mock"], 3, 2)');
  assert.match(h.nodes.get('bench-table').innerHTML, /<td>10.0<\/td><td>20.0<\/td><td>20<\/td>/);
});

test('every starter Choice includes a non-empty semantic-none option', () => {
  const h = harness();
  assert.equal(h.run('PRESETS.every(p => Object.values(p.request.questions).every(q => q.type !== "choice" || q.criteria[NONE_KEY]?.trim()))'), true);
});
