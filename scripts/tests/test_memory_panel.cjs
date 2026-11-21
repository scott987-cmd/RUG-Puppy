// Actual panel functions with fake DOM/API only. No browser, network or device.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const test = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../../companion/panel.html'), 'utf8');
for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
const helperStart = html.indexOf('function memoryRoute()');
const helpers = html.slice(helperStart, html.indexOf('</script>', helperStart));
const key = 'preference_statement:' + 'a'.repeat(64);
const statement = '  我不喜欢 <script>alert(1)</script>\n只有周末喜欢小船。  ';
const original = {
  memories: [{memory_key: key, value: statement}, {memory_key: 'preference:old', value: '旧喜好'}],
  english: [], story: null,
};
const refreshed = {memories: [], english: [{word: 'dog', presentations: 2}], story: null};
const refreshedSummary = '词汇接触记录（不代表掌握）：dog 2 次';

function element(tag) {
  return {
    tag, children: [], textContent: '', value: '',
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    set innerHTML(_) { throw new Error('Unsafe HTML write'); },
  };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function setup() {
  const elements = Object.fromEntries(['device', 'person', 'memories', 'learning', 'englishChecks', 'story']
    .map(id => [id, element(id)]));
  elements.device.value = 'device-one';
  elements.person.value = 'child-one';
  const calls = [];
  let handler = async () => original;
  const context = {
    document: {createElement: element}, $: id => elements[id], JSON, encodeURIComponent, Error,
    async api(route, method = 'GET') {
      calls.push({route, method});
      const response = await handler(route, method);
      return method === 'GET' ? {english_checks: [], english_words: [], english_review_plan: [], ...response} : response;
    },
  };
  vm.createContext(context);
  vm.runInContext(helpers, context);
  return {elements, calls, context, setHandler(next) { handler = next; }};
}

test('renders verbatim text and legacy labels without interpreting HTML', async () => {
  const {elements, context} = setup();
  await context.memories();
  assert.equal(elements.memories.children[0].children[0].textContent, '原话：' + statement + ' ');
  assert.equal(elements.memories.children[1].children[0].textContent, '旧记录：旧喜好 ');
  assert.equal(elements.memories.children[0].children[1].textContent, '删除整条');
});

test('deletes the exact displayed key in its original member/device route', async () => {
  const {elements, context, calls} = setup();
  await context.memories();
  await elements.memories.children[0].children[1].onclick();
  assert.deepEqual(calls.filter(call => call.method === 'DELETE'), [{
    route: '/api/memory/device-one/child-one/' + encodeURIComponent(key), method: 'DELETE',
  }]);
});

test('a button from another selected scope neither deletes nor displays its old error', async () => {
  const {elements, context, calls} = setup();
  await context.memories();
  const staleButton = elements.memories.children[0].children[1];
  elements.person.value = 'child-two';
  elements.learning.textContent = '当前成员的状态';
  await staleButton.onclick();
  assert.equal(calls.filter(call => call.method === 'DELETE').length, 0);
  assert.equal(elements.learning.textContent, '当前成员的状态');
});

test('a response from a previously selected scope is not rendered', async () => {
  const {elements, context, setHandler} = setup();
  const pending = deferred();
  setHandler(() => pending.promise);
  const existing = elements.memories.children;
  const request = context.memories();
  elements.device.value = 'device-two';
  pending.resolve(original);
  await request;
  assert.equal(elements.memories.children, existing);
  assert.equal(elements.learning.textContent, '');
});

test('two same-scope GET responses arriving in reverse order keep the newest data', async () => {
  const {elements, context, setHandler} = setup();
  const first = deferred(), second = deferred();
  const pending = [first, second];
  setHandler(() => pending.shift().promise);
  const older = context.memories();
  const newer = context.memories();
  second.resolve(refreshed);
  await newer;
  const latestRows = elements.memories.children;
  first.resolve(original);
  await older;
  assert.equal(elements.memories.children, latestRows);
  assert.equal(elements.memories.children.length, 0);
  assert.equal(elements.learning.textContent, refreshedSummary);
});

test('DELETE invalidates earlier GETs during deletion and after its fresh result', async () => {
  const {elements, context, setHandler, calls} = setup();
  await context.memories();
  const button = elements.memories.children[0].children[1];
  const originalRows = elements.memories.children;
  const earlyGet = deferred(), lateGet = deferred(), deletion = deferred();
  const pending = [earlyGet, lateGet];
  let deleted = false;
  setHandler((_route, method) => {
    if (method === 'DELETE') return deletion.promise;
    return deleted ? Promise.resolve(refreshed) : pending.shift().promise;
  });
  const earlyRequest = context.memories();
  const lateRequest = context.memories();
  const deleting = button.onclick();
  earlyGet.resolve(original);
  await earlyRequest;
  assert.equal(elements.memories.children, originalRows);
  deleted = true;
  deletion.resolve({deleted: key});
  await deleting;
  const latestRows = elements.memories.children;
  assert.equal(latestRows.length, 0);
  lateGet.resolve(original);
  await lateRequest;
  assert.equal(elements.memories.children, latestRows);
  assert.equal(elements.learning.textContent, refreshedSummary);
  assert.equal(calls.filter(call => call.method === 'DELETE').length, 1);
});

test('stale GET errors cannot replace current state, while the latest error is visible', async () => {
  const {elements, context, setHandler} = setup();
  const old = deferred();
  setHandler(() => old.promise);
  const older = context.memories();
  setHandler(async () => refreshed);
  await context.memories();
  old.reject(new Error('过时的读取失败'));
  await older;
  assert.equal(elements.learning.textContent, refreshedSummary);
  setHandler(async () => { throw new Error('当前读取失败'); });
  await context.memories();
  assert.equal(elements.learning.textContent, '当前读取失败');
});

test('late DELETE success or error does not refresh or overwrite another scope', async () => {
  for (const fail of [false, true]) {
    const {elements, context, setHandler, calls} = setup();
    await context.memories();
    const button = elements.memories.children[0].children[1];
    const deletion = deferred();
    setHandler((_route, method) => method === 'DELETE' ? deletion.promise : Promise.resolve(refreshed));
    const deleting = button.onclick();
    elements.person.value = 'child-two';
    await context.memories();
    const callCount = calls.length;
    if (fail) deletion.reject(new Error('旧成员的删除失败'));
    else deletion.resolve({deleted: key});
    await deleting;
    assert.equal(calls.length, callCount);
    assert.equal(elements.memories.children.length, 0);
    assert.equal(elements.learning.textContent, refreshedSummary);
  }
});

for (const kind of ['Person', 'Story']) {
  test(`forget${kind} invalidates old GETs and keeps cleared data after delayed responses`, async () => {
    const {elements, context, setHandler, calls} = setup();
    const before = {...original, story: {title: '旧故事', chapter: 2, summary: '旧情节。'}};
    const after = kind === 'Person' ? {memories: [], english: [], story: null}
      : {...original, story: null};
    setHandler(async () => before);
    await context.memories();
    assert.match(elements.story.textContent, /旧故事/);
    const initialRows = elements.memories.children;
    const earlyGet = deferred(), lateGet = deferred(), deletion = deferred();
    const pending = [earlyGet, lateGet];
    let deleted = false;
    setHandler((_route, method) => {
      if (method === 'DELETE') return deletion.promise;
      return deleted ? Promise.resolve(after) : pending.shift().promise;
    });
    const earlyRequest = context.memories();
    const lateRequest = context.memories();
    const deleting = context[`forget${kind}`]();
    earlyGet.resolve(before);
    await earlyRequest;
    assert.equal(elements.memories.children, initialRows);
    deleted = true;
    deletion.resolve({deleted: true});
    await deleting;
    const latestRows = elements.memories.children;
    const expectedCount = kind === 'Person' ? 0 : original.memories.length;
    assert.equal(latestRows.length, expectedCount);
    assert.equal(elements.story.textContent, '还没有保存故事进度');
    lateGet.resolve(before);
    await lateRequest;
    assert.equal(elements.memories.children, latestRows);
    assert.equal(elements.story.textContent, '还没有保存故事进度');
    assert.deepEqual(calls.filter(call => call.method === 'DELETE'), [{
      route: kind === 'Person' ? '/api/memory/device-one/child-one' : '/api/story/device-one/child-one',
      method: 'DELETE',
    }]);

    // A later deletion failure after selecting another member must also remain
    // silent and must not start a refresh in either the old or new scope.
    const failedDeletion = deferred();
    setHandler((_route, method) => method === 'DELETE' ? failedDeletion.promise : Promise.resolve(after));
    const failing = context[`forget${kind}`]();
    elements.person.value = 'child-two';
    await context.memories();
    const count = calls.length;
    failedDeletion.reject(new Error('旧成员的删除错误'));
    await failing;
    assert.equal(calls.length, count);
    assert.equal(elements.story.textContent, '还没有保存故事进度');
    assert.equal(elements.learning.textContent, '还没有词汇练习记录');
  });
}
