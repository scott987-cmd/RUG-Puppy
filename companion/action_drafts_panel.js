/* Text preview only: this module never requests physical action execution. */
(function (global) {
  'use strict';
  function mountActionDraftsPanel(container, options) {
    if (!(container instanceof HTMLElement) || typeof options.fetch !== 'function' ||
        typeof options.getDeviceId !== 'function') throw new Error('动作草稿面板配置无效');
    const root = document.createElement('section');
    const title = document.createElement('h2'); title.textContent = '孩子的新把戏草稿';
    const note = document.createElement('p');
    note.textContent = '草稿不会执行，也不会进入已批准动作库。请查看步骤，保存编辑后再明确批准；批准只保存动作，不会让 Puppy 动起来。同名动作不会被覆盖。';
    const status = document.createElement('p'); status.setAttribute('role', 'status');
    const rows = document.createElement('div');
    const refreshButton = document.createElement('button'); refreshButton.type = 'button'; refreshButton.textContent = '刷新待审核草稿';
    let disposed = false, busy = false, sequence = 0, shownDevice = null;
    const actionLabels = {wave: '挥手', sit: '坐下', stretch: '伸懒腰', hug: '抱抱'};
    const states = {draft: '待家长审核', approved: '已批准并保存', rejected: '已拒绝', revoked: '已撤销对应动作',
      approving: '正在批准', revoking: '正在撤销', approval_unknown: '批准结果待核对', revoke_unknown: '撤销结果待核对'};
    function device() { const id = String(options.getDeviceId() || '').trim(); if (!id) throw new Error('请先选择 Puppy'); return id; }
    async function request(path, method = 'GET', body) {
      const init = {method};
      if (body !== undefined) { init.headers = {'Content-Type': 'application/json'}; init.body = JSON.stringify(body); }
      const response = await options.fetch('/api/action-drafts' + path, init), value = await response.json();
      if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : '草稿操作未完成');
      return value;
    }
    function preview(steps) {
      return steps.map((step, index) => (index + 1) + '. ' + (step.action ? actionLabels[step.action] : '等待') +
        (step.wait_ms ? (step.action ? '完成后等待 ' : ' ') + step.wait_ms + ' 毫秒' : '')).join('\n');
    }
    function render(items) {
      rows.replaceChildren();
      for (const draft of items) {
        const card = document.createElement('fieldset'), legend = document.createElement('legend');
        legend.textContent = draft.name + ' · ' + (states[draft.state] || draft.state) + ' · 版本 ' + draft.version;
        const source = document.createElement('p');
        source.textContent = '孩子描述：' + draft.source_text + '；设备：' + draft.device_id + '；会话：' + (draft.session_id || '未标记');
        const name = document.createElement('input'); name.type = 'text'; name.maxLength = 32; name.value = draft.name;
        const nameLabel = document.createElement('label'); nameLabel.append('动作名称 ', name);
        const editor = document.createElement('div'), rendered = document.createElement('pre');
        const steps = draft.steps.map(item => ({...item}));
        let dirty = false;
        const approve = document.createElement('button'); approve.type = 'button'; approve.textContent = '批准当前已保存草稿';
        function changed() { dirty = true; approve.disabled = true; rendered.textContent = preview(steps); }
        name.addEventListener('input', changed);
        function drawSteps() {
          editor.replaceChildren();
          steps.forEach((step, index) => {
            const line = document.createElement('div'), select = document.createElement('select'), wait = document.createElement('input');
            for (const [value, label] of [['', '仅等待'], ...Object.entries(actionLabels)]) {
              const option = document.createElement('option'); option.value = value; option.textContent = label; select.append(option);
            }
            select.value = step.action || ''; wait.type = 'number'; wait.min = 0; wait.max = 3000; wait.step = 1; wait.value = step.wait_ms || 0;
            select.addEventListener('change', () => { if (select.value) step.action = select.value; else delete step.action; changed(); });
            wait.addEventListener('input', () => { step.wait_ms = Number(wait.value); changed(); });
            const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '删除这一步';
            remove.addEventListener('click', () => { if (steps.length > 1) { steps.splice(index, 1); changed(); drawSteps(); } });
            line.append(String(index + 1) + '. ', select, ' 完成后等待（毫秒） ', wait, remove); editor.append(line);
          });
          rendered.textContent = preview(steps);
        }
        async function mutate(suffix, method, body) {
          if (busy || disposed) return;
          busy = true; refreshButton.disabled = true; card.disabled = true; ++sequence;
          try {
            if (shownDevice !== device()) throw new Error('设备已变化，请先刷新');
            await request('/' + encodeURIComponent(draft.draft_id) + suffix, method, body);
            busy = false;
            await refresh();
          } catch (error) { if (!disposed) status.textContent = error.message; }
          finally { busy = false; refreshButton.disabled = false; card.disabled = false; }
        }
        function button(label, suffix, method = 'POST') {
          const element = document.createElement('button'); element.type = 'button'; element.textContent = label;
          element.addEventListener('click', () => mutate(suffix, method, {version: draft.version}));
          return element;
        }
        card.append(legend, source);
        if (draft.state === 'draft') {
          drawSteps();
          const add = document.createElement('button'); add.type = 'button'; add.textContent = '添加一步';
          add.addEventListener('click', () => { if (steps.length < 6) { steps.push({action: 'wave'}); changed(); drawSteps(); } });
          const save = document.createElement('button'); save.type = 'button'; save.textContent = '保存编辑供审核';
          save.addEventListener('click', () => mutate('', 'PATCH', {version: draft.version, name: name.value, steps}));
          approve.addEventListener('click', () => { if (!dirty) mutate('/approve', 'POST', {version: draft.version}); });
          card.append(nameLabel, editor, add, rendered, save, approve, button('拒绝草稿', '/reject'));
        } else {
          rendered.textContent = preview(steps); card.append(rendered);
          if (draft.can_revoke === true && draft.state === 'approved') {
            const warning = document.createElement('p'); warning.textContent = '撤销只删除仍属于本次批准且内容一致的动作；修改或重新创建的同名动作会拒绝删除。';
            card.append(warning, button('撤销对应已保存动作', '/revoke'));
          } else if (['approved', 'approval_unknown', 'revoke_unknown'].includes(draft.state)) {
            const warning = document.createElement('p'); warning.textContent = '批准或撤销结果不确定，或缺少归属凭据。请到已保存动作管理核对；此处不会按同名同内容删除。';
            card.append(warning);
          }
        }
        if (draft.error) { const error = document.createElement('p'); error.textContent = draft.error; card.append(error); }
        const deletion = document.createElement('p'); deletion.textContent = '删除草稿只清理本条记录，不删除已保存动作。';
        card.append(deletion, button('删除草稿记录', '', 'DELETE')); rows.append(card);
      }
      if (!items.length) rows.textContent = '暂无草稿。孩子提出命名新把戏后，会在这里等待家长审核。';
    }
    async function refresh() {
      if (disposed || busy) return;
      const revision = ++sequence;
      try {
        const id = device(), result = await request('/devices/' + encodeURIComponent(id));
        if (disposed || revision !== sequence || id !== device()) return;
        shownDevice = id; render(result.drafts); status.textContent = '这里只展示文字步骤，不执行动作。';
      } catch (error) { if (!disposed) status.textContent = error.message; }
    }
    refreshButton.addEventListener('click', refresh); root.append(title, note, refreshButton, status, rows); container.append(root); refresh();
    return {refresh, destroy() { disposed = true; ++sequence; root.remove(); }};
  }
  global.mountActionDraftsPanel = mountActionDraftsPanel;
})(window);
