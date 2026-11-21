/* Explicit parent synchronization only. Mount/refresh never sends a library. */
(function (global) {
  'use strict';
  function mountOfflineLibraryPanel(container, options) {
    if (!(container instanceof HTMLElement) || typeof options.fetch !== 'function' ||
        typeof options.getDeviceId !== 'function') throw new Error('离线动作库面板配置无效');
    const root = document.createElement('section');
    const heading = document.createElement('h2'); heading.textContent = '离线小动作库';
    const explanation = document.createElement('p');
    explanation.textContent = '选择最多 3 个家长已保存动作，每个最多 6 步。同步只保存到 Puppy；断网后通过机身按钮选择，不会因同步而执行。修改面板动作不会改变设备已保存的快照。';
    const status = document.createElement('p'); status.setAttribute('role', 'status');
    const choices = document.createElement('div'), history = document.createElement('div');
    const sync = document.createElement('button'); sync.type = 'button'; sync.textContent = '同步本次选择（新版本）';
    const retry = document.createElement('button'); retry.type = 'button'; retry.textContent = '重试显示的版本和快照';
    const clear = document.createElement('button'); clear.type = 'button'; clear.textContent = '清空设备离线库（新版本）';
    const refreshButton = document.createElement('button'); refreshButton.type = 'button'; refreshButton.textContent = '刷新状态与动作列表';
    let disposed = false, busy = false, sequence = 0, shownDevice = null, current = null, selected = [];
    const path = id => '/api/offline-library/devices/' + encodeURIComponent(id);
    function device() { const id = String(options.getDeviceId() || '').trim(); if (!id) throw new Error('请先选择 Puppy'); return id; }
    async function request(id, suffix = '', body) {
      const init = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
      const response = await options.fetch(path(id) + suffix, init);
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : '离线库请求未完成');
      return result;
    }
    const states = {prepared: '已记录，未确认发送', sending: '等待设备保存回执', saved: '设备确认保存，未执行', unknown: '保存结果未确认', conflict: '设备已有更高版本'};
    function buttons() {
      const enabled = !busy && current && current.device.online && current.device.supported;
      sync.disabled = !enabled || selected.length === 0;
      clear.disabled = !enabled;
      retry.disabled = !enabled || !current.latest || current.device.version > current.latest.version;
      refreshButton.disabled = busy;
      for (const input of choices.querySelectorAll('input')) input.disabled = busy || input.dataset.eligible !== 'true';
    }
    function render(value) {
      current = value;
      status.textContent = !value.device.online ? '设备不在线' : !value.device.supported ? '请升级支持离线动作库的固件' :
        '设备报告版本 ' + value.device.version + ' · 已保存 ' + value.device.action_count + ' 个动作';
      if (value.latest) {
        status.textContent += '；本地版本 ' + value.latest.version + '：' + (states[value.latest.state] || value.latest.state);
        if (value.latest.state === 'unknown') status.textContent += '。重试保留原版本原内容；新同步/清空会建立更高版本，上版结果仍保留在记录中。';
      }
      choices.replaceChildren(); history.replaceChildren();
      const eligible = new Set(value.catalog.filter(item => item.eligible).map(item => item.name));
      selected = selected.filter(name => eligible.has(name));
      for (const item of value.catalog) {
        const row = document.createElement('label'), checkbox = document.createElement('input');
        checkbox.dataset.eligible = String(item.eligible);
        checkbox.type = 'checkbox'; checkbox.style.width = 'auto'; checkbox.disabled = !item.eligible || busy;
        checkbox.checked = selected.includes(item.name);
        checkbox.addEventListener('change', () => {
          if (checkbox.checked) {
            if (selected.length >= 3) { checkbox.checked = false; status.textContent = '最多选择 3 个动作'; return; }
            selected.push(item.name);
          } else selected = selected.filter(name => name !== item.name);
          buttons();
        });
        row.append(checkbox, item.name + (item.eligible ? ' · ' + item.step_count + ' 步' : ' · ' + item.reason));
        choices.append(row);
      }
      if (!value.catalog.length) choices.textContent = '请先在自定义动作区保存动作';
      for (const entry of value.history) {
        const row = document.createElement('details'), summary = document.createElement('summary'), snapshot = document.createElement('pre');
        summary.textContent = '版本 ' + entry.version + ' · ' + (states[entry.state] || entry.state) + ' · 已尝试 ' + entry.attempts + ' 次';
        snapshot.textContent = JSON.stringify(entry.actions, null, 2);
        row.append(summary, snapshot); history.append(row);
      }
      buttons();
    }
    async function refresh() {
      if (busy || disposed) return;
      const revision = ++sequence;
      try {
        const id = device(), value = await request(id);
        if (disposed || revision !== sequence || id !== device()) return;
        if (shownDevice !== id) selected = [];
        shownDevice = id; render(value);
      } catch (error) { if (!disposed) { current = null; buttons(); status.textContent = error.message; } }
    }
    async function send(kind) {
      if (busy || disposed) return;
      busy = true; ++sequence; buttons();
      try {
        const id = device();
        if (shownDevice !== id || !current) throw new Error('设备已变化，请先刷新');
        const suffix = kind === 'retry' ? '/retry' : '/sync';
        const body = kind === 'retry' ? {version: current.latest.version} : {names: kind === 'clear' ? [] : [...selected]};
        status.textContent = '正在等待设备保存回执，最长约 35 秒。不会执行动作。';
        const value = await request(id, suffix, body);
        if (disposed || id !== device()) return;
        if (kind === 'clear') selected = [];
        render(value);
      } catch (error) { if (!disposed) status.textContent = error.message; }
      finally { busy = false; if (!disposed) buttons(); }
    }
    sync.addEventListener('click', () => send('sync')); retry.addEventListener('click', () => send('retry'));
    clear.addEventListener('click', () => send('clear')); refreshButton.addEventListener('click', refresh);
    root.append(heading, explanation, choices, sync, retry, clear, refreshButton, status, history); container.append(root);
    buttons(); refresh();
    return {refresh, destroy() { disposed = true; ++sequence; root.remove(); }};
  }
  global.mountOfflineLibraryPanel = mountOfflineLibraryPanel;
})(window);
