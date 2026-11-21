/* Parent opt-in only. No enabling from refresh, polling, mount or reload. */
(function (global) {
  'use strict';
  function mountGesturePanel(container, options) {
    if (!(container instanceof HTMLElement) || typeof options.fetch !== 'function' ||
        typeof options.getDeviceId !== 'function') throw new Error('手势面板配置无效');
    const root = document.createElement('section');
    const title = document.createElement('h2'); title.textContent = '短时手势指令';
    const description = document.createElement('p');
    description.textContent = '家长每次开启 60 秒，最多 3 次挥手或坐下。张开手掌＝停止，竖大拇指＝挥手，握拳＝坐下。单手保持清晰，再移开手后才能下一次。';
    const note = document.createElement('p');
    note.textContent = '开启时 Puppy 会连续拍照，在家中 Mac 本地识别，照片不保存。看不清、多只手或断联不会继续动作。尚未验证儿童识别准确率；紧急停止仍请使用家长面板停止按钮。';
    const label = document.createElement('label'), consent = document.createElement('input');
    consent.type = 'checkbox'; consent.style.width = 'auto';
    label.append(consent, '我在旁看护，同意本次短时拍照和动作');
    const status = document.createElement('p'); status.setAttribute('role', 'status');
    const enable = document.createElement('button'); enable.type = 'button'; enable.textContent = '开启 60 秒';
    const disable = document.createElement('button'); disable.type = 'button'; disable.textContent = '关闭手势指令';
    let disposed = false, refreshing = false, mutation = false, sequence = 0, shownDevice = null, last = null;
    function device() { const id = String(options.getDeviceId() || '').trim(); if (!id) throw new Error('请先选择 Puppy'); return id; }
    const path = id => '/api/gestures/devices/' + encodeURIComponent(id);
    async function request(id, suffix = '', body) {
      const response = await options.fetch(path(id) + suffix, body === undefined ? {} : {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : '手势请求失败');
      return result;
    }
    function render(value) {
      last = value;
      enable.disabled = mutation || !consent.checked || !value.available || value.enabled || value.busy;
      disable.disabled = mutation || (!value.enabled && !value.busy);
      const reasons = {disabled: '已关闭', parent_disabled: '家长已关闭', expired: '本次 60 秒已结束',
        disconnected: '设备断联，已关闭', capture_failed: '拍照未完成，已关闭', recognition_failed: '识别未完成，已关闭',
        action_failed: '动作未完成，已关闭', action_unknown: '动作结果不确定，已关闭', service_closed: '服务已关闭'};
      status.textContent = value.enabled ? '已开启 · 剩余 ' + value.remaining_seconds + ' 秒 · 动作 ' + value.movement_count + '/3' :
        (reasons[value.reason] || '已关闭或正在清理');
      if (!value.available) status.textContent += ' · 本地手势模型暂不可用';
      if (value.last_result && value.last_result.state === 'cancelled') status.textContent += ' · 已请求取消，物理停止状态未确认';
    }
    async function refresh() {
      if (disposed || refreshing || mutation) return;
      refreshing = true;
      const version = ++sequence;
      try {
        const id = device();
        if (shownDevice !== id) { consent.checked = false; enable.disabled = true; disable.disabled = true; }
        const value = await request(id);
        if (disposed || version !== sequence || id !== device()) return;
        shownDevice = id; render(value);
      } catch (error) { if (!disposed) { enable.disabled = true; status.textContent = error.message; } }
      finally { refreshing = false; }
    }
    async function change(open) {
      if (mutation || disposed) return;
      mutation = true; ++sequence;
      let failure = null;
      enable.disabled = disable.disabled = true;
      try {
        const id = device();
        if (id !== shownDevice) throw new Error('设备已变化，请稍候刷新');
        if (open && !consent.checked) throw new Error('请家长先勾选本次同意');
        const value = await request(id, open ? '/enable' : '/disable', open ? {seconds: 60, consent: true} : {});
        consent.checked = false;
        if (!disposed && id === device()) render(value);
      } catch (error) { failure = error.message; }
      finally {
        mutation = false;
        if (!disposed) { if (last) render(last); await refresh(); if (failure) status.textContent = failure; }
      }
    }
    consent.addEventListener('change', () => { if (last) render(last); });
    enable.addEventListener('click', () => change(true));
    disable.addEventListener('click', () => change(false));
    enable.disabled = disable.disabled = true;
    root.append(title, description, note, label, enable, disable, status); container.append(root);
    const timer = setInterval(refresh, 1000); refresh();
    return {refresh, destroy() { disposed = true; ++sequence; clearInterval(timer); root.remove(); }};
  }
  global.mountGesturePanel = mountGesturePanel;
})(window);
