/* Mount with authenticated fetch and getDeviceId. No camera, microphone,
 * movement, playback, cloud deletion, or immediate-capture controls. */
(function (global) {
  'use strict';
  function mountPatrolPanel(container, options) {
    if (!(container instanceof HTMLElement) || typeof options.fetch !== 'function' ||
        typeof options.getDeviceId !== 'function') throw new Error('巡看面板配置无效');
    const root = document.createElement('section');
    const title = document.createElement('h2'); title.textContent = '原地巡看';
    const explanation = document.createElement('p');
    explanation.textContent = '默认关闭。启用后按时间窗在原地拍照，不移动、不播声。离线或错过的巡看不会补跑；保存设置不会立即拍照。';
    const status = document.createElement('p'); status.setAttribute('role', 'status');
    const history = document.createElement('div');
    const form = document.createElement('div');
    let disposed = false, activeDevice = null, sequence = 0;
    const urls = new Set();
    function input(label, type, value) {
      const row = document.createElement('label'); const element = document.createElement('input');
      element.type = type;
      if (type === 'checkbox') { element.checked = value; element.style.width = 'auto'; }
      else element.value = value;
      row.append(label, element); form.append(row); return element;
    }
    const enabled = input('我授权按下列计划自动拍照 ', 'checkbox', false);
    const interval = input('间隔（分钟，15–1440） ', 'number', 60); interval.min = 15; interval.max = 1440;
    const start = input('开始时间 ', 'time', '09:00'); const end = input('结束时间 ', 'time', '21:00');
    const zone = input('时区 ', 'text', 'Asia/Shanghai'); zone.maxLength = 64;
    const describe = input('附上本地模型的照片描述 ', 'checkbox', false);
    const retention = input('本地照片保留天数（1–30） ', 'number', 7); retention.min = 1; retention.max = 30;
    const scope = document.createElement('p');
    scope.textContent = '支持跨午夜时间窗；开始与结束相同表示全天。只清理本功能登记的本地照片。iCloud 副本不会被删除。';
    function device() {
      const value = String(options.getDeviceId() || '').trim();
      if (!value) throw new Error('请先选择 Puppy 设备');
      return value;
    }
    const path = id => '/api/patrol/devices/' + encodeURIComponent(id);
    async function request(url, method = 'GET', body) {
      const init = {method};
      if (body !== undefined) { init.headers = {'Content-Type': 'application/json'}; init.body = JSON.stringify(body); }
      const response = await options.fetch(url, init);
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : '巡看请求未完成');
      return result;
    }
    function releaseImages() { for (const url of urls) URL.revokeObjectURL(url); urls.clear(); }
    function button(text, action, parent = form) {
      const element = document.createElement('button'); element.type = 'button'; element.textContent = text;
      element.addEventListener('click', async () => {
        element.disabled = true;
        try { await action(); } catch (error) { if (!disposed) status.textContent = error.message; }
        finally { element.disabled = false; }
      }); parent.append(element); return element;
    }
    async function refresh() {
      const id = device(), version = ++sequence;
      const [config, records] = await Promise.all([request(path(id)), request(path(id) + '/runs')]);
      if (disposed || version !== sequence || device() !== id) return;
      activeDevice = id;
      enabled.checked = config.enabled; interval.value = config.interval_minutes;
      start.value = config.window_start; end.value = config.window_end; zone.value = config.timezone;
      describe.checked = config.describe; describe.disabled = !config.description_available;
      retention.value = config.retention_days;
      status.textContent = config.enabled ? '下次巡看：' + new Date(config.next_due * 1000).toLocaleString() : '巡看已关闭';
      releaseImages(); history.replaceChildren();
      const states = {running: '正在巡看', completed: '已完成', failed: '本次失败，不补跑', cancelled: '已取消',
        interrupted: '运行中断，不补跑', missed: '错过计划，不补跑', outside_window: '不在时间窗内', skipped_busy: '已有巡看，跳过'};
      for (const run of records.runs) {
        const row = document.createElement('div'); const text = document.createElement('p');
        text.textContent = new Date(run.scheduled_at * 1000).toLocaleString() + ' · ' + (states[run.state] || run.state) +
          (run.error ? (run.state === 'completed' ? '（照片已保存，描述未完成）' : '（请检查设备或服务连接）') : ''); row.append(text);
        if (run.description) { const description = document.createElement('p'); description.textContent = run.description; row.append(description); }
        if (run.photo_id) button('查看照片', async () => {
          const response = await options.fetch('/api/patrol/photos/' + encodeURIComponent(run.photo_id));
          if (!response.ok) throw new Error('照片已清理或暂不可用');
          const blob = await response.blob();
          if (disposed || version !== sequence || device() !== id) return;
          const url = URL.createObjectURL(blob); urls.add(url);
          const image = document.createElement('img'); image.src = url; image.alt = 'Puppy 原地巡看照片';
          image.style.maxWidth = '100%'; row.append(image);
        }, row);
        history.append(row);
      }
      if (!records.runs.length) history.textContent = '尚无巡看记录';
    }
    button('保存计划', async () => {
      const id = device();
      if (activeDevice !== id) throw new Error('设备已变化，请先刷新计划');
      await request(path(id), 'PUT', {enabled: enabled.checked, interval_minutes: Number(interval.value),
        window_start: start.value, window_end: end.value, timezone: zone.value,
        describe: describe.checked, retention_days: Number(retention.value)});
      await refresh();
    });
    button('刷新计划和记录', refresh);
    button('清理到期的本地巡看照片', async () => {
      const result = await request('/api/patrol/cleanup', 'POST');
      await refresh();
      status.textContent = '已清理 ' + result.deleted + ' 张；路径变化而保留 ' + result.blocked + ' 张。iCloud 未变更。';
    });
    root.append(title, explanation, form, scope, status, history); container.append(root);
    refresh().catch(error => { if (!disposed) status.textContent = error.message; });
    return {refresh, destroy() { disposed = true; ++sequence; releaseImages(); root.remove(); }};
  }
  global.mountPatrolPanel = mountPatrolPanel;
})(window);
