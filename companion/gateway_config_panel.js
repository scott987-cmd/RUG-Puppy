/* Parent provisioning only; opening the panel never changes the Puppy. */
(function (global) {
  'use strict';
  global.mountGatewayConfigPanel = function (container, options) {
    const root = document.createElement('section');
    const title = document.createElement('h2'); title.textContent = 'Puppy 网关配置';
    const note = document.createElement('p');
    note.textContent = '保存家中 Mac 网关地址，下次启动 Puppy 时使用。保存本身不重启、不移动。请先确认目标服务可用；填错地址时，在 Puppy 空闲时连续三击机身键可恢复旧网关配置，再手动重启。';
    const urlLabel = document.createElement('label'), url = document.createElement('input');
    urlLabel.textContent = '语音网关地址（须含端口和 /v1/audio）';
    url.placeholder = 'ws://网关当前局域网地址:8000/v1/audio'; urlLabel.append(url);
    const tokenLabel = document.createElement('label'), token = document.createElement('input');
    tokenLabel.textContent = '新配对令牌（留空使用本 Mac 的既有令牌）';
    token.type = 'password'; token.autocomplete = 'off'; token.maxLength = 256; tokenLabel.append(token);
    const save = document.createElement('button'); save.textContent = '保存配置（下次启动生效）';
    const retry = document.createElement('button'); retry.textContent = '重试相同版本';
    const refreshButton = document.createElement('button'); refreshButton.textContent = '读取网关配置状态';
    const status = document.createElement('p'); status.setAttribute('role', 'status');
    const transport = document.createElement('p');
    transport.textContent = 'ws 为家庭局域网明文连接。wss 需要目标证书受 Puppy 信任；仅改成 wss 并不能完成证书配置。';
    let current = null, shownDevice = null, busy = false, disposed = false, serial = 0;
    function device() { const id = String(options.getDeviceId() || '').trim(); if (!id) throw Error('请先选择设备'); return id; }
    async function request(id, suffix = '', body) {
      const response = await options.fetch('/api/gateway/devices/' + encodeURIComponent(id) + suffix,
        body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const result = await response.json();
      if (!response.ok) throw Error(typeof result.detail === 'string' ? result.detail : '网关配置请求失败');
      return result;
    }
    function buttons() {
      const ready = !busy && current && current.device.online && current.device.supported;
      save.disabled = !ready;
      retry.disabled = !ready || !current.latest || current.device.version > current.latest.version;
      refreshButton.disabled = busy;
      url.disabled = token.disabled = busy;
    }
    function render(value) {
      current = value;
      const names = {prepared: '已记录，尚未确认保存', sending: '正在保存', saved: '已确认保存，下次启动生效', unknown: '结果未确认，可重试同一版本', conflict: '设备已有更高版本'};
      status.textContent = !value.device.online ? '设备离线' : !value.device.supported ? '需要升级支持配置入口的固件' : '设备配置版本 ' + value.device.version;
      if (value.latest) status.textContent += '；本次版本 ' + value.latest.version + '：' + names[value.latest.state];
      if (value.device.restart_required) status.textContent += '；设备提示需要重新启动才能应用';
      const savedIsCurrent = value.latest && value.device.version <= value.latest.version;
      if (!url.value || (!savedIsCurrent && value.latest && url.value === value.latest.url)) {
        url.value = (savedIsCurrent && value.latest.url) || value.device.url || '';
      }
      buttons();
    }
    async function refresh() {
      if (busy || disposed) return;
      const revision = ++serial;
      try {
        const id = device(), value = await request(id);
        if (disposed || revision !== serial || id !== device()) return;
        if (shownDevice !== id) { url.value = ''; token.value = ''; }
        shownDevice = id; render(value);
      } catch (error) { if (!disposed) { current = null; buttons(); status.textContent = error.message; } }
    }
    async function submit(isRetry) {
      if (busy || disposed) return;
      busy = true; ++serial; buttons();
      try {
        const id = device();
        if (id !== shownDevice) throw Error('设备已变化，请先刷新');
        let body;
        if (isRetry) body = {version: current.latest.version};
        else { body = {url: url.value.trim()}; if (token.value) body.token = token.value; }
        token.value = ''; // Never persist or display a secret after submission.
        const value = await request(id, isRetry ? '/retry' : '/configure', body);
        if (!disposed && id === device()) render(value);
      } catch (error) { if (!disposed) status.textContent = error.message; }
      finally { busy = false; if (!disposed) buttons(); }
    }
    save.addEventListener('click', () => submit(false)); retry.addEventListener('click', () => submit(true));
    refreshButton.addEventListener('click', refresh);
    root.append(title, note, urlLabel, tokenLabel, save, retry, refreshButton, status, transport);
    container.append(root); buttons(); refresh();
    return {refresh, destroy() { disposed = true; ++serial; token.value = ''; root.remove(); }};
  };
})(window);
