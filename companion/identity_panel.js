/* Mount with {fetch: authenticatedFetch, getDeviceId: () => currentDeviceId}.
 * authenticatedFetch(url, init) returns a standard Response and adds parent auth.
 * This component neither reads nor saves any token; it never opens a Mac camera/mic.
 */
(function (global) {
  'use strict';
  function mountIdentityPanel(container, options) {
    if (!(container instanceof HTMLElement) || typeof options.fetch !== 'function' ||
        typeof options.getDeviceId !== 'function') throw new Error('身份面板配置无效');
    const root = document.createElement('section');
    const heading = document.createElement('h2'); heading.textContent = '录入声纹与人脸 · 认识家人';
    const explanation = document.createElement('p');
    explanation.textContent = '机身语音统一按访客处理。家长可以在这里查看和移除已登记的样本。';
    const select = document.createElement('select'); select.setAttribute('aria-label', '已登记成员');
    const person = document.createElement('input'); person.placeholder = '成员 ID，例如 child-1';
    person.maxLength = 64; person.autocomplete = 'off';
    const name = document.createElement('input'); name.placeholder = '成员名字或称呼'; name.maxLength = 32;
    const consent = document.createElement('input'); consent.type = 'checkbox'; consent.style.width = 'auto';
    const consentLabel = document.createElement('label');
    consentLabel.append(consent, ' 我确认照片或声音属于当前成员，同意在家中保存身份特征');
    const status = document.createElement('p'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    const ticketStatus = document.createElement('p'); ticketStatus.setAttribute('aria-live', 'polite');
    const samples = document.createElement('div');
    const fileFace = document.createElement('input'); fileFace.type = 'file'; fileFace.accept = 'image/jpeg';
    fileFace.setAttribute('aria-label', '上传 JPEG 人脸照片');
    const fileVoice = document.createElement('input'); fileVoice.type = 'file'; fileVoice.accept = '.wav,audio/wav';
    fileVoice.setAttribute('aria-label', '上传 WAV 声音样本');
    let disposed = false, timer = null, activeTicket = null, members = [];

    function id() {
      const value = person.value.trim();
      if (!/^[\p{L}\p{N}_-]{1,64}$/u.test(value)) throw new Error('请填写有效成员 ID');
      return value;
    }
    function device() {
      const value = String(options.getDeviceId() || '').trim();
      if (!value) throw new Error('请先选择 Puppy 设备');
      return value;
    }
    function consentGiven() { if (!consent.checked) throw new Error('请先确认当前成员并勾选同意保存身份特征'); }
    const memberPath = value => '/api/identity/members/' + encodeURIComponent(value);
    async function request(url, {method = 'GET', json, form} = {}) {
      const init = {method};
      if (json !== undefined) { init.headers = {'Content-Type': 'application/json'}; init.body = JSON.stringify(json); }
      if (form) init.body = form; // Let the browser supply the multipart boundary.
      const response = await options.fetch(url, init);
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求未完成，请检查输入或连接');
      return data;
    }
    function button(text, action, target = root) {
      const node = document.createElement('button'); node.type = 'button'; node.textContent = text;
      node.addEventListener('click', async () => {
        node.disabled = true; status.textContent = '正在处理…';
        try { await action(); }
        catch (error) { status.textContent = error.message; }
        finally { node.disabled = false; }
      });
      target.append(node); return node;
    }
    function label(text, input, target = root) { const node = document.createElement('label'); node.append(text, input); target.append(node); }
    function group(title) {
      const node = document.createElement('div'); node.className = 'identity-group';
      const heading = document.createElement('h3'); heading.textContent = title; node.append(heading);
      root.append(node); return node;
    }
    async function refreshSamples() {
      samples.replaceChildren();
      if (!person.value.trim()) return;
      const owner = id();
      const result = await request(memberPath(owner) + '/samples');
      if (disposed || person.value.trim() !== owner) return;
      for (const sample of result) {
        const row = document.createElement('p');
        const title = document.createElement('span');
        title.textContent = (sample.modality === 'face' ? '人脸' : '声音') + ' · ' + new Date(sample.created_at * 1000).toLocaleString() + ' ';
        row.append(title);
        button('删除这份样本', async () => {
          await request(memberPath(owner) + '/samples/' + encodeURIComponent(sample.sample_id), {method: 'DELETE'});
          await refresh(); status.textContent = '这份身份样本已删除';
        }, row);
        samples.append(row);
      }
      if (!result.length) samples.textContent = '尚未登记样本';
    }
    async function refresh() {
      const result = await request('/api/identity/members'); members = result.members;
      select.replaceChildren();
      const blank = document.createElement('option'); blank.value = ''; blank.textContent = '选择已有成员，或填写新成员'; select.append(blank);
      for (const member of members) {
        const option = document.createElement('option'); option.value = member.person_id;
        option.textContent = member.display_name + '（' + member.person_id + '）人脸 ' + member.counts.face + ' / 声音 ' + member.counts.voice;
        select.append(option);
      }
      select.value = members.some(member => member.person_id === person.value.trim()) ? person.value.trim() : '';
      if (select.value) await refreshSamples(); else samples.replaceChildren();
    }
    async function pollTicket() {
      if (!activeTicket || disposed) return;
      const polledTicket = activeTicket;
      try {
        const ticket = await request('/api/identity/voice-tickets/' + encodeURIComponent(polledTicket));
        if (disposed || activeTicket !== polledTicket) return;
        const labels = {waiting: '等待声纹样本', processing: '正在检查声纹样本', completed: '声纹样本已保存；请换一句话再录一段',
          expired: '注册窗口已过期，请重新开启', cancelled: '注册已取消', failed: '这份声音未通过检查'};
        ticketStatus.textContent = ticket.person_id + '：' + (labels[ticket.state] || ticket.state) +
          (ticket.state === 'waiting' ? '，剩余 ' + ticket.remaining_seconds + ' 秒' : '') + (ticket.error ? '；' + ticket.error : '');
        if (ticket.state === 'waiting' || ticket.state === 'processing') timer = setTimeout(pollTicket, 2000);
        else { activeTicket = null; await refresh(); }
      } catch (error) {
        if (!disposed && activeTicket === polledTicket) { ticketStatus.textContent = error.message; activeTicket = null; }
      }
    }
    async function upload(input, modality) {
      consentGiven(); const file = input.files[0];
      if (!file) throw new Error('请先选择文件');
      const limit = modality === 'face' ? 512000 : 644096;
      if (file.size > limit) throw new Error(modality === 'face' ? 'JPEG 请保持在 512 KB 以内' : 'WAV 超过大小限制');
      const form = new FormData(); form.append('file', file);
      await request(memberPath(id()) + '/' + modality, {method: 'POST', form});
      input.value = ''; await refresh(); status.textContent = '样本已保存';
    }
    root.append(heading, explanation, status);
    const memberGroup = group('1 · 选择或新建成员');
    memberGroup.append(select);
    label('成员 ID ', person, memberGroup); label('名字或称呼 ', name, memberGroup);
    button('保存成员', async () => {
      const displayName = name.value.trim(); if (!displayName) throw new Error('请填写名字或称呼');
      await request(memberPath(id()), {method: 'PUT', json: {display_name: displayName}});
      await refresh(); status.textContent = '成员已保存，可以登记样本';
    }, memberGroup);
    button('刷新成员与样本', async () => { await refresh(); status.textContent = '已刷新'; }, memberGroup);
    const voiceGroup = group('2 · 录入声纹');
    voiceGroup.classList.add('voice-group');
    voiceGroup.append(consentLabel);
    const voiceGuidance = document.createElement('p'); voiceGuidance.className = 'hint';
    voiceGuidance.textContent = '开启 60 秒录入窗口后，让当前成员对 Puppy 说唤醒词，屏幕显示聆听时说一段约 3–12 秒的话。换一句再录第二段；不需要按键。';
    voiceGroup.append(voiceGuidance);
    button('录入声纹 · 开启 60 秒窗口', async () => {
      consentGiven();
      const ticket = await request('/api/identity/voice-tickets', {method: 'POST', json: {device_id: device(), person_id: id()}});
      if (timer) clearTimeout(timer); activeTicket = ticket.ticket_id;
      status.textContent = '请现在对 Puppy 说唤醒词，等屏幕显示聆听后，由当前成员单独说话约 3–12 秒；不用按键'; await pollTicket();
    }, voiceGroup);
    button('取消声纹录入', async () => {
      if (!activeTicket) throw new Error('没有进行中的声纹录入');
      await request('/api/identity/voice-tickets/' + encodeURIComponent(activeTicket), {method: 'DELETE'});
      if (timer) clearTimeout(timer); activeTicket = null; ticketStatus.textContent = '声纹录入已取消'; status.textContent = '';
    }, voiceGroup);
    voiceGroup.append(ticketStatus);
    label('也可以上传 WAV（16 kHz / 单声道 / PCM16 / 3–20 秒） ', fileVoice, voiceGroup);
    button('登记所选声纹样本', () => upload(fileVoice, 'voice'), voiceGroup);
    const faceGroup = group('3 · 登记人脸');
    const faceGuidance = document.createElement('p'); faceGuidance.className = 'hint';
    faceGuidance.textContent = '镜头前只留当前成员，正脸、光线充足，换角度登记两张。拍照与上传都需要上方确认。';
    faceGroup.append(faceGuidance);
    button('用 Puppy 拍一份人脸样本', async () => {
      consentGiven(); await request(memberPath(id()) + '/capture-face', {method: 'POST', json: {device_id: device()}});
      await refresh(); status.textContent = '人脸样本已保存，请换个轻微角度再登记一份';
    }, faceGroup);
    label('也可以上传 JPEG（≤512 KB） ', fileFace, faceGroup);
    button('登记所选照片', () => upload(fileFace, 'face'), faceGroup);
    const sampleGroup = group('已登记的身份样本');
    sampleGroup.append(samples);
    button('移除此成员的全部身份样本', async () => {
      await request(memberPath(id()), {method: 'DELETE'});
      person.value = ''; name.value = ''; consent.checked = false; await refresh(); status.textContent = '该成员的身份注册已移除';
    }, sampleGroup);
    container.append(root);
    select.addEventListener('change', async () => {
      const member = members.find(item => item.person_id === select.value);
      person.value = member ? member.person_id : ''; name.value = member ? member.display_name : ''; consent.checked = false;
      try { await refreshSamples(); } catch (error) { status.textContent = error.message; }
    });
    person.addEventListener('input', () => { consent.checked = false; samples.replaceChildren(); });
    refresh().catch(error => { if (!disposed) status.textContent = error.message; });
    return {refresh, destroy() { disposed = true; if (timer) clearTimeout(timer); root.remove(); }};
  }
  global.PuppyIdentityPanel = {mount: mountIdentityPanel};
})(window);
