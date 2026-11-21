/* Mount with {fetch: authenticatedFetch, getDeviceId: () => id,
 *             getPersonId?: () => confirmedPersonIdOrNull}.
 * fetch returns a Response with parent authorization. No tokens are stored here.
 */
(function (global) {
  'use strict';
  function mountLearningPanel(container, options) {
    if (!(container instanceof HTMLElement) || !options || typeof options.fetch !== 'function' ||
        typeof options.getDeviceId !== 'function') throw new Error('英语练习面板配置无效');
    const root = document.createElement('section');
    const title = document.createElement('h2'); title.textContent = '英语练习安排';
    const description = document.createElement('p');
    description.textContent = '按成员设置每日活动会话时长；留空成员 ID 时，设置设备默认并查看未知身份的用时。';
    const person = document.createElement('input'); person.placeholder = '成员 ID（留空为设备默认）'; person.maxLength = 128;
    let externalPerson = typeof options.getPersonId === 'function' ? String(options.getPersonId() || '') : '';
    person.value = externalPerson;
    const minutes = document.createElement('input'); minutes.type = 'number'; minutes.min = '1'; minutes.max = '20'; minutes.value = '5';
    const theme = document.createElement('select');
    for (const [value, text] of [['animals', '动物'], ['colors', '颜色'], ['greetings', '打招呼'], ['objects', '生活物品'], ['body_parts', '身体部位']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = text; theme.append(option);
    }
    const saveProgress = document.createElement('input'); saveProgress.type = 'checkbox'; saveProgress.style.width = 'auto';
    const photo = document.createElement('input'); photo.type = 'checkbox'; photo.style.width = 'auto';
    const timerView = document.createElement('p'); timerView.setAttribute('aria-live', 'off');
    const stateView = document.createElement('p'); stateView.setAttribute('role', 'status'); stateView.setAttribute('aria-live', 'polite');
    const message = document.createElement('p'); message.setAttribute('aria-live', 'polite');
    const note = document.createElement('small');
    note.textContent = '活动会话计时包含等待回复和短暂思考，回复后 45 秒没有交流会停止计时。达到每日时长后由小狗结束本次学习。这不是实际说话时长，也不代表发音或掌握程度。';
    let disposed = false, poll = null, snapshot = null, receivedAt = 0, lastRoute = '', sequence = 0;

    function route() {
      if (typeof options.getPersonId === 'function') {
        const current = String(options.getPersonId() || '');
        if (current !== externalPerson) { externalPerson = current; person.value = current; }
      }
      const device = String(options.getDeviceId() || '').trim();
      if (!device) throw new Error('请先选择 Puppy 设备');
      const member = person.value.trim();
      return '/api/learning/' + encodeURIComponent(device) + (member ? '/members/' + encodeURIComponent(member) : '');
    }
    function duration(seconds) {
      seconds = Math.max(0, Math.ceil(seconds));
      return Math.floor(seconds / 60) + ' 分 ' + String(seconds % 60).padStart(2, '0') + ' 秒';
    }
    function drawTime() {
      if (!snapshot || disposed) return;
      const elapsed = snapshot.active ? Math.min((performance.now() - receivedAt) / 1000,
        Math.max(0, snapshot.active_until - snapshot.server_time), snapshot.remaining_seconds) : 0;
      timerView.textContent = snapshot.local_day + ' · 活动会话今日已用 ' + duration(snapshot.used_seconds + elapsed) +
        '，剩余 ' + duration(snapshot.remaining_seconds - elapsed) + '（每天 ' + snapshot.settings.daily_minutes + ' 分钟）';
    }
    async function request(url, method = 'GET', body) {
      const init = {method};
      if (body !== undefined) { init.headers = {'Content-Type': 'application/json'}; init.body = JSON.stringify(body); }
      const response = await options.fetch(url, init);
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请检查设置，每日时长为 1–20 分钟');
      return data;
    }
    function current(requestedRoute, serial) {
      if (disposed || serial !== sequence) return false;
      try { return route() === requestedRoute; } catch (_) { return false; }
    }
    function clearView() {
      snapshot = null; timerView.textContent = ''; stateView.textContent = '';
      if (poll) { clearTimeout(poll); poll = null; }
    }
    function accept(data, requestedRoute, updateSettings, serial) {
      if (!current(requestedRoute, serial)) return false;
      snapshot = data; receivedAt = performance.now(); lastRoute = requestedRoute;
      if (updateSettings) {
        minutes.value = data.settings.daily_minutes; theme.value = data.settings.theme;
        saveProgress.checked = data.settings.save_progress; photo.checked = data.settings.photo_learning;
      }
      stateView.textContent = data.clock_adjusted ? '系统时间有变化，练习计时暂时停止，请检查电脑时间。' :
        data.remaining_seconds <= 0 ? '今天的活动会话时间已用完。' : data.active ? '本次活动会话正在计时。' : '当前未计时。';
      drawTime();
      if (poll) clearTimeout(poll);
      if (data.active) poll = setTimeout(() => { if (current(requestedRoute, serial)) refresh(false); }, 5000);
      return true;
    }
    async function load(method, body, updateSettings, suffix = '', success = '') {
      if (disposed) return;
      const serial = ++sequence;
      let requestedRoute;
      try {
        requestedRoute = route();
        if (poll) { clearTimeout(poll); poll = null; }
        if (requestedRoute !== lastRoute) clearView();
        lastRoute = requestedRoute; message.textContent = '';
        const data = await request(requestedRoute + suffix, method, body);
        if (accept(data, requestedRoute, updateSettings, serial)) message.textContent = success;
      } catch (error) {
        if (!disposed && serial === sequence && (!requestedRoute || current(requestedRoute, serial))) {
          clearView(); message.textContent = error.message;
        }
      }
    }
    function refresh(updateSettings = true) { return load('GET', undefined, updateSettings); }
    function label(text, control) { const node = document.createElement('label'); node.append(text, control); root.append(node); }
    function button(text, action) {
      const node = document.createElement('button'); node.type = 'button'; node.textContent = text;
      node.addEventListener('click', async () => {
        if (disposed || node.disabled) return;
        node.disabled = true;
        try { await action(); } catch (error) { if (!disposed) message.textContent = error.message; }
        finally { if (!disposed) node.disabled = false; }
      });
      root.append(node);
    }
    root.append(title, description); label('成员 ID ', person); label('每日练习分钟（1–20） ', minutes); label('练习主题 ', theme);
    label('保存词汇练习进度（每日用时仍计入） ', saveProgress); label('允许用 Puppy 拍照学英语 ', photo);
    button('保存安排', async () => {
      const limit = Number(minutes.value);
      if (!Number.isInteger(limit) || limit < 1 || limit > 20) throw new Error('每日时长须为 1–20 分钟');
      await load('PATCH', {daily_minutes: limit, theme: theme.value,
        save_progress: saveProgress.checked, photo_learning: photo.checked}, true, '', '英语练习安排已保存。');
    });
    button('刷新今日用时', () => load('GET', undefined, true, '', '已刷新。'));
    button('结束本次计时', () => load('POST', undefined, false, '/stop', '本次计时已结束。'));
    root.append(timerView, stateView, note, message); container.append(root);
    person.addEventListener('change', () => refresh());
    const ticker = setInterval(() => {
      try { if (!disposed && route() !== lastRoute) refresh(); }
      catch (_) { ++sequence; clearView(); message.textContent = ''; }
      drawTime();
    }, 1000);
    refresh();
    return {refresh, destroy() { disposed = true; ++sequence; clearView(); clearInterval(ticker); root.remove(); }};
  }
  global.PuppyLearningPanel = {mount: mountLearningPanel};
})(window);
