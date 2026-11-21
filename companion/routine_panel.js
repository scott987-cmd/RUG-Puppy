/* Parent-only UI. Inject authenticated fetch; this module never stores a token. */
(function (global) {
  'use strict';
  function mount(container, options) {
    if (!(container instanceof HTMLElement) || !options || typeof options.fetch !== 'function' ||
        typeof options.getDeviceId !== 'function') throw new Error('宠物资料面板配置无效');
    const root = document.createElement('section');
    const controls = [];
    const element = (tag, text) => { const node = document.createElement(tag); if (text) node.textContent = text; return node; };
    function input(placeholder, maximum) { const node = element('input'); node.placeholder = placeholder; node.maxLength = maximum; controls.push(node); return node; }
    function select(items) {
      const node = element('select');
      for (const [value, label] of items) { const item = element('option', label); item.value = value; node.append(item); }
      controls.push(node); return node;
    }
    function label(text, control) { const node = element('label', text); node.append(control); root.append(node); }
    const style = select([['gentle', '温和'], ['lively', '活泼'], ['patient', '耐心']]);
    const greeting = select([['none', '不动，只说话'], ['wave', '挥手'], ['sit', '坐下'], ['stretch', '伸懒腰'], ['hug', '拥抱']]);
    const person = input('家长登记的成员 ID', 64);
    const routineId = input('流程 ID，如 tidy-toys', 64);
    const title = input('流程名称，如收玩具', 40);
    const steps = element('textarea'); steps.rows = 8; steps.maxLength = 647; controls.push(steps);
    steps.placeholder = '每行一步，最多 8 步，每步最多 80 字';
    const choices = select([]), state = element('p'), message = element('p');
    message.setAttribute('role', 'status'); message.setAttribute('aria-live', 'polite');
    let disposed = false, sequence = 0, busy = false, profile = null, selected = null, records = [], lastContext = '';
    let externalPerson = typeof options.getPersonId === 'function' ? String(options.getPersonId() || '') : '';
    person.value = externalPerson;
    const stateNames = {idle: '尚未开始', active: '正在进行', paused: '已暂停', completed: '已完成（自行报告）'};

    function context() { return {device: String(options.getDeviceId() || '').trim(), person: person.value.trim()}; }
    function key(value) { return value.device + '\n' + value.person; }
    function profileRoute(value) { return '/api/pet-profile/' + encodeURIComponent(value.device); }
    function routinesRoute(value) { return '/api/routines/' + encodeURIComponent(value.device) + '/members/' + encodeURIComponent(value.person); }
    function setBusy(value) { busy = value; for (const control of controls) control.disabled = value; }
    async function request(url, method = 'GET', body) {
      const init = {method};
      if (body !== undefined) { init.headers = {'Content-Type': 'application/json'}; init.body = JSON.stringify(body); }
      const response = await options.fetch(url, init);
      const value = await response.json();
      if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : '请检查输入并刷新后重试');
      return value;
    }
    function showRoutine(value) {
      selected = value; routineId.value = value ? value.routine_id : ''; title.value = value ? value.title : '';
      steps.value = value ? value.steps.join('\n') : ''; routineId.readOnly = !!value;
      state.textContent = value ? stateNames[value.state] + ' · 已完成 ' + value.step_index + '/' + value.steps.length +
        ' 步' + (value.current_step ? ' · ' + value.current_step : '') : '选一个流程或新建。没有确认身份时，小狗不会读取这些进度。';
    }
    function clear() { profile = null; selected = null; records = []; choices.replaceChildren(); showRoutine(null); }
    async function refresh() {
      const captured = context(), serial = ++sequence;
      const selectedId = key(captured) === lastContext && selected ? selected.routine_id : null;
      clear(); setBusy(true); lastContext = key(captured);
      try {
        if (!captured.device) throw new Error('请先选择 Puppy 设备');
        const values = await Promise.all([request(profileRoute(captured)), captured.person ? request(routinesRoute(captured)) : Promise.resolve({routines: []})]);
        if (disposed || serial !== sequence || key(context()) !== key(captured)) return;
        profile = values[0]; style.value = profile.style; greeting.value = profile.greeting_action; records = values[1].routines;
        choices.replaceChildren();
        choices.append(element('option', '新建生活流程'));
        choices.firstChild.value = '';
        for (const item of records) { const option = element('option', item.title); option.value = item.routine_id; choices.append(option); }
        const restored = records.find(item => item.routine_id === selectedId);
        if (restored) { choices.value = restored.routine_id; showRoutine(restored); }
        message.textContent = captured.person ? '已读取当前设备与成员的安排。' : '共享资料已读取；填写成员 ID 后管理其个人流程。';
      } catch (error) {
        if (!disposed && serial === sequence && key(context()) === key(captured)) message.textContent = error.message;
      } finally { if (!disposed && serial === sequence) setBusy(false); }
    }
    function button(text, action) {
      const node = element('button', text); node.type = 'button'; controls.push(node); root.append(node);
      node.addEventListener('click', async () => {
        if (busy || disposed) return;
        const captured = context(), serial = ++sequence; setBusy(true);
        try {
          await action(captured);
          if (!disposed && serial === sequence && key(context()) === key(captured)) await refresh();
        } catch (error) {
          if (!disposed && serial === sequence && key(context()) === key(captured)) message.textContent = error.message;
        } finally { if (!disposed && serial === sequence) setBusy(false); }
      });
    }
    root.append(element('h2', '小狗资料与生活小任务'));
    root.append(element('p', '性格和招呼约定由全家共享。保存不会让小狗说话或动作；只有明确要求按约定打招呼时才可能提议动作。'));
    label('性格 ', style); label('招呼动作 ', greeting);
    button('保存共享资料', async captured => {
      if (!profile || profile.device_id !== captured.device) throw new Error('请先刷新当前设备资料');
      await request(profileRoute(captured), 'PUT', {style: style.value, greeting_action: greeting.value, expected_revision: profile.revision});
    });
    label('当前成员 ', person); label('已有流程 ', choices); label('流程 ID ', routineId); label('流程名称 ', title); label('文字步骤 ', steps);
    root.append(element('small', '生活流程仅用文字引导，不验证孩子是否真的完成。编辑并保存步骤会重置进度；每次推进只进入下一步。不会定时提醒或自动执行动作。'));
    button('保存流程并重置进度', async captured => {
      if (!captured.person || !routineId.value.trim()) throw new Error('请填写成员和流程 ID');
      const saved = await request(routinesRoute(captured) + '/' + encodeURIComponent(routineId.value.trim()), 'PUT', {
        title: title.value.trim(), steps: steps.value.split('\n').map(text => text.trim()), expected_revision: selected ? selected.revision : 0});
      if (key(context()) === key(captured)) selected = saved;
    });
    for (const [operation, text] of [['start', '开始'], ['advance', '这一步完成'], ['pause', '暂停'], ['resume', '继续'], ['reset', '重置进度']]) {
      button(text, async captured => {
        if (!selected || !captured.person) throw new Error('请先选一个已保存的流程');
        await request(routinesRoute(captured) + '/' + encodeURIComponent(selected.routine_id) + '/transition', 'POST', {
          operation, expected_revision: selected.revision, expected_step_index: selected.step_index});
      });
    }
    button('删除此流程和进度', async captured => {
      if (!selected || !captured.person) throw new Error('请先选一个流程');
      await request(routinesRoute(captured) + '/' + encodeURIComponent(selected.routine_id), 'DELETE', {expected_revision: selected.revision});
    });
    button('刷新', async () => {});
    choices.addEventListener('change', () => showRoutine(records.find(item => item.routine_id === choices.value) || null));
    person.addEventListener('change', refresh);
    root.append(state, message); container.append(root);
    const ticker = setInterval(() => {
      if (disposed) return;
      if (typeof options.getPersonId === 'function') {
        const current = String(options.getPersonId() || '');
        if (current !== externalPerson) { externalPerson = current; person.value = current; }
      }
      if (key(context()) !== lastContext) refresh();
    }, 500);
    refresh();
    return {refresh, destroy() { disposed = true; ++sequence; clearInterval(ticker); root.remove(); }};
  }
  global.PuppyRoutinePanel = {mount};
})(window);
