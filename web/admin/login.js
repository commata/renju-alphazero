'use strict';
document.querySelector('#login').addEventListener('submit', async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button');
  const message = document.querySelector('#message');
  button.disabled = true;
  message.textContent = '로그인 중…';
  try {
    const response = await fetch('/renju/admin/login', {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username: form.elements.username.value, password: form.elements.password.value})
    });
    form.elements.password.value = '';
    if (response.ok) {
      location.replace('/renju/admin/');
      return;
    }
    const data = await response.json();
    message.textContent = data.error || '로그인하지 못했습니다.';
  } catch {
    message.textContent = '서버에 연결할 수 없습니다. 잠시 후 다시 시도하세요.';
  } finally {
    button.disabled = false;
  }
});
