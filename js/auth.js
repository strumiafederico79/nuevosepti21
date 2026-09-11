// js/auth.js — login/JWT/session
import { apiFetch, apiBase, setToken as setApiToken } from './api.js';
import * as state from './state.js';

export function initAuth(container) {
  container.innerHTML = `
    <div class="auth-box">
      <h2>Iniciar Sesión</h2>
      <form id="auth-form">
        <input type="email" id="auth-email" placeholder="Email" required>
        <input type="password" id="auth-pass" placeholder="Contraseña" required>
        <button type="submit">Entrar</button>
        <div id="auth-error" class="auth-error"></div>
      </form>
    </div>
  `;

  const form = container.querySelector('#auth-form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = container.querySelector('#auth-email').value;
    const pass = container.querySelector('#auth-pass').value;
    const errEl = container.querySelector('#auth-error');
    errEl.textContent = '';

    try {
      const res = await apiFetch(`${apiBase()}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password: pass }),
        maxRetries: 0,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const token = data.access_token || data.token;
      if (!token) throw new Error('No se recibió token');
      setToken(token);
      state.set('user', data.user || { email });
      state.set('authenticated', true);
    } catch (err) {
      errEl.textContent = err.message;
    }
  });
}

export function setToken(token) {
  setApiToken(token);
  state.set('token', token);
}

export function getToken() {
  return state.get('token');
}

export function getUser() {
  return state.get('user');
}

export function logout() {
  setToken(null);
  state.set('user', null);
  state.set('authenticated', false);
}
