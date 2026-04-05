/**
 * ProductMind — common.js
 * Shared module: auth, nav, transitions, query storage, and backend API calls.
 */

/* ─────────────────────────────────────────────
   CONSTANTS
───────────────────────────────────────────── */
const AUTH_KEY = 'pm_auth_token';
const USERNAME_KEY = 'pm_username';
const QUERY_KEY = 'pm_last_query';
const API_BASE = 'http://localhost:8000';  // FastAPI backend URL

const PAGES = {
  discover: 'home.html',
  compare: 'compare.html',
  saved: 'saved.html',
  insights: 'insights.html',
  auth: 'authenticationpage.html',
};

/* ─────────────────────────────────────────────
   BACKEND API HELPERS
───────────────────────────────────────────── */

/** POST /signup — register a new user */
async function apiSignup(username, email, password) {
  const res = await fetch(`${API_BASE}/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Signup failed.');
  return data;
}

/** POST /login — get JWT token */
async function apiLogin(username, password) {
  const res = await fetch(`${API_BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Login failed.');
  return data;
}

/** POST /recommend — AI recommendations (needs JWT) */
async function apiRecommend(query) {
  const token = localStorage.getItem(AUTH_KEY);
  if (!token) throw new Error('Not authenticated.');
  const res = await fetch(`${API_BASE}/recommend`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ query }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Recommendation failed.');
  return data;
}

/* ─────────────────────────────────────────────
   PAGE TRANSITION HELPERS
───────────────────────────────────────────── */
function _initFadeIn() {
  if (!document.getElementById('pm-transition-style')) {
    const style = document.createElement('style');
    style.id = 'pm-transition-style';
    style.textContent = `
      body { opacity: 0; transition: opacity 220ms ease; }
      body.pm-visible { opacity: 1; }
      body.pm-fade-out { opacity: 0 !important; transition: opacity 180ms ease !important; }
    `;
    document.head.appendChild(style);
  }
  requestAnimationFrame(() => {
    requestAnimationFrame(() => document.body.classList.add('pm-visible'));
  });
}

function navigateTo(url) {
  document.body.classList.remove('pm-visible');
  document.body.classList.add('pm-fade-out');
  setTimeout(() => { window.location.href = url; }, 190);
}

/* ─────────────────────────────────────────────
   AUTH HELPERS
───────────────────────────────────────────── */
function isAuthenticated() {
  return !!localStorage.getItem(AUTH_KEY);
}

function checkAuth() {
  if (!isAuthenticated()) {
    window.location.replace(PAGES.auth);
  }
}

/** Store real JWT token from backend */
function login(token, username) {
  localStorage.setItem(AUTH_KEY, token);
  if (username) localStorage.setItem(USERNAME_KEY, username);
}

function logout() {
  localStorage.removeItem(AUTH_KEY);
  localStorage.removeItem(USERNAME_KEY);
  navigateTo(PAGES.auth);
}

/* ─────────────────────────────────────────────
   QUERY STORAGE
───────────────────────────────────────────── */
function saveQuery(q) {
  if (q && q.trim()) localStorage.setItem(QUERY_KEY, q.trim());
}

function getLastQuery() {
  return localStorage.getItem(QUERY_KEY) || '';
}

/* ─────────────────────────────────────────────
   NAV HTML BUILDERS
───────────────────────────────────────────── */

const NAV_ITEMS = [
  { key: 'discover', label: 'Discover', page: PAGES.discover },
  { key: 'compare', label: 'Compare', page: PAGES.compare },
  { key: 'saved', label: 'Saved', page: PAGES.saved },
  { key: 'insights', label: 'Insights', page: PAGES.insights },
];

const SIDE_ITEMS = [
  { key: 'discover', label: 'Home', icon: 'grid_view', page: PAGES.discover },
  { key: 'compare', label: 'Compare', icon: 'compare_arrows', page: PAGES.compare },
  { key: 'saved', label: 'Saved', icon: 'bookmark', page: PAGES.saved },
  { key: 'insights', label: 'Insights', icon: 'insights', page: PAGES.insights },
];

const BOTTOM_ITEMS = [
  { key: 'discover', label: 'Discover', icon: 'explore', page: PAGES.discover },
  { key: 'compare', label: 'Compare', icon: 'compare_arrows', page: PAGES.compare },
  { key: 'saved', label: 'Saved', icon: 'bookmark', page: PAGES.saved },
  { key: 'insights', label: 'Insights', icon: 'insights', page: PAGES.insights },
];

function renderTopNav(activePage) {
  const mount = document.getElementById('topnav-mount');
  if (!mount) return;

  const navLinks = NAV_ITEMS.map(item => {
    const isActive = item.key === activePage;
    const activeClass = isActive
      ? 'text-[#a3a6ff] border-b-2 border-[#a3a6ff] pb-1'
      : 'text-[#dee5ff]/70 hover:text-[#dee5ff] transition-colors';
    return `<a class="${activeClass} cursor-pointer font-light tracking-tight" onclick="navigateTo('${item.page}')">${item.label}</a>`;
  }).join('\n');

  const user = localStorage.getItem(USERNAME_KEY) || '';

  mount.innerHTML = `
    <nav class="fixed top-0 w-full z-50 bg-[#091328]/70 backdrop-blur-xl shadow-[0_8px_32px_rgba(96,99,238,0.06)] flex justify-between items-center px-8 py-4 font-['Manrope']">
      <div class="text-2xl font-bold tracking-tighter bg-gradient-to-br from-[#a3a6ff] to-[#ac8aff] bg-clip-text text-transparent cursor-pointer" onclick="navigateTo('${PAGES.discover}')">
        ProductMind
      </div>
      <div class="hidden md:flex items-center space-x-8">
        ${navLinks}
      </div>
      <div class="flex items-center space-x-4">
        ${user ? `<span class="text-[#a3a6ff]/70 text-xs font-label hidden md:block">${user}</span>` : ''}
        <button class="material-symbols-outlined text-[#dee5ff]/70 hover:text-[#dee5ff] active:scale-95 transition-all p-2 rounded-full hover:bg-white/5">notifications</button>
        <button
          id="pm-logout-btn"
          onclick="logout()"
          title="Logout"
          class="w-8 h-8 rounded-full bg-gradient-to-br from-[#a3a6ff] to-[#ac8aff] flex items-center justify-center text-[#060e20] hover:scale-105 transition-transform"
        >
          <span class="material-symbols-outlined text-base">logout</span>
        </button>
      </div>
    </nav>
  `;
}

function renderSidebar(activePage) {
  const mount = document.getElementById('sidenav-mount');
  if (!mount) return;

  const links = SIDE_ITEMS.map(item => {
    const isActive = item.key === activePage;
    const fillStyle = isActive ? "font-variation-settings:'FILL' 1;" : '';
    if (isActive) {
      return `
        <a class="flex items-center gap-3 bg-[#192540] text-[#a3a6ff] rounded-full mx-2 px-4 py-3 transition-all cursor-pointer" onclick="navigateTo('${item.page}')">
          <span class="material-symbols-outlined" style="${fillStyle}">${item.icon}</span>
          <span>${item.label}</span>
        </a>`;
    }
    return `
      <a class="flex items-center gap-4 px-6 py-3 text-[#40485d] hover:bg-[#141f38] hover:text-[#dee5ff] transition-all hover:translate-x-1 duration-200 cursor-pointer" onclick="navigateTo('${item.page}')">
        <span class="material-symbols-outlined">${item.icon}</span>
        ${item.label}
      </a>`;
  }).join('\n');

  mount.innerHTML = `
    <aside class="hidden lg:flex flex-col py-8 h-screen w-64 fixed left-0 top-0 z-40 bg-[#091328] shadow-[12px_0_32px_rgba(0,0,0,0.4)] font-['Inter'] font-medium tracking-wide">
      <div class="px-6 mb-12 flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-[#a3a6ff] to-[#ac8aff] flex items-center justify-center shadow-lg">
          <span class="material-symbols-outlined text-[#060e20]" style="font-variation-settings:'FILL' 1;">auto_awesome</span>
        </div>
        <div>
          <div class="text-lg font-bold text-[#dee5ff] leading-none">ProductMind</div>
          <div class="text-xs text-[#a3a6ff] opacity-60">Intelligence v2.4</div>
        </div>
      </div>
      <div class="flex-1 space-y-2">
        ${links}
      </div>
      <div class="px-6 mb-4">
        <button class="w-full py-3 bg-gradient-to-br from-[#a3a6ff] to-[#ac8aff] text-[#0f00a4] font-bold rounded-full shadow-[0_4px_20px_rgba(163,166,255,0.3)] hover:scale-105 transition-transform">
          Upgrade to Pro
        </button>
      </div>
      <div class="px-6">
        <a class="flex items-center gap-4 py-3 text-[#40485d] hover:text-[#ff6e84] transition-colors cursor-pointer" onclick="logout()">
          <span class="material-symbols-outlined">logout</span>
          Log Out
        </a>
      </div>
    </aside>
  `;
}

function renderBottomNav(activePage) {
  const mount = document.getElementById('bottomnav-mount');
  if (!mount) return;

  const tabs = BOTTOM_ITEMS.map(item => {
    const isActive = item.key === activePage;
    const colorClass = isActive ? 'text-[#a3a6ff]' : 'text-[#dee5ff]/60';
    const fillStyle = isActive ? "font-variation-settings:'FILL' 1;" : '';
    return `
      <a class="flex flex-col items-center gap-1 ${colorClass} cursor-pointer" onclick="navigateTo('${item.page}')">
        <span class="material-symbols-outlined" style="${fillStyle}">${item.icon}</span>
        <span class="text-[10px] uppercase tracking-widest font-bold">${item.label}</span>
      </a>`;
  }).join('\n');

  mount.innerHTML = `
    <nav class="md:hidden fixed bottom-0 left-0 w-full bg-[#091328]/90 backdrop-blur-2xl z-50 flex justify-around items-center py-4 px-6 border-t border-[#40485d]/20 shadow-[0_-8px_32px_rgba(0,0,0,0.5)]">
      ${tabs}
    </nav>
  `;
}

/* ─────────────────────────────────────────────
   BOOTSTRAP — runs on every page load
───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  _initFadeIn();
});
