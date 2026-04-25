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
const HISTORY_KEY = 'pm_user_history';
const MAX_HISTORY = 20;
const SAVED_KEY = 'pm_saved_products';
const RESULTS_KEY = 'pm_last_results';
const API_BASE = 'http://localhost:8000';  // FastAPI backend URL

const PAGES = {
  discover: 'home.html',
  compare: 'compare.html',
  saved: 'saved.html',
  insights: 'insights.html',
  auth: 'authenticationpage.html',
};

const PRODUCT_IMAGE_LIBRARY = {
  mobile: [
    'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1598327105666-5b89351aff97?auto=format&fit=crop&w=1200&q=80',
  ],
  laptop: [
    'https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1517336714739-489689fd1ca8?auto=format&fit=crop&w=1200&q=80',
  ],
  headphones: [
    'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1546435770-a3e426bf472b?auto=format&fit=crop&w=1200&q=80',
  ],
  earphones: [
    'https://images.unsplash.com/photo-1606220588913-b3aacb4d2f46?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1572569511254-d8f925fe2cbb?auto=format&fit=crop&w=1200&q=80',
  ],
  monitor: [
    'https://images.unsplash.com/photo-1527443154391-507e9dc6c5cc?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1547082299-de196ea013d6?auto=format&fit=crop&w=1200&q=80',
  ],
  tablet: [
    'https://images.unsplash.com/photo-1544244015-0df4b3ffc6b0?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1561154464-82e9adf32764?auto=format&fit=crop&w=1200&q=80',
  ],
  keyboard: [
    'https://images.unsplash.com/photo-1511467687858-23d96c32e4ae?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1515879218367-8466d910aaa4?auto=format&fit=crop&w=1200&q=80',
  ],
  mouse: [
    'https://images.unsplash.com/photo-1527814050087-3793815479db?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?auto=format&fit=crop&w=1200&q=80',
  ],
  webcam: [
    'https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1587614382346-4ec70e388b28?auto=format&fit=crop&w=1200&q=80',
  ],
  speaker: [
    'https://images.unsplash.com/photo-1545454675-3531b543be5d?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1519677100203-a0e668c92439?auto=format&fit=crop&w=1200&q=80',
  ],
  chair: [
    'https://images.unsplash.com/photo-1505843513577-22bb7d21e455?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1580480055273-228ff5388ef8?auto=format&fit=crop&w=1200&q=80',
  ],
  desk: [
    'https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=1200&q=80',
    'https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?auto=format&fit=crop&w=1200&q=80',
  ],
  generic: [
    'https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=1200&q=80',
  ],
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
async function apiRecommend(query, history = []) {
  const token = localStorage.getItem(AUTH_KEY);
  if (!token) throw new Error('Not authenticated.');
  const res = await fetch(`${API_BASE}/recommend`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ query, history }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Recommendation failed.');
  return data;
}

/* ─────────────────────────────────────────────
   DEBOUNCE
───────────────────────────────────────────── */
/**
 * Wraps a function with a delay to prevent rapid API calls.
 */
function debounce(fn, delay = 500) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
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

function saveResults(results) {
  if (results) localStorage.setItem(RESULTS_KEY, JSON.stringify(results));
}

function getLastResults() {
  try {
    const raw = localStorage.getItem(RESULTS_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function clearSearchPersistence() {
  localStorage.removeItem(QUERY_KEY);
  localStorage.removeItem(RESULTS_KEY);
}

/* ─────────────────────────────────────────────
   PRODUCT IMAGES
───────────────────────────────────────────── */
/**
 * Returns a normalized visual type for the product so we can show a matching image.
 */
function getProductVisualType(product) {
  const text = [
    product?.name || '',
    product?.category || '',
    ...(product?.tags || []),
    ...(product?.features || []),
  ].join(' ').toLowerCase();

  if (/(earbud|earbuds|earphone|earphones|tws|neckband)/.test(text)) return 'earphones';
  if (/(headphone|headphones|headset)/.test(text)) return 'headphones';
  if (/(mobile|smartphone|iphone|android phone|camera phone|gaming phone)/.test(text)) return 'mobile';
  if (/(laptop|notebook|ultrabook|macbook)/.test(text)) return 'laptop';
  if (/(monitor|display|usb-c monitor|qhd monitor)/.test(text)) return 'monitor';
  if (/(tablet|ipad)/.test(text)) return 'tablet';
  if (/(keyboard|mechanical keyboard)/.test(text)) return 'keyboard';
  if (/(mouse|gaming mouse)/.test(text)) return 'mouse';
  if (/(webcam)/.test(text)) return 'webcam';
  if (/(speaker|soundbar)/.test(text)) return 'speaker';
  if (/(chair|office chair|ergonomic chair)/.test(text)) return 'chair';
  if (/(desk|table|study table)/.test(text)) return 'desk';
  return 'generic';
}

/**
 * Returns the preferred product image URL.
 */
function getProductImageUrl(product) {
  if (product && product.image_url) return product.image_url;
  const name = product?.name || product?.category || 'product';
  return `https://source.unsplash.com/400x300/?${encodeURIComponent(name)}`;
}

/**
 * Enhanced image sources with fallbacks.
 */
function getProductImageSources(product) {
  const name = product?.name || product?.category || 'product';
  const url = product?.image_url || `https://source.unsplash.com/400x300/?${encodeURIComponent(name)}`;
  return [
    url,
    `https://via.placeholder.com/400x300?text=${encodeURIComponent(product?.name || 'Product')}`
  ];
}

/**
 * Wires image fallback swapping for any rendered product images inside a container.
 */
function wireProductImageFallbacks(root = document) {
  root.querySelectorAll('img[data-product-image]').forEach(img => {
    let sources = [];
    try {
      sources = JSON.parse(img.dataset.productImage || '[]');
    } catch {
      sources = [];
    }
    if (!sources.length) return;

    let index = Number(img.dataset.imageIndex || 0);
    img.onerror = () => {
      index += 1;
      img.dataset.imageIndex = String(index);
      if (index < sources.length) {
        img.src = sources[index];
      } else {
        img.onerror = null;
      }
    };
  });
}

/* ─────────────────────────────────────────────
   USER HISTORY
───────────────────────────────────────────── */
/**
 * Adds a successful recommendation interaction to local history.
 */
function addToHistory(query, topPickName, category) {
  const history = getHistory();
  history.unshift({ query, topPickName, category, ts: Date.now() });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
}

/**
 * Reads saved interaction history from localStorage.
 */
function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
}

/* ─────────────────────────────────────────────
   SAVED PRODUCTS
───────────────────────────────────────────── */
/**
 * Saves a product locally when it is not already bookmarked.
 */
function saveProduct(product) {
  const saved = getSavedProducts();
  if (!saved.find(p => p.id === product.id)) {
    saved.unshift(product);
    localStorage.setItem(SAVED_KEY, JSON.stringify(saved));
  }
}

/**
 * Removes a saved product by id.
 */
function removeSavedProduct(productId) {
  const saved = getSavedProducts().filter(product => product.id !== productId);
  localStorage.setItem(SAVED_KEY, JSON.stringify(saved));
}

/**
 * Returns all saved products from localStorage.
 */
function getSavedProducts() {
  try {
    return JSON.parse(localStorage.getItem(SAVED_KEY) || '[]');
  } catch {
    return [];
  }
}

/**
 * Checks whether a product is already saved.
 */
function isProductSaved(productId) {
  return getSavedProducts().some(product => product.id === productId);
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
