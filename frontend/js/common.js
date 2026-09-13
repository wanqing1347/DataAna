/**
 * DataAna 前端共享工具层（所有页面通过 <script src="js/common.js"></script> 引入）。
 *
 * 暴露全局对象 window.DA，提供：
 *   - token 管理（localStorage 持久化）
 *   - 统一 API 请求封装（自动带 satoken header，401 自动跳登录页）
 *   - SSE 流式请求封装（POST + fetch ReadableStream）
 *   - 当前用户管理（缓存 + 刷新）
 *   - 鉴权守卫（requireAuth / requireAdmin）
 *   - 主题切换 / Toast / 顶栏渲染
 *
 * 设计原则：与具体页面解耦，不依赖 Vue，纯原生 JS。
 */
(function () {
    'use strict';

    // ==================== 常量 ====================
    const TOKEN_KEY = 'da-token';
    const USER_KEY = 'da-user';
    const THEME_KEY = 'da-theme';
    // 支持部署在域名子路径（如 /DataAna/）下：API 请求跟随当前应用前缀，
    // 仍兼容直接以根路径运行的本地开发模式。
    const APP_BASE_PATH = /^\/DataAna(?:\/|$)/i.test(window.location.pathname)
        ? '/DataAna'
        : '';
    const BACKEND_URL = window.location.origin + APP_BASE_PATH;

    // ==================== token 管理 ====================
    function getToken() {
        try { return localStorage.getItem(TOKEN_KEY) || ''; } catch (e) { return ''; }
    }

    function setToken(token) {
        try { localStorage.setItem(TOKEN_KEY, token || ''); } catch (e) { /* ignore */ }
    }

    function clearToken() {
        try {
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USER_KEY);
        } catch (e) { /* ignore */ }
    }

    // ==================== 当前用户管理 ====================
    /**
     * 拉取并缓存当前登录用户完整信息（含角色/部门/dataScope）。
     * 失败（401）返回 null，调用方可据 requireAuth 决定是否跳转。
     */
    async function fetchCurrentUser() {
        const token = getToken();
        if (!token) {
            return null;
        }
        try {
            const resp = await fetch(BACKEND_URL + '/auth/me', {
                method: 'GET',
                headers: { 'satoken': token }
            });
            if (resp.status === 401) {
                clearToken();
                return null;
            }
            if (!resp.ok) {
                return null;
            }
            const json = await resp.json();
            if (json && json.code === 200 && json.data) {
                cacheUser(json.data);
                return json.data;
            }
            return null;
        } catch (e) {
            console.error('[DA] fetchCurrentUser 失败', e);
            return null;
        }
    }

    function cacheUser(user) {
        try { localStorage.setItem(USER_KEY, JSON.stringify(user || {})); } catch (e) { /* ignore */ }
    }

    function getCachedUser() {
        try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null'); } catch (e) { return null; }
    }

    function isAdmin(user) {
        const u = user || getCachedUser();
        if (!u || !Array.isArray(u.roles)) return false;
        return u.roles.some(r => r && r.code === 'admin');
    }

    // ==================== 鉴权守卫 ====================
    /**
     * 检查登录状态，未登录则跳转 /login.html。
     * @returns 当前用户（已登录）或 null（已跳转）
     */
    async function requireAuth() {
        let user = getCachedUser();
        if (!user) {
            user = await fetchCurrentUser();
        }
        if (!user) {
            redirectToLogin();
            return null;
        }
        return user;
    }

    /**
     * 检查 admin 角色，非 admin 跳回主页。
     * @returns 当前用户（admin）或 null（已跳转）
     */
    async function requireAdmin() {
        const user = await requireAuth();
        if (!user) return null;
        if (!isAdmin(user)) {
            window.location.href = 'index.html';
            return null;
        }
        return user;
    }

    function redirectToLogin() {
        const current = window.location.pathname.split('/').pop() || 'index.html';
        if (current !== 'login.html') {
            window.location.href = 'login.html';
        }
    }

    async function logout() {
        try {
            await apiPost('/auth/logout', {});
        } catch (e) { /* 忽略，无论如何都跳转 */ }
        clearToken();
        window.location.href = 'login.html';
    }

    // ==================== 统一请求封装 ====================
    function buildHeaders(extra) {
        const token = getToken();
        const headers = Object.assign({ 'Content-Type': 'application/json' }, extra || {});
        if (token) headers['satoken'] = token;
        return headers;
    }

    function handleResponse401() {
        clearToken();
        redirectToLogin();
    }

    async function apiGet(url, params) {
        const qs = params ? '?' + new URLSearchParams(params).toString() : '';
        const resp = await fetch(BACKEND_URL + url + qs, {
            method: 'GET',
            headers: buildHeaders()
        });
        return await parseResponse(resp);
    }

    async function apiPost(url, data) {
        const resp = await fetch(BACKEND_URL + url, {
            method: 'POST',
            headers: buildHeaders(),
            body: JSON.stringify(data || {})
        });
        return await parseResponse(resp);
    }

    async function apiPut(url, data) {
        const resp = await fetch(BACKEND_URL + url, {
            method: 'PUT',
            headers: buildHeaders(),
            body: JSON.stringify(data || {})
        });
        return await parseResponse(resp);
    }

    async function apiDelete(url) {
        const resp = await fetch(BACKEND_URL + url, {
            method: 'DELETE',
            headers: buildHeaders()
        });
        return await parseResponse(resp);
    }

    /**
     * 文件上传（multipart/form-data）。
     * 注意：不要手动设置 Content-Type，浏览器会自动加上 boundary。
     */
    async function apiUpload(url, formData) {
        const token = getToken();
        const headers = {};
        if (token) headers['satoken'] = token;
        const resp = await fetch(BACKEND_URL + url, {
            method: 'POST',
            headers: headers,
            body: formData
        });
        return await parseResponse(resp);
    }

    async function parseResponse(resp) {
        if (resp.status === 401) {
            handleResponse401();
            throw new Error('未登录或登录已过期');
        }
        const text = await resp.text();
        let json;
        try { json = text ? JSON.parse(text) : {}; } catch (e) {
            throw new Error('响应解析失败：' + text.slice(0, 200));
        }
        if (resp.status === 403) {
            const msg = (json && json.msg) || '无权限访问';
            throw new Error(msg);
        }
        if (!resp.ok) {
            const msg = (json && json.msg) || ('HTTP ' + resp.status);
            throw new Error(msg);
        }
        return json;
    }

    /**
     * SSE 流式请求（POST + fetch ReadableStream）。
     *
     * Spring MVC 返回的 SSE 格式：每个 event 以 \n\n 分隔，data 行以 "data:" 前缀。
     *
     * @param url 后端接口路径（如 /agent/stream）
     * @param body 请求体（会被 JSON.stringify）
     * @param handlers 回调对象：
     *        { onEvent(eventData, eventName), onError(err), onClose() }
     * @returns AbortController（用于中止请求）
     */
    function apiStream(url, body, handlers) {
        const controller = new AbortController();
        const token = getToken();
        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        };
        if (token) headers['satoken'] = token;

        fetch(BACKEND_URL + url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(body || {}),
            signal: controller.signal
        }).then(async (resp) => {
            if (resp.status === 401) {
                handleResponse401();
                if (handlers.onError) handlers.onError(new Error('未登录或登录已过期'));
                return;
            }
            if (!resp.ok || !resp.body) {
                const txt = await resp.text().catch(() => '');
                if (handlers.onError) handlers.onError(new Error(txt || ('HTTP ' + resp.status)));
                return;
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    // 按 SSE 标准的 \n\n 切分 event
                    let idx;
                    while ((idx = buffer.indexOf('\n\n')) >= 0) {
                        const eventStr = buffer.slice(0, idx);
                        buffer = buffer.slice(idx + 2);
                        const parsed = parseSseEvent(eventStr);
                        if (parsed && handlers.onEvent) {
                            handlers.onEvent(parsed.data, parsed.event);
                        }
                    }
                }
                if (handlers.onClose) handlers.onClose();
            } catch (e) {
                if (e && e.name === 'AbortError') {
                    if (handlers.onClose) handlers.onClose();
                } else if (handlers.onError) {
                    const detail = e && e.message ? e.message : String(e || '未知网络错误');
                    handlers.onError(new Error('数据分析流连接中断：' + detail));
                }
            }
        }).catch((e) => {
            if (e && e.name === 'AbortError') {
                if (handlers.onClose) handlers.onClose();
            } else if (handlers.onError) {
                const detail = e && e.message ? e.message : String(e || '未知网络错误');
                handlers.onError(new Error('无法连接数据分析流接口：' + detail));
            }
        });

        return controller;
    }

    function parseSseEvent(eventStr) {
        const lines = eventStr.split('\n');
        let eventName = 'message';
        let data = '';
        for (const line of lines) {
            if (line.startsWith('event:')) {
                eventName = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
                data += line.slice(5).trim();
            }
        }
        return { event: eventName, data };
    }

    // ==================== UI 工具 ====================
    function getTheme() {
        try { return localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light'; } catch (e) { return 'light'; }
    }

    function applyTheme(t) {
        document.documentElement.setAttribute('data-theme', t);
        const lightLink = document.getElementById('hljs-light');
        const darkLink = document.getElementById('hljs-dark');
        if (lightLink) lightLink.disabled = (t === 'dark');
        if (darkLink) darkLink.disabled = (t !== 'dark');
        try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* ignore */ }
    }

    function toggleTheme() {
        const next = getTheme() === 'light' ? 'dark' : 'light';
        applyTheme(next);
        return next;
    }

    /**
     * 全局 Toast 提示（操作 DOM，不依赖 Vue）。
     */
    function showToast(msg, duration) {
        if (!msg) return;
        let container = document.getElementById('da-toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'da-toast-container';
            container.className = 'da-toast-container';
            document.body.appendChild(container);
        }
        const el = document.createElement('div');
        el.className = 'da-toast';
        el.textContent = msg;
        container.appendChild(el);
        // 触发淡入动画
        requestAnimationFrame(() => el.classList.add('show'));
        const ms = duration || 2000;
        setTimeout(() => {
            el.classList.remove('show');
            setTimeout(() => el.remove(), 300);
        }, ms);
    }

    /**
     * 渲染公共顶栏（登录后的所有页面都用）。
     *
     * @param {Object} options
     *        - active: 当前激活的菜单 key（'analysis'）
     *        - user: 当前用户对象（含 nickname/username/roles）
     *        - sidebarToggle: 是否显示侧栏折叠按钮（仅聊天页用）
     */
    function renderHeader(options) {
        const opts = options || {};
        const user = opts.user || getCachedUser() || {};
        const realName = user.profile && user.profile.realName;
        const nick = user.nickname || user.username;
        // 有真实姓名时显示"真实姓名(昵称)"，否则回退到昵称/用户名/游客
        const displayName = realName
            ? (nick ? `${realName}(${nick})` : realName)
            : (nick || '游客');

        // DataAna 前端只展示数据分析入口；管理页仍由后端鉴权保护。
        const navItems = [
            { key: 'analysis', href: 'index.html', icon: 'fa-chart-line', text: '数据分析' }
        ];

        const navHtml = navItems.map(item =>
            `<a href="${item.href}" class="da-nav-link ${opts.active === item.key ? 'active' : ''}">
                <i class="fas ${item.icon}"></i><span>${item.text}</span>
            </a>`
        ).join('');

        const initial = (displayName || '?').charAt(0).toUpperCase();
        const sidebarToggleHtml = opts.sidebarToggle
            ? `<button id="da-sidebar-toggle" class="da-sidebar-toggle" title="收起/展开侧栏">
                   <i class="fas fa-outdent"></i>
               </button>`
            : '';

        const roleTags = (Array.isArray(user.roles) && user.roles.length > 0)
            ? user.roles
                .filter(r => r && (r.name || r.code))
                .map(r => `<span class="da-tag da-tag-role">${escapeHtml(r.name || r.code)}</span>`)
                .join('')
            : `<span class="da-tag da-tag-empty">未分配</span>`;
        const deptTags = (Array.isArray(user.depts) && user.depts.length > 0)
            ? user.depts
                .filter(d => d && (Array.isArray(d.path) ? d.path.length > 0 : d.name))
                .map(d => {
                    const text = (Array.isArray(d.path) && d.path.length > 0)
                        ? d.path.join(' / ')
                        : (d.name || '');
                    return `<span class="da-tag da-tag-dept" title="${escapeHtml(text)}">${escapeHtml(text)}</span>`;
                })
                .join('')
            : `<span class="da-tag da-tag-empty">未分配</span>`;

        return `
        <header class="da-header">
            ${sidebarToggleHtml}
            <div class="da-brand">
                <span class="da-logo"><i class="fas fa-chart-simple"></i></span>
                <span class="da-title">DataAna</span>
                <span class="da-badge">ANALYTICS</span>
            </div>
            <nav class="da-header-nav">${navHtml}</nav>
            <div class="da-header-actions">
                <button class="da-theme-toggle" title="切换主题">
                    <i class="fas fa-moon icon-moon"></i>
                    <i class="fas fa-sun icon-sun"></i>
                </button>
                <div class="da-user-menu">
                    <div class="da-user-avatar">${initial}</div>
                    <span class="da-user-name">${escapeHtml(displayName)}</span>
                    <i class="fas fa-chevron-down"></i>
                    <div class="da-user-dropdown">
                        <div class="da-user-info">
                            <div class="da-user-info-name">${escapeHtml(displayName)}</div>
                            <div class="da-user-info-id">@${escapeHtml(user.username || '')}</div>
                            <div class="da-user-info-meta">
                                <span class="da-user-info-label">角色</span>
                                <div class="da-user-info-tags">${roleTags}</div>
                            </div>
                            <div class="da-user-info-meta">
                                <span class="da-user-info-label">部门</span>
                                <div class="da-user-info-tags">${deptTags}</div>
                            </div>
                        </div>
                        <button class="da-user-dropdown-item" data-action="profile">
                            <i class="fas fa-id-card"></i><span>个人档案</span>
                        </button>
                        <button class="da-user-dropdown-item" data-action="logout">
                            <i class="fas fa-right-from-bracket"></i><span>退出登录</span>
                        </button>
                    </div>
                </div>
            </div>
        </header>`;
    }

    /**
     * 绑定顶栏交互（主题切换、个人档案、退出登录）。
     * 在 renderHeader 后调用。
     */
    function bindHeaderEvents() {
        const themeBtn = document.querySelector('.da-theme-toggle');
        if (themeBtn) {
            themeBtn.addEventListener('click', () => toggleTheme());
        }
        const profileBtn = document.querySelector('.da-user-dropdown-item[data-action="profile"]');
        if (profileBtn) {
            profileBtn.addEventListener('click', () => {
                const userMenu = document.querySelector('.da-user-menu');
                if (userMenu) userMenu.classList.remove('open');
                openProfileModal();
            });
        }
        const logoutBtn = document.querySelector('.da-user-dropdown-item[data-action="logout"]');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => logout());
        }
        // 用户菜单点击展开/收起
        const userMenu = document.querySelector('.da-user-menu');
        if (userMenu) {
            userMenu.addEventListener('click', (e) => {
                userMenu.classList.toggle('open');
                e.stopPropagation();
            });
            document.addEventListener('click', () => userMenu.classList.remove('open'));
        }
    }

    /**
     * 打开"个人档案"弹框（只读展示，编辑入口在 admin 用户管理页）。
     */
    function openProfileModal() {
        const user = getCachedUser() || {};
        // 已存在则先关闭，避免叠层
        closeProfileModal();

        const realName = user.profile && user.profile.realName;
        const nick = user.nickname || user.username;
        const displayName = realName || nick || '游客';
        const initial = (displayName || '?').charAt(0).toUpperCase();
        const p = user.profile || {};

        const roleTags = (Array.isArray(user.roles) && user.roles.length > 0)
            ? user.roles.filter(r => r && (r.name || r.code))
                .map(r => `<span class="da-profile-tag">${escapeHtml(r.name || r.code)}</span>`).join('')
            : `<span class="da-profile-tag">未分配</span>`;
        const deptTags = (Array.isArray(user.depts) && user.depts.length > 0)
            ? user.depts.filter(d => d && (Array.isArray(d.path) ? d.path.length > 0 : d.name))
                .map(d => {
                    const text = (Array.isArray(d.path) && d.path.length > 0)
                        ? d.path.join(' / ') : (d.name || '');
                    return `<span class="da-profile-tag">${escapeHtml(text)}</span>`;
                }).join('')
            : `<span class="da-profile-tag">未挂载</span>`;

        const row = (label, value) => `<div class="da-profile-grid-label">${label}</div>` +
            `<div class="da-profile-grid-value">${value == null || value === '' ? '—' : escapeHtml(String(value))}</div>`;

        const mask = document.createElement('div');
        mask.className = 'da-profile-mask';
        mask.innerHTML = `
            <div class="da-profile-card">
                <div class="da-profile-header">
                    <div class="da-profile-avatar">${escapeHtml(initial)}</div>
                    <div class="da-profile-header-text">
                        <div class="da-profile-header-name">${escapeHtml(displayName)}</div>
                        <div class="da-profile-header-id">@${escapeHtml(user.username || '')}</div>
                    </div>
                    <button class="da-profile-close" title="关闭"><i class="fas fa-xmark"></i></button>
                </div>
                <div class="da-profile-body">
                    <div class="da-profile-section-title">身份</div>
                    <div class="da-profile-grid">
                        ${row('用户 ID', user.id)}
                        ${row('用户名', user.username)}
                        ${row('昵称', user.nickname)}
                    </div>
                    <div class="da-profile-section-title">个人档案</div>
                    <div class="da-profile-grid">
                        ${row('真实姓名', p.realName)}
                        ${row('身份证号', p.idCard)}
                        ${row('年龄', p.age)}
                        ${row('学历', p.education)}
                        ${row('家庭住址', p.homeAddress)}
                    </div>
                    <div class="da-profile-section-title">角色</div>
                    <div class="da-profile-tags">${roleTags}</div>
                    <div class="da-profile-section-title">部门</div>
                    <div class="da-profile-tags">${deptTags}</div>
                </div>
            </div>`;
        mask.addEventListener('click', (e) => {
            if (e.target === mask || e.target.closest('.da-profile-close')) {
                closeProfileModal();
            }
        });
        document.addEventListener('keydown', profileEscHandler);
        document.body.appendChild(mask);
    }

    function profileEscHandler(e) {
        if (e.key === 'Escape') closeProfileModal();
    }

    function closeProfileModal() {
        document.removeEventListener('keydown', profileEscHandler);
        document.querySelectorAll('.da-profile-mask').forEach(el => el.remove());
    }

    function escapeHtml(s) {
        if (s == null) return '';
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // ==================== 暴露全局 ====================
    window.DA = {
        // token
        getToken, setToken, clearToken,
        // user
        fetchCurrentUser, getCachedUser, cacheUser, isAdmin,
        requireAuth, requireAdmin, logout,
        // api
        apiGet, apiPost, apiPut, apiDelete, apiUpload, apiStream,
        // ui
        getTheme, applyTheme, toggleTheme, showToast,
        renderHeader, bindHeaderEvents, escapeHtml,
        // 常量
        BACKEND_URL
    };

    // 启动时立即应用主题，避免白屏闪烁
    applyTheme(getTheme());
})();
