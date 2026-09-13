/**
 * 登录页逻辑。
 *
 * 启动时检查已登录则跳 index.html；表单提交调 /auth/login；
 * 成功后缓存 token + user，跳 index.html。
 */
const { createApp, ref, onMounted, nextTick } = Vue;

createApp({
    setup() {
        const username = ref('');
        const password = ref('');
        const showPassword = ref(false);
        const loading = ref(false);
        const errorMsg = ref('');
        const usernameInput = ref(null);

        onMounted(async () => {
            const user = await DA.fetchCurrentUser();
            if (user) {
                window.location.href = 'index.html';
                return;
            }
            nextTick(() => {
                if (usernameInput.value) usernameInput.value.focus();
            });
        });

        const onSubmit = async () => {
            errorMsg.value = '';
            if (!username.value.trim() || !password.value) {
                errorMsg.value = '请输入用户名和密码';
                return;
            }

            loading.value = true;
            try {
                const resp = await DA.apiPost('/auth/login', {
                    username: username.value.trim(),
                    password: password.value
                });
                if (resp && resp.code === 200 && resp.data) {
                    DA.setToken(resp.data.token);
                    DA.cacheUser(resp.data);
                    DA.showToast('登录成功');
                    setTimeout(() => {
                        window.location.href = 'index.html';
                    }, 300);
                } else {
                    errorMsg.value = (resp && resp.msg) || '登录失败';
                }
            } catch (e) {
                errorMsg.value = e.message || '登录失败，请稍后重试';
            } finally {
                loading.value = false;
            }
        };

        return {
            username, password, showPassword,
            loading, errorMsg, usernameInput,
            onSubmit
        };
    }
}).mount('#da-login-app');
