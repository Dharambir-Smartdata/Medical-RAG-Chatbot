/**
 * Medical AI Assistant — Clinical RAG Client Application
 * Handles chat messaging, streaming UX, citations, multi-session management, and PDF uploads.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentSessionId = null;
    let selectedFiles = [];
    let isProcessing = false;
    let loadingInterval = null;

    // Elements
    const chatForm = document.getElementById('chatForm');
    const questionInput = document.getElementById('questionInput');
    const sendBtn = document.getElementById('sendBtn');
    const charCounter = document.getElementById('charCounter');
    const chatMessagesContainer = document.getElementById('chatMessagesContainer');
    const emptyChatState = document.getElementById('emptyChatState');
    const messageStream = document.getElementById('messageStream');
    const loadingBubble = document.getElementById('loadingBubble');
    const loadingStageText = document.getElementById('loadingStageText');

    // Topbar & Sidebar
    const activeSessionTitle = document.getElementById('activeSessionTitle');
    const docScopeSelect = document.getElementById('docScopeSelect');
    const sessionList = document.getElementById('sessionList');
    const sidebarDocList = document.getElementById('sidebarDocList');
    const docCountBadge = document.getElementById('docCountBadge');
    const newChatBtn = document.getElementById('newChatBtn');
    const appSidebar = document.getElementById('appSidebar');
    const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
    const sidebarCloseBtn = document.getElementById('sidebarCloseBtn');

    // Upload Modal
    const uploadModal = document.getElementById('uploadModal');
    const openUploadModalBtn = document.getElementById('openUploadModalBtn');
    const closeUploadModalBtn = document.getElementById('closeUploadModalBtn');
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    const startUploadBtn = document.getElementById('startUploadBtn');
    const pdfDropzone = document.getElementById('pdfDropzone');
    const pdfFileInput = document.getElementById('pdfFileInput');
    const uploadFileList = document.getElementById('uploadFileList');
    const uploadProgressContainer = document.getElementById('uploadProgressContainer');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const uploadStatusText = document.getElementById('uploadStatusText');

    // Citation Drawer
    const citationDrawer = document.getElementById('citationDrawer');
    const closeCitationDrawerBtn = document.getElementById('closeCitationDrawerBtn');
    const citationSourceBadge = document.getElementById('citationSourceBadge');
    const citationDocTitle = document.getElementById('citationDocTitle');
    const citationPageNum = document.getElementById('citationPageNum');
    const citationSection = document.getElementById('citationSection');
    const citationChunkId = document.getElementById('citationChunkId');
    const citationFullText = document.getElementById('citationFullText');

    // ==========================================
    // Theme Management (Dark / Light Mode)
    // ==========================================
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeLabel = document.getElementById('themeLabel');

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('medassist_theme', theme);
        if (themeLabel) {
            themeLabel.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
        }
    }

    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(currentTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const activeTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = activeTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
        });
    }

    // ==========================================
    // Transformer Model Readiness Loader
    // ==========================================
    const modelLoaderOverlay = document.getElementById('modelLoaderOverlay');
    const modelLoaderStatus = document.getElementById('modelLoaderStatus');
    const modelLoaderBar = document.getElementById('modelLoaderBar');
    const stepEmbeddings = document.getElementById('stepEmbeddings');
    const stepReranker = document.getElementById('stepReranker');
    const stepReady = document.getElementById('stepReady');

    async function checkTransformerReadiness() {
        if (!modelLoaderOverlay) return;

        let pollCount = 0;
        const maxPolls = 60;

        async function poll() {
            try {
                pollCount++;
                const resp = await fetch('/chat/model-status/');
                if (resp.ok) {
                    const data = await resp.json();

                    if (data.embedding_loaded && stepEmbeddings) {
                        stepEmbeddings.classList.add('active');
                    }
                    if (data.reranker_loaded && stepReranker) {
                        stepReranker.classList.add('active');
                    }

                    if (data.ready) {
                        if (stepReady) stepReady.classList.add('active');
                        if (modelLoaderStatus) {
                            modelLoaderStatus.textContent = 'Neural models ready! Launching workspace...';
                        }
                        if (modelLoaderBar) {
                            modelLoaderBar.style.animation = 'none';
                            modelLoaderBar.style.width = '100%';
                            modelLoaderBar.style.left = '0';
                        }

                        setTimeout(() => {
                            modelLoaderOverlay.classList.add('hidden');
                            setTimeout(() => {
                                modelLoaderOverlay.style.display = 'none';
                            }, 450);
                        }, 500);
                        return;
                    }
                }
            } catch (e) {
                console.warn('Transformer model readiness poll error:', e);
            }

            if (pollCount < maxPolls) {
                setTimeout(poll, 1000);
            } else {
                modelLoaderOverlay.classList.add('hidden');
                setTimeout(() => {
                    modelLoaderOverlay.style.display = 'none';
                }, 450);
            }
        }

        poll();
    }

    checkTransformerReadiness();

    // CSRF Helper
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    const csrftoken = getCookie('csrftoken');

    // Toast Notifications
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 250);
        }, 3500);
    }

    // Auto-expand textarea
    function autoResizeTextarea() {
        if (!questionInput) return;
        questionInput.style.height = 'auto';
        questionInput.style.height = (questionInput.scrollHeight) + 'px';
        const len = questionInput.value.length;
        if (charCounter) charCounter.textContent = `${len} / 2000`;
        if (sendBtn) sendBtn.disabled = len === 0 || isProcessing;
    }

    if (questionInput) {
        questionInput.addEventListener('input', autoResizeTextarea);
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled && !isProcessing) handleSendMessage();
            }
        });
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', (e) => {
            e.preventDefault();
            if (!sendBtn.disabled && !isProcessing) {
                handleSendMessage();
            }
        });
    }

    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            if (!sendBtn.disabled && !isProcessing) {
                handleSendMessage();
            }
        });
    }

    // ==========================================
    // Sidebar Drawer & Collapse Management
    // ==========================================
    const sidebarBackdrop = document.getElementById('sidebarBackdrop');

    function isMobileView() {
        return window.innerWidth <= 768;
    }

    function toggleSidebar(e) {
        if (e) e.stopPropagation();
        if (!appSidebar) return;

        if (isMobileView()) {
            const isOpen = appSidebar.classList.contains('open');
            if (isOpen) {
                closeSidebar();
            } else {
                openSidebar();
            }
        } else {
            appSidebar.classList.toggle('collapsed');
        }
    }

    function openSidebar() {
        if (!appSidebar) return;
        appSidebar.classList.add('open');
        if (sidebarBackdrop) sidebarBackdrop.classList.add('active');
        appSidebar.classList.remove('collapsed');
    }

    function closeSidebar(e) {
        if (e) e.stopPropagation();
        if (!appSidebar) return;

        if (isMobileView()) {
            appSidebar.classList.remove('open');
            if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
        } else {
            appSidebar.classList.add('collapsed');
        }
    }

    if (sidebarToggleBtn) {
        sidebarToggleBtn.addEventListener('click', toggleSidebar);
    }

    if (sidebarCloseBtn) {
        sidebarCloseBtn.addEventListener('click', closeSidebar);
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', closeSidebar);
    }

    // Records Repository Button Navigation Toggle
    const repoNavBtn = document.getElementById('repoNavBtn');
    if (repoNavBtn) {
        repoNavBtn.addEventListener('click', (e) => {
            if (window.location.pathname.includes('/documents/')) {
                e.preventDefault();
                window.location.href = '/';
            }
        });
    }

    window.addEventListener('resize', () => {
        if (!isMobileView()) {
            if (appSidebar) appSidebar.classList.remove('open');
            if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
        }
    });

    // Suggested Prompt Chips
    document.querySelectorAll('.prompt-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const promptText = btn.getAttribute('data-prompt');
            if (questionInput && promptText) {
                questionInput.value = promptText;
                autoResizeTextarea();
                handleSendMessage();
            }
        });
    });

    // ==========================================================================
    // Sessions Management
    // ==========================================================================
    async function loadSessions() {
        try {
            const res = await fetch('/chat/history/');
            if (!res.ok) throw new Error('Failed to load chat history');
            const data = await res.json();
            renderSessionsList(data.sessions || []);
        } catch (err) {
            console.error(err);
            if (sessionList) sessionList.innerHTML = '<div class="text-muted" style="padding: 0.5rem; font-size: 0.8rem;">No previous consultations.</div>';
        }
    }

    function renderSessionsList(sessions) {
        if (!sessionList) return;
        sessionList.innerHTML = '';
        if (sessions.length === 0) {
            sessionList.innerHTML = '<div class="text-muted" style="padding: 0.5rem; font-size: 0.8rem;">No past consultations.</div>';
            return;
        }

        sessions.forEach(sess => {
            const div = document.createElement('div');
            div.className = `session-item ${sess.id === currentSessionId ? 'active' : ''}`;
            div.innerHTML = `
                <span class="session-name-text" title="${sess.session_name}">${sess.session_name}</span>
                <button class="session-del-btn" title="Delete Consultation" data-id="${sess.id}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            `;

            div.addEventListener('click', (e) => {
                if (e.target.closest('.session-del-btn')) return;
                switchSession(sess.id, sess.session_name);
            });

            const delBtn = div.querySelector('.session-del-btn');
            delBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteSession(sess.id);
            });

            sessionList.appendChild(div);
        });
    }

    async function createNewSession() {
        if (window.location.pathname.includes('/documents/')) {
            window.location.href = '/';
            return;
        }
        try {
            const res = await fetch('/chat/new/', {
                method: 'POST',
                headers: { 'X-CSRFToken': csrftoken }
            });
            const data = await res.json();
            currentSessionId = data.session_id;
            if (activeSessionTitle) activeSessionTitle.textContent = data.session_name;

            // Reset Chat View
            if (messageStream) {
                messageStream.innerHTML = '';
                messageStream.style.display = 'none';
            }
            if (emptyChatState) emptyChatState.style.display = 'block';

            await loadSessions();
            if (isMobileView()) closeSidebar();
        } catch (err) {
            showToast('Failed to create consultation session', 'error');
        }
    }

    if (newChatBtn) {
        newChatBtn.addEventListener('click', createNewSession);
    }

    async function switchSession(sessionId, title) {
        if (window.location.pathname.includes('/documents/')) {
            window.location.href = '/';
            return;
        }
        currentSessionId = sessionId;
        if (activeSessionTitle) activeSessionTitle.textContent = title;
        if (isMobileView()) closeSidebar();

        document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
        const activeItem = document.querySelector(`.session-item button[data-id="${sessionId}"]`);
        if (activeItem) activeItem.closest('.session-item').classList.add('active');

        // Fetch Messages
        try {
            const res = await fetch(`/chat/session/${sessionId}/`);
            if (!res.ok) throw new Error('Failed to fetch messages');
            const data = await res.json();
            renderMessageStream(data.messages || []);
        } catch (err) {
            showToast('Unable to load session messages', 'error');
        }
    }

    async function deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this consultation history?')) return;
        try {
            const res = await fetch(`/chat/session/${sessionId}/delete/`, {
                method: 'DELETE',
                headers: { 'X-CSRFToken': csrftoken }
            });
            if (res.ok) {
                showToast('Consultation removed', 'info');
                if (currentSessionId === sessionId) {
                    await createNewSession();
                } else {
                    await loadSessions();
                }
            }
        } catch (err) {
            showToast('Failed to delete consultation', 'error');
        }
    }

    // ==========================================================================
    // Documents Management & Scope Filter
    // ==========================================================================
    async function loadDocuments() {
        try {
            const res = await fetch('/documents/');
            if (!res.ok) throw new Error('Failed to load documents');
            const data = await res.json();
            const docs = data.documents || [];

            // Update badge
            if (docCountBadge) docCountBadge.textContent = docs.length;

            // Update sidebar doc list
            if (sidebarDocList) {
                sidebarDocList.innerHTML = '';
                if (docs.length === 0) {
                    sidebarDocList.innerHTML = '<div class="doc-empty" style="font-size: 0.8rem; color: var(--text-muted); padding: 0.25rem 0;">No documents uploaded</div>';
                } else {
                    docs.forEach(d => {
                        const div = document.createElement('div');
                        div.className = 'doc-sidebar-item';
                        div.innerHTML = `
                            <span class="doc-sidebar-title" title="${d.document_name}">📄 ${d.document_name}</span>
                            <span class="status-badge status-${d.processing_status}">${d.processing_status}</span>
                        `;
                        sidebarDocList.appendChild(div);
                    });
                }
            }

            // Update Scope Dropdown
            if (docScopeSelect) {
                const currentVal = docScopeSelect.value;
                docScopeSelect.innerHTML = '<option value="all">Query Scope: All Uploaded Records</option>';
                docs.filter(d => d.processing_status === 'Completed').forEach(d => {
                    const opt = document.createElement('option');
                    opt.value = d.id;
                    opt.textContent = `Document: ${d.document_name}`;
                    docScopeSelect.appendChild(opt);
                });
                if (currentVal) docScopeSelect.value = currentVal;
            }

        } catch (err) {
            console.error('Documents load error:', err);
        }
    }

    // ==========================================================================
    // Messaging & Loading UX
    // ==========================================================================
    function startLoadingAnimation() {
        if (!loadingBubble || !loadingStageText) return;
        loadingBubble.style.display = 'flex';
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;

        const stages = [
            "Validating medical document scope...",
            "Retrieving relevant PDF chunks...",
            "Validating question context & evidence...",
            "Generating AI response...",
            "Verifying citations & document grounding..."
        ];
        let stageIdx = 0;
        loadingStageText.textContent = stages[0];

        loadingInterval = setInterval(() => {
            stageIdx = (stageIdx + 1) % stages.length;
            loadingStageText.textContent = stages[stageIdx];
        }, 1500);
    }

    function stopLoadingAnimation() {
        if (loadingInterval) clearInterval(loadingInterval);
        if (loadingBubble) loadingBubble.style.display = 'none';
    }

    function appendUserMessage(text) {
        if (emptyChatState) emptyChatState.style.display = 'none';
        if (messageStream) messageStream.style.display = 'flex';

        const turn = document.createElement('div');
        turn.className = 'message-turn user-turn';
        turn.innerHTML = `
            <div class="message-bubble">
                <div class="user-text">${escapeHtml(text)}</div>
            </div>
        `;
        messageStream.appendChild(turn);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    function getErrorBadgeLabel(errorType) {
        switch (errorType) {
            case 'service_unavailable':
                return 'AI Service Unavailable (503)';
            case 'quota_exceeded':
                return 'Usage Limit Reached (429)';
            case 'authentication_error':
                return 'Authentication Failed (401/403)';
            case 'timeout':
                return 'Request Timed Out';
            case 'network_error':
                return 'Connection Failed';
            case 'invalid_request':
                return 'Configuration Error (400)';
            case 'model_not_found':
                return 'Model Unavailable (404)';
            case 'internal_error':
                return 'Internal Error (500)';
            case 'empty_response':
                return 'Empty Generation';
            default:
                return 'AI Service Error';
        }
    }

    function appendAIErrorMessage(message, errorType, retryable, question) {
        if (!messageStream) return;

        const turn = document.createElement('div');
        turn.className = 'message-turn ai-turn ai-error-turn';

        let retryHtml = '';
        if (retryable && question) {
            retryHtml = `
                <div class="ai-error-actions">
                    <button class="btn-retry-question">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"></polyline><polyline points="23 20 23 14 17 14"></polyline><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path></svg>
                        <span>Retry Question</span>
                    </button>
                </div>
            `;
        }

        turn.innerHTML = `
            <div class="ai-avatar ai-avatar-error">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            </div>
            <div class="message-bubble error-bubble">
                <div class="ai-error-badge">${getErrorBadgeLabel(errorType)}</div>
                <div class="ai-error-text">${escapeHtml(message)}</div>
                ${retryHtml}
            </div>
        `;

        if (retryable && question) {
            const retryBtn = turn.querySelector('.btn-retry-question');
            if (retryBtn) {
                retryBtn.addEventListener('click', () => {
                    if (isProcessing) return;
                    questionInput.value = question;
                    handleSendMessage();
                });
            }
        }

        messageStream.appendChild(turn);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    function appendAIMessage(answer, citations = []) {
        if (!messageStream) return;

        const turn = document.createElement('div');
        turn.className = 'message-turn ai-turn';

        let citationsHtml = '';
        if (citations && citations.length > 0) {
            citationsHtml = `
                <div class="citations-box">
                    <div class="citations-label">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
                        <span>Verified Citations & Sources</span>
                    </div>
                    <div class="citation-cards-grid">
                        ${citations.map((c, idx) => `
                            <button class="citation-tag" data-citation='${JSON.stringify(c).replace(/'/g, "&#39;")}'>
                                <span class="citation-tag-id">${c.source_id || `[S${idx+1}]`}</span>
                                <span>${escapeHtml(c.document_name)} — Page ${c.page_number}</span>
                            </button>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        turn.innerHTML = `
            <div class="ai-avatar">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20M7 7l10 10M17 7L7 17"/></svg>
            </div>
            <div class="message-bubble">
                <div class="ai-text-content">${formatAnswerMarkdown(answer)}</div>
                ${citationsHtml}
            </div>
        `;

        // Attach citation click events
        turn.querySelectorAll('.citation-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                const data = JSON.parse(tag.getAttribute('data-citation'));
                openCitationDrawer(data);
            });
        });

        messageStream.appendChild(turn);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    function renderMessageStream(messages) {
        if (!messageStream) return;
        messageStream.innerHTML = '';

        if (!messages || messages.length === 0) {
            if (emptyChatState) emptyChatState.style.display = 'block';
            messageStream.style.display = 'none';
            return;
        }

        if (emptyChatState) emptyChatState.style.display = 'none';
        messageStream.style.display = 'flex';

        messages.forEach(msg => {
            if (msg.role === 'user') {
                appendUserMessage(msg.message);
            } else if (msg.role === 'assistant') {
                appendAIMessage(msg.message, msg.citations);
            }
        });
    }

    async function handleSendMessage() {
        if (isProcessing) return;
        const question = questionInput.value.trim();
        if (!question) return;

        const submittedQuestion = question;

        // Reset input box
        questionInput.value = '';
        autoResizeTextarea();

        appendUserMessage(submittedQuestion);
        isProcessing = true;
        sendBtn.disabled = true;
        startLoadingAnimation();

        const documentId = docScopeSelect ? docScopeSelect.value : 'all';

        try {
            const res = await fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken
                },
                body: JSON.stringify({
                    question: submittedQuestion,
                    session_id: currentSessionId,
                    document_id: documentId
                })
            });

            const data = await res.json();
            stopLoadingAnimation();

            if (data.session_id && data.session_id !== currentSessionId) {
                currentSessionId = data.session_id;
                if (activeSessionTitle && data.session_name) {
                    activeSessionTitle.textContent = data.session_name;
                }
                loadSessions();
            }

            if (data.status === 'llm_error') {
                appendAIErrorMessage(data.answer, data.error_type, data.retryable, submittedQuestion);
                showToast(getErrorBadgeLabel(data.error_type), 'error');
            } else if (data.status === 'no_documents') {
                appendAIMessage(data.answer, []);
                showToast('Please upload a PDF document first', 'error');
            } else if (data.status === 'insufficient_context') {
                appendAIMessage(data.answer, []);
                showToast('Information not found in documents', 'info');
            } else if (data.status === 'out_of_scope') {
                appendAIMessage(data.answer, []);
                showToast('Question outside document scope', 'info');
            } else {
                appendAIMessage(data.answer, data.citations || []);
            }

        } catch (err) {
            stopLoadingAnimation();
            appendAIErrorMessage(
                "Unable to connect to the AI service. Please check your connection and try again.",
                "network_error",
                true,
                submittedQuestion
            );
            showToast('Connection failed', 'error');
        } finally {
            isProcessing = false;
            sendBtn.disabled = questionInput.value.trim().length === 0;
            questionInput.focus();
        }
    }

    // ==========================================================================
    // Citation Inspection Drawer
    // ==========================================================================
    function openCitationDrawer(citationData) {
        if (!citationDrawer) return;
        if (citationSourceBadge) citationSourceBadge.textContent = citationData.source_id || '[Source]';
        if (citationDocTitle) citationDocTitle.textContent = citationData.document_name || 'Medical Document';
        if (citationPageNum) citationPageNum.textContent = `Page ${citationData.page_number || 1}`;
        if (citationSection) citationSection.textContent = citationData.section_title || 'General';
        if (citationChunkId) citationChunkId.textContent = citationData.chunk_id || 'chunk_001';
        if (citationFullText) citationFullText.textContent = citationData.chunk_text || citationData.snippet || '';

        citationDrawer.style.display = 'flex';
    }

    if (closeCitationDrawerBtn) {
        closeCitationDrawerBtn.addEventListener('click', () => {
            if (citationDrawer) citationDrawer.style.display = 'none';
        });
    }

    if (citationDrawer) {
        citationDrawer.addEventListener('click', (e) => {
            if (e.target === citationDrawer) citationDrawer.style.display = 'none';
        });
    }

    // ==========================================================================
    // PDF Upload Modal & Drag-and-Drop
    // ==========================================================================
    function openUploadModal() {
        selectedFiles = [];
        renderUploadFilesList();
        if (uploadProgressContainer) uploadProgressContainer.style.display = 'none';
        if (startUploadBtn) startUploadBtn.disabled = true;
        if (uploadModal) uploadModal.style.display = 'flex';
    }

    function closeUploadModal() {
        if (uploadModal) uploadModal.style.display = 'none';
        selectedFiles = [];
    }

    if (openUploadModalBtn) openUploadModalBtn.addEventListener('click', openUploadModal);
    if (closeUploadModalBtn) closeUploadModalBtn.addEventListener('click', closeUploadModal);
    if (cancelUploadBtn) cancelUploadBtn.addEventListener('click', closeUploadModal);

    if (pdfDropzone) {
        pdfDropzone.addEventListener('click', () => pdfFileInput.click());

        ['dragenter', 'dragover'].forEach(eventName => {
            pdfDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                pdfDropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            pdfDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                pdfDropzone.classList.remove('dragover');
            });
        });

        pdfDropzone.addEventListener('drop', (e) => {
            const files = Array.from(e.dataTransfer.files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
            if (files.length === 0) {
                showToast('Only PDF files are supported', 'error');
                return;
            }
            addFilesToUploadList(files);
        });
    }

    if (pdfFileInput) {
        pdfFileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            addFilesToUploadList(files);
        });
    }

    function addFilesToUploadList(files) {
        files.forEach(file => {
            if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
                selectedFiles.push(file);
            }
        });
        renderUploadFilesList();
    }

    function renderUploadFilesList() {
        if (!uploadFileList) return;
        uploadFileList.innerHTML = '';
        if (selectedFiles.length === 0) {
            if (startUploadBtn) startUploadBtn.disabled = true;
            return;
        }

        if (startUploadBtn) startUploadBtn.disabled = false;

        selectedFiles.forEach((f, idx) => {
            const sizeMb = (f.size / (1024 * 1024)).toFixed(2);
            const item = document.createElement('div');
            item.className = 'upload-file-item';
            item.innerHTML = `
                <div style="display: flex; align-items: center; gap: 0.5rem; overflow: hidden;">
                    <span>📄</span>
                    <strong style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 250px;">${escapeHtml(f.name)}</strong>
                    <span style="color: var(--text-muted); font-size: 0.75rem;">(${sizeMb} MB)</span>
                </div>
                <button class="icon-btn remove-file-btn" data-index="${idx}" title="Remove file">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                </button>
            `;
            item.querySelector('.remove-file-btn').addEventListener('click', () => {
                selectedFiles.splice(idx, 1);
                renderUploadFilesList();
            });
            uploadFileList.appendChild(item);
        });
    }

    if (startUploadBtn) {
        startUploadBtn.addEventListener('click', async () => {
            if (selectedFiles.length === 0) return;

            const formData = new FormData();
            selectedFiles.forEach(f => formData.append('files', f));

            startUploadBtn.disabled = true;
            if (cancelUploadBtn) cancelUploadBtn.disabled = true;
            if (uploadProgressContainer) uploadProgressContainer.style.display = 'block';
            if (uploadProgressBar) uploadProgressBar.style.width = '30%';
            if (uploadStatusText) uploadStatusText.textContent = 'Extracting and chunking PDF contents...';

            try {
                if (uploadProgressBar) uploadProgressBar.style.width = '65%';
                if (uploadStatusText) uploadStatusText.textContent = 'Generating dense embeddings and indexing to Qdrant...';

                const res = await fetch('/documents/upload/', {
                    method: 'POST',
                    headers: { 'X-CSRFToken': csrftoken },
                    body: formData
                });

                const data = await res.json();
                if (uploadProgressBar) uploadProgressBar.style.width = '100%';

                if (res.ok) {
                    showToast(`Successfully processed ${data.uploaded.length} document(s)!`, 'success');
                    closeUploadModal();
                    await loadDocuments();
                } else {
                    const errDetail = data.errors && data.errors.length > 0 ? data.errors[0].error : data.error || 'Upload failed.';
                    showToast(errDetail, 'error');
                }
            } catch (err) {
                showToast('Failed to upload documents. Please check file format and size.', 'error');
            } finally {
                startUploadBtn.disabled = false;
                if (cancelUploadBtn) cancelUploadBtn.disabled = false;
                if (uploadProgressContainer) uploadProgressContainer.style.display = 'none';
            }
        });
    }

    // ==========================================================================
    // Helpers
    // ==========================================================================
    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatAnswerMarkdown(rawText) {
        if (!rawText) return '';
        let escaped = escapeHtml(rawText);

        // Bold **text**
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic *text*
        escaped = escaped.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Inline citation pills: [S1], [S2]
        escaped = escaped.replace(/\[S(\d+)\]/g, '<span class="source-inline-badge">[S$1]</span>');

        // Headings: ### Heading or ## Heading
        escaped = escaped.replace(/^###\s+(.*$)/gim, '<h4 class="ai-heading">$1</h4>');
        escaped = escaped.replace(/^##\s+(.*$)/gim, '<h3 class="ai-heading">$1</h3>');

        // Process lists and paragraphs
        const lines = escaped.split('\n');
        let inUl = false;
        let inOl = false;
        let formattedLines = [];

        lines.forEach(line => {
            const trimmed = line.trim();
            if (!trimmed) {
                if (inUl) { formattedLines.push('</ul>'); inUl = false; }
                if (inOl) { formattedLines.push('</ol>'); inOl = false; }
                return;
            }

            // Unordered list items: - , * , •
            if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
                if (inOl) { formattedLines.push('</ol>'); inOl = false; }
                if (!inUl) { formattedLines.push('<ul>'); inUl = true; }
                const itemContent = trimmed.replace(/^[-*•]\s+/, '');
                formattedLines.push(`<li>${itemContent}</li>`);
            }
            // Ordered list items: 1. or 1)
            else if (/^\d+[\.\)]\s+/.test(trimmed)) {
                if (inUl) { formattedLines.push('</ul>'); inUl = false; }
                if (!inOl) { formattedLines.push('<ol>'); inOl = true; }
                const itemContent = trimmed.replace(/^\d+[\.\)]\s+/, '');
                formattedLines.push(`<li>${itemContent}</li>`);
            }
            // Headings
            else if (trimmed.startsWith('<h3') || trimmed.startsWith('<h4')) {
                if (inUl) { formattedLines.push('</ul>'); inUl = false; }
                if (inOl) { formattedLines.push('</ol>'); inOl = false; }
                formattedLines.push(trimmed);
            }
            // Standard paragraph
            else {
                if (inUl) { formattedLines.push('</ul>'); inUl = false; }
                if (inOl) { formattedLines.push('</ol>'); inOl = false; }
                formattedLines.push(`<p>${trimmed}</p>`);
            }
        });

        if (inUl) formattedLines.push('</ul>');
        if (inOl) formattedLines.push('</ol>');

        return formattedLines.join('');
    }

    // Initial Load
    loadSessions();
    loadDocuments();
});
