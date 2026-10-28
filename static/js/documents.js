/**
 * Medical AI Assistant — Documents Management Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const documentsTableBody = document.getElementById('documentsTableBody');
    const openUploadBtnFromPage = document.getElementById('openUploadBtnFromPage');
    const chunksModal = document.getElementById('chunksModal');
    const closeChunksModalBtn = document.getElementById('closeChunksModalBtn');
    const chunksModalDocTitle = document.getElementById('chunksModalDocTitle');
    const chunksInspectorList = document.getElementById('chunksInspectorList');

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

    if (openUploadBtnFromPage) {
        openUploadBtnFromPage.addEventListener('click', () => {
            const openModal = document.getElementById('openUploadModalBtn');
            if (openModal) openModal.click();
        });
    }

    async function loadDocumentsTable() {
        if (!documentsTableBody) return;
        try {
            const res = await fetch('/documents/');
            const data = await res.json();
            const docs = data.documents || [];

            if (docs.length === 0) {
                documentsTableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center py-4 text-muted">
                            No medical records found. Upload a PDF document to begin.
                        </td>
                    </tr>
                `;
                return;
            }

            documentsTableBody.innerHTML = '';
            docs.forEach(doc => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span>📄</span>
                            <strong>${escapeHtml(doc.document_name)}</strong>
                        </div>
                        ${doc.error_message ? `<div style="color: var(--status-danger); font-size: 0.75rem; margin-top: 0.2rem;">${escapeHtml(doc.error_message)}</div>` : ''}
                    </td>
                    <td><span class="status-badge status-${doc.processing_status}">${doc.processing_status}</span></td>
                    <td>${doc.total_pages}</td>
                    <td>${doc.total_chunks}</td>
                    <td><small>${doc.uploaded_at || '—'}</small></td>
                    <td><small>${doc.processed_at ? doc.processed_at.substring(0, 16) : '—'}</small></td>
                    <td class="text-right">
                        <button class="btn btn-secondary btn-sm view-chunks-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.document_name)}" title="Inspect Chunks">
                            Chunks
                        </button>
                        <button class="btn btn-secondary btn-sm delete-doc-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.document_name)}" style="color: var(--status-danger); margin-left: 0.35rem;" title="Delete Document">
                            Delete
                        </button>
                    </td>
                `;

                tr.querySelector('.view-chunks-btn').addEventListener('click', () => {
                    inspectChunks(doc.id, doc.document_name);
                });

                tr.querySelector('.delete-doc-btn').addEventListener('click', () => {
                    deleteDocument(doc.id, doc.document_name);
                });

                documentsTableBody.appendChild(tr);
            });
        } catch (err) {
            console.error(err);
            documentsTableBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">Error loading documents.</td></tr>';
        }
    }

    async function inspectChunks(docId, docName) {
        if (!chunksModal) return;
        chunksModalDocTitle.textContent = `Chunks: ${docName}`;
        chunksInspectorList.innerHTML = '<div class="text-center py-4">Fetching chunk records...</div>';
        chunksModal.style.display = 'flex';

        try {
            const res = await fetch(`/documents/${docId}/`);
            const data = await res.json();
            const chunks = data.chunks || [];

            if (chunks.length === 0) {
                chunksInspectorList.innerHTML = '<div class="text-muted text-center py-4">No chunks indexed for this document.</div>';
                return;
            }

            chunksInspectorList.innerHTML = chunks.map(c => `
                <div style="background: var(--bg-subtle); padding: 0.85rem; border-radius: var(--radius-sm); margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.35rem;">
                        <strong>[ID: ${c.chunk_id}] — Page ${c.page_number}</strong>
                        <span>Section: ${escapeHtml(c.section_title)}</span>
                    </div>
                    <div style="font-size: 0.825rem; line-height: 1.5; color: var(--text-secondary); white-space: pre-wrap;">${escapeHtml(c.chunk_text)}</div>
                </div>
            `).join('');

        } catch (err) {
            chunksInspectorList.innerHTML = '<div class="text-danger text-center">Failed to load chunks.</div>';
        }
    }

    if (closeChunksModalBtn) {
        closeChunksModalBtn.addEventListener('click', () => {
            if (chunksModal) chunksModal.style.display = 'none';
        });
    }

    if (chunksModal) {
        chunksModal.addEventListener('click', (e) => {
            if (e.target === chunksModal) chunksModal.style.display = 'none';
        });
    }

    async function deleteDocument(docId, docName) {
        if (!confirm(`Are you sure you want to delete '${docName}'? This will remove its vectors and chunks.`)) return;

        try {
            const res = await fetch(`/documents/${docId}/delete/`, {
                method: 'DELETE',
                headers: { 'X-CSRFToken': csrftoken }
            });
            const data = await res.json();
            if (res.ok) {
                showToast(`Deleted ${docName}`, 'info');
                loadDocumentsTable();
            } else {
                showToast('Failed to delete document', 'error');
            }
        } catch (err) {
            showToast('Delete operation failed', 'error');
        }
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    loadDocumentsTable();
});
