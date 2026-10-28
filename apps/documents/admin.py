from django.contrib import admin
from apps.documents.models import Document, DocumentChunk


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('document_name', 'processing_status', 'total_pages', 'total_chunks', 'uploaded_at', 'processed_at')
    list_filter = ('processing_status', 'uploaded_at')
    search_fields = ('document_name', 'file_hash')


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ('chunk_id', 'document', 'page_number', 'chunk_index', 'section_title')
    list_filter = ('document', 'page_number')
    search_fields = ('chunk_id', 'chunk_text', 'section_title')
