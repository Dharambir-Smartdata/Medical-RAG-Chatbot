import uuid
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class Document(models.Model):
    """
    Represents an uploaded medical PDF document.
    """
    STATUS_CHOICES = [
        ('Uploaded', 'Uploaded'),
        ('Processing', 'Processing'),
        ('Completed', 'Completed'),
        ('Failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='documents')
    document_name = models.CharField(max_length=255)
    file_path = models.FileField(upload_to='medical_pdfs/')
    file_hash = models.CharField(max_length=64, db_index=True)
    total_pages = models.PositiveIntegerField(default=0)
    total_chunks = models.PositiveIntegerField(default=0)
    processing_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Uploaded')
    error_message = models.TextField(blank=True, default='')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.document_name} ({self.processing_status})"


class DocumentChunk(models.Model):
    """
    Represents a chunk derived from an uploaded medical document,
    persisting text and metadata for BM25 and citations.
    """
    chunk_id = models.CharField(max_length=128, unique=True, db_index=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.PositiveIntegerField(default=0)
    page_number = models.PositiveIntegerField(default=1)
    all_pages = models.JSONField(default=list, blank=True)
    chunk_text = models.TextField()
    section_title = models.CharField(max_length=255, blank=True, default='General')
    metadata_json = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'page_number']),
        ]

    def __str__(self):
        return f"{self.chunk_id} (Page {self.page_number})"
