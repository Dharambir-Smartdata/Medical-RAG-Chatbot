import uuid
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class ChatSession(models.Model):
    """
    Maintains distinct conversation sessions for users.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='chat_sessions')
    session_name = models.CharField(max_length=255, default='Medical Consultation')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.session_name} ({self.id})"


class ChatMessage(models.Model):
    """
    Stores conversational turns with user messages, assistant responses, and citations.
    """
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    message = models.TextField()
    citations = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role.upper()}] {self.message[:40]}"
