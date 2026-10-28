from django.contrib import admin
from apps.chat.models import ChatSession, ChatMessage


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ('role', 'message', 'citations', 'created_at')


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_name', 'user', 'created_at', 'updated_at')
    search_fields = ('session_name',)
    inlines = [ChatMessageInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'role', 'created_at')
    list_filter = ('role', 'created_at')
    search_fields = ('message',)
