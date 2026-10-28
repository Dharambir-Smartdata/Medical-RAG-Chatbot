from django.urls import path
from apps.chat import views

app_name = 'chat'

urlpatterns = [
    path('', views.chat_home, name='chat_home'),
    path('chat/', views.submit_question, name='submit_question'),
    path('chat/history/', views.get_chat_history, name='get_chat_history'),
    path('chat/new/', views.create_new_session, name='create_new_session'),
    path('chat/session/<uuid:session_id>/', views.get_session_messages, name='get_session_messages'),
    path('chat/session/<uuid:session_id>/delete/', views.delete_session, name='delete_session'),
    path('chat/model-status/', views.check_model_status, name='model_status'),
]
