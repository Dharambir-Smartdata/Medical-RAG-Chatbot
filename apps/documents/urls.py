from django.urls import path
from apps.documents import views

app_name = 'documents'

urlpatterns = [
    path('', views.list_documents, name='list_documents'),
    path('manage/', views.documents_page, name='documents_page'),
    path('upload/', views.upload_documents, name='upload_documents'),
    path('<uuid:document_id>/', views.document_detail, name='document_detail'),
    path('<uuid:document_id>/delete/', views.delete_document, name='delete_document'),
]
