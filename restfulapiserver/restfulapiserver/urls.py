from django.conf.urls import url, include
from django.contrib.auth.models import User
from rest_framework import routers, serializers, viewsets
from addresses import views

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    url(r'^register/', views.register),
    url(r'^register1/', views.register_1),
    url(r'^otp1/', views.otp_r),
    url(r'^otp2/', views.otp_t),
    url(r'^img_test/', views.FileView.as_view()),
    url('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]