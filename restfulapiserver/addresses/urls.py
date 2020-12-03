from django.urls import path, include
from views import register, otp_t

urlpatterns = [
    path("User/", register),
    path("User/", otp_t)
]