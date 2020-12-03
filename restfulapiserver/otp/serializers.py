from rest_framework import serializers
from .models import *


class Userserializer(serializers.ModelSerializer):
    otp = serializers.StringRelatedField(many=True)

    class Meta:
        model = User
        fields = ['phone', 'user_name', 'otp']