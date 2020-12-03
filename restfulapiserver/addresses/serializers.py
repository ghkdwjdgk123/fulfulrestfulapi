from rest_framework import serializers
from .models import *
import pyotp
import datetime
from django_filters import filterset

class imgserializer(serializers.ModelSerializer):

    class Meta:
        model = Face
        fields = ('image_path',)

class Userserializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = ('user_name', 'phone', 'otp_id')

    def create(self, validated_data):
        otp = User(
            phone=validated_data['phone'],
            user_name=validated_data['user_name'],
            otp_id=pyotp.random_base32(),
        )
        otp.save()

    def get_user(self, validated_data):
        user_info = User.objects.filter(phone=validated_data['phone'])
        print(user_info)
        if not user_info:
            return '승인'
        else:
            return '중복'


    def send(self, validated_data):
        user_info = User.objects.filter(user_name = validated_data.get('user_name'), phone = validated_data.get('phone'))
        totp = pyotp.TOTP(user_info.get('otp_id'), interval=180)
        return totp.now()

    def getSerial(self, validated_data):
        otp = validated_data.Post.get(user_name = validated_data['user_name'])

        otp = User(
            phone=validated_data['phone'],
            user_name=validated_data['user_name'],
            otp_id='',
        )
        return otp