from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from .models import User
from .serializers import Userserializer, imgserializer
import pyotp
import subprocess
import os
from django.http.request import QueryDict
import numpy as np
import sys
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
# Create your views here.

@api_view(['GET','POST'])
def register_1(request):
    if request.method =='POST':
        data = JSONParser().parse(request)
        # request.POST.get('title', '')
        serializer = Userserializer(data=data)
        if serializer.is_valid():
            print(serializer.data)
            tt = serializer.get_user(serializer.data)
            if tt == "승인":
                return JsonResponse({"message": "사용"})
            else:
                return JsonResponse({"message": "중복"})
        return JsonResponse(serializer.errors, status=400)

@api_view(['GET', 'POST'])
def img(request):
    if request.method == 'POST':
        data = JSONParser().parse(request)
        file = open('output1.txt','a')
        file.write(data['img'])
        file.close()
        serializer = imgserializer(data=data)


        if serializer.is_valid():
            print(serializer.data, "★★★serializer.data★★★")
            img_byte = serializer.data['img']
            file = open('output2.txt','a')
            file.write(img_byte)
            file.close()


            return JsonResponse({'message': '받음'})
        return JsonResponse({'message': "없음"})

def rechargeapplication(request):
    if request.method == 'POST':
        img =request.get()
        print(img)
        uploadpic = request.FILES['filename']
        img.picture.save("image.jpg", uploadpic)
        img.save()
        return JsonResponse({'result': 'Success'})

@api_view(['GET','POST'])
def register(request):

    if request.method == 'GET':
        phone=request.GET.get('phone', "")
        query_set=User.objects.filter(phone=phone)
        if not query_set:
            return JsonResponse({'message':'사용'})
        else:
            return JsonResponse({'message':'중복'})



    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = Userserializer(data=data)
        if serializer.is_valid():
            tt=serializer.get_user(serializer.data)
            print(serializer.data)
            if len(serializer.data['phone']) == 11:
                if tt == '승인':
                    serializer.create(data)
                    return JsonResponse({"message":"등록 성공"}, status=201)
                else:
                    return JsonResponse({"message":"중복"})
            else:
                return JsonResponse({"message":"전화번호 자리 수가 맞지 않습니다."})
        return JsonResponse(serializer.errors, status=400)


@api_view(['GET', 'POST'])
def otp_r(request):
    if request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = Userserializer(data=data)
        if serializer.is_valid():
            print(serializer.data)
            query_set = User.objects.get(user_name=serializer.data['user_name'], phone=serializer.data['phone'])
            totp = pyotp.TOTP(query_set.otp_id, interval=180)
            return JsonResponse({'otp': totp.now()})
        return JsonResponse(serializer.errors, status=400)


@api_view(['GET', 'POST'])
def otp_t(request):
    if request.method == 'GET':
        data = JSONParser().parse(request)
        serializer = Userserializer(data=data)
        if serializer.is_valid():
            query_set = User.objects.get(user_name=serializer.data['user_name'], phone=serializer.data['phone'])
            totp = pyotp.TOTP(query_set.otp_id, interval=180)
            return JsonResponse({'otp': totp.now()})
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = Userserializer(data=data)

        if serializer.is_valid():
            query_set=User.objects.get(user_name = serializer.data['user_name'], phone = serializer.data['phone'])
            totp = pyotp.TOTP(query_set.otp_id, interval=180)
            print(totp.now())
            print(serializer.data)
            if totp.now() == serializer.data['otp_id']:
                print('승인')
                return JsonResponse({"message": "승인"})
            else:
                print('거부')
                return JsonResponse({"message": "거부"})


class FileView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, req, *args, **kwargs):
        file_name=str(req.data.get('upload'))
        file_object=req.data.get('upload')
        video_path = 'C:/Users/Playdata/PycharmProjects/restfulapi/restfulapiserver/addresses/train_data/' + file_name

        with open(video_path, 'wb+') as f:
            for chunk in file_object.chunks():
                f.write(chunk)
        #
        # cmd_authorization = ['python3', 'main2.py', video_path ]
        # fd_popen = subprocess.Popen(cmd_authorization, stdout=subprocess.PIPE).stdout
        # fd_popen.read().strip()
        # fd_popen.close()

        return JsonResponse({"message": 'success'})