from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt
from .serializers import Userserializer
import pyotp


# Create your views here.


@csrf_exempt
def OTP_list(request):
    if request.method =='GET':
        query_set = OTP.objects.filter(name=request)
        serializer = Userserializer(query_set, many=True)
        pyotp.totp.now(serializer['otp_name'])
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = Userserializer(data=data)
        serializer.data['otp'] = pyotp.random_base32()
        print(serializer)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)