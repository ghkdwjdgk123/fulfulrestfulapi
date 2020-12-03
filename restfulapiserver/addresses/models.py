# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Emotion(models.Model):
    emotion_state = models.IntegerField(primary_key=True)
    emtion_image_path = models.CharField(max_length=1000, blank=True, null=True)
    phone = models.ForeignKey('User', models.CASCADE, related_name='emotion', db_column='phone', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'Emotion'


class Face(models.Model):
    user_id = models.ForeignKey('User', models.CASCADE, related_name='face')
    image_path = models.CharField(max_length=255, blank=True, null=True)
    image_register_date = models.DateTimeField(blank=True, null=True, auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'Face'


class Otp(models.Model):
    otp_id = models.OneToOneField('User', models.CASCADE, related_name='otp_1', primary_key=True)
    sent_date = models.DateTimeField(blank=True, null=True, auto_now_add=True)
    phone = models.ForeignKey('User', models.CASCADE, related_name='otp_phone', db_column='phone', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'Otp'


class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    account_number = models.CharField(max_length=30, blank=True, null=True)
    user_name = models.CharField(max_length=10, blank=True, null=True)
    created_date = models.DateTimeField(blank=True, null=True, auto_now_add=True)
    phone = models.CharField(max_length=12)
    istrain = models.IntegerField(blank=True, null=True)
    otp_id = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'User'
        unique_together = (('user_id', 'phone'),)

class Img(models.Model):
    img = models.ImageField()
