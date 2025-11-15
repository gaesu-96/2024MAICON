from IPython.display import display, clear_output
import ipywidgets as widgets

from tiki.mini import TikiMini

import cv2
import numpy as np
from PIL import Image

import threading
import time
import yaml
import sys, os

import util
import drive
from detect import myYOLO



def update_frame():
    last_photo_time = time.time()
    while bot.cap.isOpened():
        ret, frame = bot.cap.read()
        frame = bot.get_frame()
        if not ret:
            print("Error: Could not read frame.")
            break
        # combined 이미지를 JPEG로 변환하여 위젯에 업데이트
        combined, _, _ = util.mask_green_black(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        #_, buffer = cv2.imencode('.jpg', combined)
        output_widget.value = buffer.tobytes()
        time.sleep(1 / 15)  # 15fps로 업데이트
        if time.time() - last_photo_time > 1.0:
            last_photo_time = time.time()
            timestamp = time.time()
            filename = f"./record/{timestamp}.jpg"
            cv2.imwrite(filename, frame)  # 이미지 저장
        

if __name__ == '__main__':
    global_stdout_backup = sys.stdout

    # 이미지 위젯, 봇 초기화
    if 'bot' in globals():
        bot.cap.release()
        del bot
    bot = drive.MaiconBot()
    output_widget = widgets.Image(format='jpeg')
    display(output_widget)

    # 스레드로 비디오 스트림 실행
    thread = threading.Thread(target=update_frame, daemon=True)
    thread.start()
    bot.detection()
    bot.tiki.log_clear()

    sys.stdout = global_stdout_backup
    bot.set_base_speed(40)
    bot.tiki.log_clear( )
    first = time.time()

    #####  A 진입  #####
    bot.track_until_green()
    bot.move(-30, -30, 1.2)
    bot.turn_left(90)
    bot.track_until_green()
    bot.move(-30, -30, 1.0)
    bot.turn_left(80)
    yolo_start = time.time()
    bot.detection('A')
    yolo_finish = time.time()
    print(f'A yolo time: {yolo_finish - yolo_start}')
    # bot.tiki.fire_cannon()
    bot.turn_left(100)
    bot.track_until_green()
    bot.move(-30, -30, 1.0)
    bot.turn_left(90)
    bot.deactivate_green(3)

    #####  B 진입  #####
    bot.track_until_green()
    bot.move(-30, -30, 0.8)
    bot.turn_left(80)
    yolo_start = time.time()
    bot.detection('B')
    yolo_finish = time.time()
    print(f'B yolo time: {yolo_finish - yolo_start}')
    # bot.tiki.fire_cannon()
    bot.turn_right(80)
    bot.deactivate_green(3)
    bot.track_until_green() # 곡선코스
    bot.move(-30, -30, 1.0)
    bot.turn_right(90)

    #####  C 진입  #####
    bot.track_until_green()
    bot.move(-30, -30, 1.0)
    bot.turn_left(75)
    yolo_start = time.time()
    bot.detection('C')
    yolo_finish = time.time()
    print(f'C yolo time: {yolo_finish - yolo_start}')
    # bot.tiki.fire_cannon()
    bot.turn_right(75)
    bot.deactivate_green(2.5)

    #####  D 진입  #####
    bot.track_until_green()
    bot.move(-30, -30, 1.0)
    bot.turn_left(70)
    yolo_start = time.time()
    bot.detection('D')
    yolo_finish = time.time()
    print(f'D yolo time: {yolo_finish - yolo_start}')
    # bot.tiki.fire_cannon()
    bot.turn_right(70)
    bot.deactivate_green(2)

    # 도착지
    bot.track_until_green()
    bot.move(-30, -30, 1.2)
    bot.turn_left(90)
    bot.track_until_green()
    bot.move(30, 30, 1.0)

    last = time.time()
    print(f'run time: {last - first}')

    bot.tiki.stop()