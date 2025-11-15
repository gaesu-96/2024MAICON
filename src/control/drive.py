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

from detect import lane_tracking as util
from detection import myYOLO


class MaiconBot:
    def __init__(self):
        print('[+] Initialize MaiconBot')
        self.tiki = TikiMini()
        self.tiki.set_motor_mode(self.tiki.MOTOR_MODE_PID)
        self.cap = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=15/1, format=NV12 ! "
            "nvvidconv flip-method=2 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=True"
        )
        self.mtx, self.dist = util.load_calibration("Calibration_result.yaml")
        self.controller = myYOLO()
        self.is_green_sensitive = True
        self.green_deactive_limit = time.time()
        self.drive_base_speed = 40
        self.drive_max_offset = self.drive_base_speed - 15

        # 카메라 체크
        ret, frame = self.cap.read()
        if ret:
            print('[+] Camera Check...Ok')
        else:
            print('[-] Camera Check...FAIL')
            self.cap.release()
        self.tiki.log_clear()
        # YOLO 예열 - 오류남
        #self.prepare_detect()

    
    def __del__(self):
        print('[+] Release Camera')
        self.cap.release()

    
    # 베이스 속도 및 그에 따른 max offset 설정
    def set_base_speed(self, base_speed):
        print(f'[+] Set Base Speed as {base_speed}')
        # 속도 30일때, offset 15
        # 40일때 25
        self.drive_base_speed = base_speed
        if base_speed >= 50:
            self.drive_max_offset = self.drive_base_speed - 8
        elif base_speed >= 45:
            self.drive_max_offset = self.drive_base_speed - 12
        else:
            self.drive_max_offset = self.drive_base_speed - 18

    
    # 카메라에 비치는 시야 전달 + Calibration
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print('[-] ERR: cannot read camera')

        H, W = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (W, H), 0)
        undistorted_frame = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)

        return undistorted_frame


    # 왼쪽 v_left, 오른쪽 v_right의 속도로 t초간 주행
    def move(self, v_left, v_right, t):
        print(f'[+] move with ({v_left}, {v_right}) for {t} sec')
        self.tiki.set_motor_power(self.tiki.MOTOR_LEFT, v_left)
        self.tiki.set_motor_power(self.tiki.MOTOR_RIGHT, v_right)
        time.sleep(t)
        self.tiki.stop()


    # Line Tracking 간 이동용
    def go(self, rad):
        # rad 값이 클수록 회전 각도를 줄이기 위해 offset 반대로 조정
        speed_offset = (3.14 - abs(rad)) * self.drive_max_offset
        # 오른쪽과 왼쪽 바퀴 속도 설정
        right_speed = self.drive_base_speed - (speed_offset if rad > 0 else -1*speed_offset)
        left_speed = self.drive_base_speed + (speed_offset if rad > 0 else -1*speed_offset)
    
        # 속도 값 제한
        right_speed = np.clip(right_speed, 0, 100)
        left_speed = np.clip(left_speed, 0, 100)
    
        # 모터에 속도 전달
        self.tiki.set_motor_power(self.tiki.MOTOR_RIGHT, int(right_speed))
        self.tiki.set_motor_power(self.tiki.MOTOR_LEFT, int(left_speed))


    # 입력 시간(sec)동안 녹색 마커 감지 비활성화
    def deactivate_green(self, sec):
        self.is_green_sensitive = False
        self.green_deactive_limit = time.time() + sec

    
    # 녹색 마커 전까지 일반적인 Line Tracking
    def track_until_green(self):
        prev_rad = 3.14
        while True:
            if (not self.is_green_sensitive) and (time.time() > self.green_deactive_limit):
                self.is_green_sensitive = True
 
            frame = self.get_frame()
            combined, blackMask, greenMask = util.mask_green_black(frame)
            black_center_x, green_center_x, center_y = util.find_line_center(blackMask, greenMask)

            if black_center_x is not None:
                cv2.circle(combined, (black_center_x, center_y), 5, (0, 255, 0), 3)
                rad = util.get_rad(black_center_x, center_y, combined)
                prev_rad = rad
                self.go(rad)
            else:
                self.go(prev_rad)
                
            if (self.is_green_sensitive) and (green_center_x is not None):
                cv2.circle(combined, (green_center_x, center_y), 5, (255, 0, 0), 3)
                self.tiki.stop()
                print('[+] Green Marker Detected')
                break
    

    # 주어진 각도(도 기준) 좌회전
    def turn_left(self, target_degree):
        print(f'[+] Turn left : {target_degree} deg')
        #불필요 IMU 출력 방지용
        backupstdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        target_degree += 1 # 오차 보정치
        sleep_time = (target_degree - 10) * 0.01 
        init_imu = self.tiki.get_imu()[0]  # 초기 IMU 각도
        self.tiki.counter_clockwise(20)  # 반시계 방향 회전
        time.sleep(sleep_time)
        
        self.tiki.stop()
        cur_imu = self.tiki.get_imu()[0]  # 회전 후 IMU 각도
        
        target_degree *= -1
        turn_degree = util.calculate_angle_difference(init_imu, cur_imu)
        # 목표 각도 차이와 비교하여 조정
        while abs(util.calculate_angle_difference(turn_degree, target_degree)) > 5:
            prev_imu = cur_imu
            if util.calculate_angle_difference(turn_degree, target_degree) < 0:  # 회전 각도가 부족한 경우
                self.tiki.counter_clockwise(10)
            else:  # 회전 각도가 초과한 경우
                self.tiki.clockwise(10)
            time.sleep(0.05)
            try:
                cur_imu = self.tiki.get_imu()[0]
            except:
                cur_imu = prev_imu
            turn_degree = util.calculate_angle_difference(init_imu, cur_imu)
        self.tiki.stop()
        sys.stdout = backupstdout

    
    # 주어진 각도(도 기준) 우회전
    def turn_right(self, target_degree):
        print(f'[+] Turn right : {target_degree} deg')
        backupstdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        target_degree += 1 # 오차 보정치
        sleep_time = (target_degree - 10) * 0.01 

        init_imu = self.tiki.get_imu()[0]
        self.tiki.clockwise(20)  # 반시계 방향 회전
        time.sleep(sleep_time)
        
        self.tiki.stop()
        cur_imu = self.tiki.get_imu()[0]

        turn_degree = util.calculate_angle_difference(init_imu, cur_imu)
        
        while abs(util.calculate_angle_difference(turn_degree, target_degree)) > 5:
            prev_imu = cur_imu
            if util.calculate_angle_difference(turn_degree, target_degree) < 0:
                self.tiki.counter_clockwise(10)
            else:
                self.tiki.clockwise(10)
            time.sleep(0.05)
            try:
                cur_imu = self.tiki.get_imu()[0]
            except:
                cur_imu = prev_imu
            turn_degree = util.calculate_angle_difference(init_imu, cur_imu)
        self.tiki.stop()
        sys.stdout = backupstdout
    

    # 미션 수행
    def detection(self, stage = 'A'):
        self.tiki.counter_clockwise(1)
        self.tiki.stop()
        time.sleep(0.5)
        frame = self.get_frame()
        cv2.imwrite(f'./photo/{time.time()}.jpg', frame)  # 이미지 저장
        detection_result = self.controller.detect(frame)
        print(f'[+] Detected Result #1 : AF-{detection_result["AF"]} / EF-{detection_result["EF"]} / AT-{detection_result["AT"]} / ET-{detection_result["ET"]}')
        
        if detection_result['ET'] != 0:
            self.tiki.fire_cannon()
        
        answer = (detection_result["AF"], detection_result["EF"])
        answer_str = f'{stage}: AF-{answer[0]} EF-{answer[1]}'
        self.tiki.log(answer_str)