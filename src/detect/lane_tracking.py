import cv2
import numpy as np
import yaml


def warp_perspect(img, height, width):
    margin = 230
    # 관심 영역(ROI)을 이미지의 상하 높이 절반, 좌우 절반으로 지정
    src_points = np.float32([
        [0, height],             # 왼쪽 아래
        [width, height],         # 오른쪽 아래
        [width//2 -margin, height//2-100],                  # 왼쪽 위
        [width//2 +margin, height//2-100]               # 오른쪽 위
    ])
    
    
    # BEV로 변환될 대상 좌표 (직사각형)
    dst_points = np.float32([
        [0, height],             # 왼쪽 아래
        [width, height],         # 오른쪽 아래
        [0, 0],                  # 왼쪽 위
        [width, 0]               # 오른쪽 위
    ])
    
    # 투시 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 투시 변환 수행
    warped_img = cv2.warpPerspective(img, matrix, (width, height))
    

    return warped_img

def load_calibration(yaml_path = "Calibration_result.yaml"):
        # 현재 스크립트의 경로를 가져옴
        #yaml_path = "Calibration_result.yaml"  # 캘리브레이션 결과 YAML 파일 경로
        
        with open(yaml_path) as f:
            yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
        
        camera_matrix = yaml_data['Camera matrix']
        dist_str = yaml_data['Distortion coefficient']
        
        mtx_list = camera_matrix.split(',')
        list_of_mtx = []
        for i in range(3):
            sub_list = []
            for j in range(3):
                sub_list.append(float(mtx_list[3*i + j]))
            list_of_mtx.append(sub_list)
        mtx = np.array(list_of_mtx)
        
        dist_list = dist_str.split(',')
        list_of_dist = []
        for i in range(5):
            list_of_dist.append(float(dist_list[i]))
        dist = np.array([list_of_dist])
        
        return mtx, dist


def calculate_angle_difference(src, dst):
    # 두 각도 간의 최소 차이 계산
    diff = (dst - src + 720) % 360
    if diff > 180:
        diff -= 360
    return diff


def mask_green_black(frame):
    # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    H, W = frame.shape[:2]
    crop_frame = frame[H//2:, (W//4):W//2 + (W//4)]
    #crop_frame = warp_perspect(frame, H, W)
    crop_frame = cv2.GaussianBlur(crop_frame, (5, 5), 0)
    HSVframe = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV)
    HSVframe[:, :, 2] = cv2.convertScaleAbs(HSVframe[:, :, 2], alpha=0.7, beta=0)

    # 검정색 범위
    range_black = [np.array([0, 0, 0]), np.array([190, 255, 80])]
    # 초록색 범위
    range_green = [np.array([40, 50, 50]), np.array([80, 255, 255])]

    blackMask = cv2.inRange(HSVframe, range_black[0], range_black[1])
    greenMask = cv2.inRange(HSVframe, range_green[0], range_green[1])

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    blackMask = cv2.morphologyEx(blackMask, cv2.MORPH_CLOSE, kernel)
    blackMask = cv2.morphologyEx(blackMask, cv2.MORPH_OPEN, kernel)
    greenMask = cv2.morphologyEx(greenMask, cv2.MORPH_CLOSE, kernel)
    greenMask = cv2.morphologyEx(greenMask, cv2.MORPH_OPEN, kernel)

    blackOverlay = cv2.merge([np.zeros_like(blackMask), np.zeros_like(blackMask), blackMask])
    greenOverlay = cv2.merge([np.zeros_like(greenMask), greenMask, np.zeros_like(greenMask)])

    # 원본 이미지 위에 마스크 덮어쓰기
    combined = cv2.addWeighted(crop_frame, 1, blackOverlay, 0.5, 0)
    combined = cv2.addWeighted(combined, 1, greenOverlay, 0.5, 0)

    return combined, blackMask, greenMask


def find_line_center(blackMask, greenMask):
    H, W = blackMask.shape
    black_line_x = np.where(blackMask[H//2, :] > 0)[0]
    green_line_x = np.where(greenMask[H//2, :] > 0)[0]

    black_center_x = int(np.mean(black_line_x)) if len(black_line_x) > 0 else None
    green_center_x = int(np.mean(green_line_x)) if len(green_line_x) > 0 else None

    return black_center_x, green_center_x, H//3


def get_rad(center_x, center_y, frame):
    origin_center_x = frame.shape[1] // 2
    origin_center_y = frame.shape[0] - 1
    rad = np.arctan2(center_x - origin_center_x, center_y - origin_center_y)
    return rad

