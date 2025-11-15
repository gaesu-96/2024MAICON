import os
import cv2
import random
import shutil
import numpy as np
from deprecated import deprecated
from datetime import datetime


def read_yolo_labels(path_label):
    with open(path_label, "r", encoding="utf-8") as file:
        _labels = []
        for line in file:
            _labels.append([float(number) for number in line.split()])

    return _labels


def save_yolo_labels(file_path, labels):
    with open(file_path, 'w') as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")


def load_bounding_box(label, h, w):
    """
    yolo label 정보를 받아 bounding box 위치를 load
    :param label: 객체 정보 리스트 (YOLO 형식 - [class, x_center, y_center, width, height])
    :param h: height of image
    :param w: width of image
    :return: bounding box 좌표
    """
    class_id, x_center, y_center, box_w, box_h = label
    x1 = int((x_center - box_w / 2) * w)
    y1 = int((y_center - box_h / 2) * h)
    x2 = int((x_center + box_w / 2) * w)
    y2 = int((y_center + box_h / 2) * h)

    return x1, y1, x2, y2


def random_object_masking(img, labels, mask_color=(0, 0, 0), mask_ratio=0.2):
    """
    bounding box 내의 이미지를 랜덤으로 masking하는 함수
    :param img: input image
    :param labels: 객체 정보 리스트 (YOLO 형식 - [class, x_center, y_center, width, height])
    :param mask_color: 마스킹할 색 (기본은 검정색)
    :param mask_ratio: 마스크 비율 (0 ~ 1 사이의 값, 0.5는 바운딩 박스의 절반만 마스킹)
    :return: 마스킹이 적용된 이미지
    """
    h, w, _ = img.shape
    for label in labels:
        class_id, x_center, y_center, box_w, box_h = label
        x1, y1, x2, y2 = load_bounding_box(label, h, w)

        # 마스킹 비율에 따라 마스크 영역 크기 조절
        mask_w = int(box_w * mask_ratio * w)
        mask_h = int(box_h * mask_ratio * h)

        # 마스킹할 영역을 바운딩 박스 내에서 랜덤하게 설정
        mask_x1 = np.random.randint(x1, x2 - mask_w)
        mask_y1 = np.random.randint(y1, y2 - mask_h)

        # 마스크 영역을 이미지에 적용
        img[mask_y1:mask_y1 + mask_h, mask_x1:mask_x1 + mask_w] = mask_color

    return img


@deprecated
def object_paste_augmentation(img, labels, paste_count=3):
    """
    객체 부분만을 잘라낸 후, 다른 위치에 붙여넣어 증강
    :param img: input image
    :param labels: 객체 정보 리스트 (YOLO 형식 - [class, x_center, y_center, width, height])
    :param paste_count: 붙여넣을 개수
    :return: 새로 paste되어 생성된 이미지
    """
    h, w, _ = img.shape
    augmented_labels = []
    for label in labels:
        class_id, x_center, y_center, box_w, box_h = label
        x1, y1, x2, y2 = load_bounding_box(label, h, w)

        # 객체 잘라내기
        obj = img[y1:y2, x1:x2].copy()

        for _ in range(paste_count):
            px = np.random.randint(0, w - (x2 - x1))
            py = np.random.randint(0, h - (y2 - y1))

            # 새로운 위치에 객체 붙여넣기
            img[py:py + (y2 - y1), px:px + (x2 - x1)] = obj

            # 새로운 바운딩 박스 좌표 추가
            new_x_center = (px + (px + x2 - x1)) / (2 * w)
            new_y_center = (py + (py + y2 - y1)) / (2 * h)
            new_box_w = box_w
            new_box_h = box_h
            augmented_labels.append([class_id, new_x_center, new_y_center, new_box_w, new_box_h])

    # 기존 라벨과 증강된 라벨 모두 반환
    return img, labels + augmented_labels


def insert_image(img_original,
                 img_adding,
                 scale_img_adding=0.8):
    """

    :param img_original: 배경이 될 이미지
    :param img_adding: 추가하려는 이미지
    :param scale_img_adding: 추가하려는 이미지의 크기를 축소
    :return: 합성된 이미지와 새로 생성된 라벨
    """
    h_original, w_original, _ = img_original.shape
    h_adding, w_adding, _ = img_adding.shape

    # 랜덤 좌우 반전 결정
    flip = random.choice([True, False])
    if flip:
        img_adding = cv2.flip(img_adding, 1)

    # 추가하려는 이미지 크기 조정
    new_w = int(w_adding * scale_img_adding)
    new_h = int(h_adding * (new_w / w_adding))
    img_adding = cv2.resize(img_adding, (new_w, new_h))
    h_adding, w_adding, _ = img_adding.shape

    # 알파 채널 분리
    alpha_channel = img_adding[:, :, 3]
    rgb_channel = img_adding[:, :, :3]

    # 알파 채널 0-1 범위로 정규화
    alpha_normalized = alpha_channel / 255.0

    # 삽입 위치 결정
    x = np.random.randint(0, w_original - w_adding)
    y = np.random.randint(0, h_original - h_adding)

    # # 이미지 삽입
    # img_original[y:y + h_adding, x:x + w_adding] = img_adding

    # 이미지 합성
    for c in range(0, 3):
        img_original[y:y + h_adding, x:x + w_adding, c] = (alpha_normalized * rgb_channel[:, :, c] + (1 - alpha_normalized) * img_original[y:y + h_adding, x:x + w_adding, c])

    # 바운딩 박스 계산 (YOLO 형식)
    x_center = (x + w_adding / 2) / w_original
    y_center = (y + h_adding / 2) / h_original
    box_width = w_adding / w_original
    box_height = h_adding / h_original
    bbox = [x_center, y_center, box_width, box_height]

    return img_original, bbox


def augment_random(img_background,
                   labels_background,
                   folder_soldier,
                   folder_tank,
                   num_images_character=3,
                   scale_character=0.6,
                   scale_tank=0.6):
    """
    이미지 A에 폴더 C의 이미지를 삽입한 뒤, 폴더 D에서 0장 또는 1장을 추가로 삽입
    :param img_background: 원본 이미지 (H, W, C)
    :param labels_background: 원본 이미지의 YOLO 라벨 ([class, x_center, y_center, width, height])
    :param folder_soldier: 군인 이미지 폴더
    :param folder_tank: 탱크 이미지 폴더
    :param num_images_character: 삽입할 군인 이미지 수
    :param scale_character: 삽입할 군인 이미지 크기 축소 비율
    :param scale_tank: 삽입할 탱크 이미지 크기 축소 비율
    :return: 수정된 이미지와 업데이트된 YOLO 라벨
    """
    h_background, w_background, _ = img_background.shape
    updated_labels = labels_background.copy()

    # tank 이미지 0장 또는 1장 삽입
    if random.random() > 0.5:  # 50% 확률로 삽입
        image_files_tank = [os.path.join(folder_tank, f) for f in os.listdir(folder_tank)]
        img_path_tank = random.choice(image_files_tank)
        label = int(os.path.splitext(os.path.basename(img_path_tank))[0].split('_')[0])
        img_tank = cv2.imread(img_path_tank, cv2.IMREAD_UNCHANGED)

        img_background, new_bbox = insert_image(img_original=img_background,
                                                img_adding=img_tank,
                                                scale_img_adding=scale_tank)

        updated_labels.append([label]+new_bbox)

    # 병사 이미지 삽입
    image_files_soldier = [os.path.join(folder_soldier, f) for f in os.listdir(folder_soldier)]
    selected_images_soldier = random.choices(image_files_soldier, k=num_images_character)

    for idx, img_path_s in enumerate(selected_images_soldier):
        label = int(os.path.splitext(os.path.basename(img_path_s))[0].split('_')[0])
        img_s = cv2.imread(img_path_s, cv2.IMREAD_UNCHANGED)

        img_background, new_bbox = insert_image(img_original=img_background,
                                                img_adding=img_s,
                                                scale_img_adding=scale_character)

        updated_labels.append([label]+new_bbox)

    return img_background, updated_labels


if __name__ == "__main__":
    # 이미지 합성 예제
    folder_path = "../dataset/images"
    folder_path_soldier = "../dataset/soldier"
    folder_path_tank = "../dataset/tank"
    n_image = len([file for file in os.listdir(folder_path)])

    for i in range(1, n_image+1):
        # load image and labels
        p_img = f'{folder_path}/{i}.jpg'
        p_label = f'../dataset/labels/{i}.txt'

        test_img = cv2.imread(p_img)
        test_labels = read_yolo_labels(p_label)

        new_img, new_labels = augment_random(img_background=test_img,
                                             labels_background=test_labels,
                                             folder_soldier=folder_path_soldier,
                                             folder_tank=folder_path_tank,
                                             num_images_character=3,
                                             scale_character=0.8,
                                             scale_tank=0.8)

        # 이미지 및 라벨 저장
        current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        save_path = f'../dataset/aug_images/{current_time}.jpg'
        cv2.imwrite(save_path, new_img)

        save_path = f'../dataset/aug_labels/{current_time}.txt'
        save_yolo_labels(save_path, new_labels)

    
    # # masking 이용 예제
    # folder_path = "../dataset/images"
    # n_image = len([file for file in os.listdir(folder_path) if file.lower().endswith('.jpg')])
    #
    # for i in range(1, n_image+1):
    #     # load image and labels
    #     p_img = f'{folder_path}/{i}.jpg'
    #     p_label = f'../dataset/labels/{i}.txt'
    #
    #     test_img = cv2.imread(p_img)
    #     test_labels = read_label(p_label)
    #
    #     # masking 이용 augmentation
    #     augmented_mask_img = random_object_masking(test_img, test_labels)
    #
    #     # 이미지 및 라벨 저장
    #     save_path = f'../dataset/aug_images/{i}.jpg'
    #     cv2.imwrite(save_path, augmented_mask_img)
    #     shutil.copy(p_label, f'../dataset/aug_labels/{i}.txt')
