import os
import cv2
import glob
import numpy as np
import time
import threading

import multiprocessing
from pathos import multiprocessing as mp
from multiprocessing import Pool
from kalman_filter import KF


class Multi_Thread(threading.Thread):
    def __init__(self, func, args=()):
        super(Multi_Thread, self).__init__()
        self.func = func
        self.args = args
        self.result = []
        self._running = True

    def input(self, args=()):
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def terminate(self):
        self._running = False

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


class Satellite_dataset(object):
    def __init__(self, root_dir=''):
        self.root_dir = root_dir
        self.left_dir = os.path.join(root_dir, 'left')
        self.right_dir = os.path.join(root_dir, 'right')
        if os.path.exists(self.left_dir) and os.path.exists(self.right_dir):
            print('=> The left and right image directories have been found.')
        else:
            print('The left and right image directories can not be found!')
            exit(0)

        # acquire all of the image paths
        self.left_img_files = sorted(glob.glob(os.path.join(self.left_dir, '*.BMP')))
        self.right_img_files = sorted(glob.glob(os.path.join(self.right_dir, '*.BMP')))

        left_img_num = len(self.left_img_files)
        right_img_num = len(self.right_img_files)
        if left_img_num != right_img_num:
            print('the number of left and right images are not the same.')
            exit(0)
        else:
            self.img_num = left_img_num

        self.img_size = (2048, 2048)

    def format_transformation(self, delimiter='\\'):
        for i in range(self.img_num):
            left_img_path = self.left_img_files[i]
            right_img_path = self.right_img_files[i]

            split_left_img_path = left_img_path.split(delimiter)
            left_img_name = int(split_left_img_path[-1].split('.')[0])
            left_img_name = '{:04d}'.format(left_img_name)
            split_left_img_path[-1] = left_img_name + '.BMP'
            modified_left_img_path = delimiter.join(split_left_img_path)
            print(left_img_name)
            print(modified_left_img_path)
            os.rename(left_img_path, modified_left_img_path)

            split_right_img_path = right_img_path.split(delimiter)
            right_img_name = int(split_right_img_path[-1].split('.')[0])
            right_img_name = '{:04d}'.format(right_img_name)
            split_right_img_path[-1] = right_img_name + '.BMP'
            modified_right_img_path = delimiter.join(split_right_img_path)
            print(right_img_name)
            print(modified_right_img_path)
            os.rename(right_img_path, modified_right_img_path)

    def dataset_display(self):
        cv2.namedWindow('dataset displaying', cv2.WINDOW_GUI_EXPANDED)
        for i in range(self.img_num):
            left_img = cv2.imread(self.left_img_files[i], 1)
            right_img = cv2.imread(self.right_img_files[i], 1)
            display_result = np.hstack((left_img, right_img))
            cv2.imshow('dataset displaying', display_result)
            key = cv2.waitKey(1)
            if key == 27:
                return

    def __getitem__(self, idx):
        left_img_path = self.left_img_files[idx]
        right_img_path = self.right_img_files[idx]
        left_img = cv2.imread(left_img_path, 0)
        right_img = cv2.imread(right_img_path, 0)

        return left_img, right_img


class Fast_detector(object):
    def __init__(self, img_size=()):
        self.img_size = img_size
        self.roi_delta_x = int(img_size[0] * 0.1)
        self.roi_delta_y = int(img_size[1] * 0.1)
        self.threshold_offset = 10
        self.roi_offset = 50
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        self.corners = []
        self.pre_corners = None
        self.pre_rotate_rect = None
        self.pre_rotate_angle = 0
        self.kf = KF(10)
        pass

    def detection(self, img):
        img_size = img.shape[0:2]
        if img_size != self.img_size:
            print('the shape of input image {} should be {}'.format(img_size, self.img_size))
            return None

        # image pre-processing
        blur_img = cv2.GaussianBlur(img, (3, 3), 1, 0.5)
        coarse_roi_img = blur_img[self.roi_delta_x:img_size[0]-self.roi_delta_x,
                         self.roi_delta_y:img_size[1]-self.roi_delta_y]

        # image segmentation
        threshold, binary_img = cv2.threshold(coarse_roi_img, 0, 255, cv2.THRESH_OTSU)
        roi_h, roi_w = binary_img.shape[0:2]
        resize_img = cv2.resize(binary_img, (roi_w >> 3, roi_h >> 3), cv2.INTER_CUBIC)
        upper_bound, bottom_bound = self.horizontal_projecting(resize_img)
        left_bound, right_bound = self.vertical_projecting(resize_img)
        x1 = (left_bound << 3) + self.roi_delta_x - self.roi_offset
        x2 = (right_bound << 3) + self.roi_delta_x + self.roi_offset
        y1 = (upper_bound << 3) + self.roi_delta_y - self.roi_offset
        y2 = (bottom_bound << 3) + self.roi_delta_y + self.roi_offset
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > img_size[0]:
            x2 = img_size[0]
        if y2 > img_size[1]:
            y2 = img_size[1]
        # print('x1:{}, x2:{}'.format(x1, x2))
        # print('y1:{}, y2:{}'.format(y1, y2))
        fine_roi_img = blur_img[y1:y2, x1:x2]

        # object detection
        roi_h, roi_w = fine_roi_img.shape[0:2]
        resize_img = cv2.resize(fine_roi_img, (roi_w >> 2, roi_h >> 2), cv2.INTER_CUBIC)
        _, binary_img = cv2.threshold(resize_img, 0, 255, cv2.THRESH_OTSU)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, self.morph_kernel)
        contours, hierachy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print('contours number: {}'.format(len(contours)))
        if len(contours) > 1:
            print('=> Failed to detect object, retry again.')
            return None
        # regression
        rotate_rect = cv2.minAreaRect(contours[0])
        print(rotate_rect)


        # original_corners = corners * (2 ** 2)
        # original_corners[:, 0] = original_corners[:, 0] + x1
        # original_corners[:, 1] = original_corners[:, 1] + y1
        # cv2.drawContours(blur_img, [np.int0(original_corners)], 0, (255, 255, 0), 2)

        # center_point = np.mean(corners, axis=0)

        if self.pre_corners is None:
            rotation_angle = 0
            self.kf.initialize(rotate_rect[0][0], rotate_rect[0][0], rotation_angle,
                               rotate_rect[1][0], rotate_rect[1])
              
            # self.pre_corners = corners
            # self.pre_center = center_point
            self.pre_rotate_rect = rotate_rect
            corners = cv2.boxPoints(rotate_rect)
            original_corners = corners * (2 ** 2)
            original_corners[:, 0] = original_corners[:, 0] + x1
            original_corners[:, 1] = original_corners[:, 1] + y1
            center_point = np.mean(corners, axis=0)
            return original_corners, center_point, rotation_angle
        else:
            original_corners = self.corners_matching(self.pre_corners, original_corners)
            rotation_angle = self.rotation_angle_calc(self.pre_center, self.pre_corners,
                                                      center_point, original_corners)
            print('rotation angle: {}'.format(rotation_angle))
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # cv2.drawContours(img, [np.int0(self.pre_corners)], 0, (255, 0, 255), 1)
            # cv2.drawContours(img, [np.int0(original_corners)], 0, (255, 255, 0), 1)
            # cv2.circle(img, (center_point[0], center_point[1]), radius=2, color=(0, 0, 255), thickness=2)
            # cv2.circle(img, (self.pre_center[0], self.pre_center[1]), radius=2, color=(0, 255, 255), thickness=2)
            # cv2.namedWindow('detection displaying', cv2.WINDOW_GUI_EXPANDED)
            # cv2.imshow('detection displaying', img)

            self.pre_corners = original_corners
            self.pre_center = center_point

            # cv2.namedWindow('roi image', cv2.WINDOW_GUI_EXPANDED)
            # cv2.imshow('roi image', blur_img)
            # cv2.waitKey(2)
            return original_corners, center_point, rotation_angle

    def horizontal_projecting(self, img):
        w, h = img.shape[0:2]
        point = np.zeros(h, dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if img[y][x] > 0:
                    point[y] = point[y] + 1
        point = cv2.normalize(point, 0, 100, cv2.NORM_INF).squeeze()
        # print(point)
        # scale = 4
        # hist_img = np.zeros((h * scale, 100, 3), dtype=np.uint8)
        flag = 0
        for i in range(len(point) - 5):
            binValue = point[i]
            if flag == 0 and binValue > 1 and point[i + 3] > 1 and point[i + 5] > 1:
                upper_bound = i
                flag = 1
                continue
            if flag == 1 and binValue <= 1 and point[i + 3] <= 1 and point[i + 5] <= 1:
                bottom_bound = i
                flag = 2
                continue
            if flag == 2 and binValue > 10 and point[i + 3] > 10 and point[i + 5] > 10:
                flag = 1
            # cv2.rectangle(hist_img, pt1=(0, i * scale), pt2=(binValue, (i + 1) * scale),
            #               color=(255, 255, 0), thickness=2)
        # print('upper bound: {}, bottom bound: {}'.format(upper_bound, bottom_bound))
        # cv2.namedWindow('horizontal projecting', cv2.WINDOW_GUI_EXPANDED)
        # cv2.imshow('horizontal projecting', hist_img)
        return upper_bound, bottom_bound

    def vertical_projecting(self, img):
        w, h = img.shape[0:2]
        point = np.zeros(h, dtype=np.uint8)
        for x in range(w):
            for y in range(h):
                if img[y][x] > 0:
                    point[x] = point[x] + 1
        point = cv2.normalize(point, 0, 100, cv2.NORM_INF).squeeze()
        # scale = 4
        # hist_img = np.zeros((100, w * scale, 3), dtype=np.uint8)
        flag = 0
        for i in range(len(point) - 5):
            binValue = point[i]
            if flag == 0 and binValue > 1 and point[i + 3] > 1 and point[i + 5] > 1:
                left_bound = i
                flag = 1
                continue
            if flag == 1 and binValue <= 1 and point[i + 3] <= 1 and point[i + 5] <= 1:
                right_bound = i
                flag = 2
                continue
            if flag == 2 and binValue > 10 and point[i + 3] > 10 and point[i + 5] > 10:
                flag = 1
            # cv2.rectangle(hist_img, pt1=(i * scale, 0), pt2=((i + 1) * scale, binValue),
            #               color=(255, 255, 0), thickness=2)
        # print('left bound: {}, right bound: {}'.format(left_bound, right_bound))
        # cv2.namedWindow('vertical projecting', cv2.WINDOW_GUI_EXPANDED)
        # cv2.imshow('vertical projecting', hist_img)
        return left_bound, right_bound

    def corners_matching(self, pre_corners, cur_corners):
        distances = []
        for i in range(len(cur_corners)):
            corners = np.roll(cur_corners, i, axis=0)
            distance = self._distance(pre_corners, corners)
            distances.append(distance)
        distances = np.array(distances)
        idx = np.argmin(distances)
        cur_corners = np.roll(cur_corners, idx, axis=0)
        return cur_corners

    def rotation_angle_calc(self, pre_center, pre_points, cur_center, cur_points):
        angles = []
        for i in range(4):
            v1 = pre_points[i] - pre_center
            v2 = cur_points[i] - cur_center
            if (v1 == v2).all():
                angle = 0
            else:
                # print("vector1: {}, vector2: {}".format(v1, v2))
                l1 = np.sqrt(v1.dot(v1))
                l2 = np.sqrt(v2.dot(v2))

                cos_angle = v1.dot(v2)/(l1 * l2)
                angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
        angle = np.mean(angles)
        return angle

    def _distance(self, pre_corners, cur_corners):
        distance = np.sum(abs(pre_corners[:, 0] - cur_corners[:, 0]) + abs(pre_corners[:, 1] - cur_corners[:, 1]))
        return distance


def detection(img):
    img_size = img.shape[0:2]
    roi_delta_x = int(img_size[0] * 0.1)
    roi_delta_y = int(img_size[1] * 0.1)
    threshold_offset = 10
    roi_offset = 50
    if img_size != (2048, 2048):
        print('the shape of input image {} should be {}'.format(img_size, (2048, 2048)))
        return None
    else:
        # print('the shape of input image: {}'.format(img_size))
        pass
    # Gauss filter
    # blur_img = img
    blur_img = cv2.GaussianBlur(img, (3, 3), 1, 0.5)
    coarse_roi_img = blur_img[roi_delta_x:img_size[0]-roi_delta_x,
                     roi_delta_y:img_size[1]-roi_delta_y]

    # threshold = self.otsu(coarse_roi_img, 0, 255) - self.threshold_offset
    # print('threshold: {}'.format(threshold))
    # _, binary_img = cv2.threshold(coarse_roi_img, 80, 255, cv2.THRESH_BINARY)

    _, binary_img = cv2.threshold(coarse_roi_img, 0, 255, cv2.THRESH_OTSU)
    roi_h, roi_w = binary_img.shape[0:2]
    resize_img = cv2.resize(binary_img, (roi_w >> 3, roi_h >> 3), cv2.INTER_CUBIC)
    upper_bound, bottom_bound = horizontal_projecting(resize_img)
    left_bound, right_bound = vertical_projecting(resize_img)

    x1 = (left_bound << 3) + roi_delta_x - roi_offset
    x2 = (right_bound << 3) + roi_delta_x + roi_offset
    y1 = (upper_bound << 3) + roi_delta_y - roi_offset
    y2 = (bottom_bound << 3) + roi_delta_y + roi_offset
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > img_size[0]:
        x2 = img_size[0]
    if y2 > img_size[1]:
        y2 = img_size[1]
    # print('x1:{}, x2:{}'.format(x1, x2))
    # print('y1:{}, y2:{}'.format(y1, y2))

    fine_roi_img = blur_img[y1:y2, x1:x2]
    roi_h, roi_w = fine_roi_img.shape[0:2]
    resize_img = cv2.resize(fine_roi_img, (roi_w >> 2, roi_h >> 2), cv2.INTER_CUBIC)
    # resize_img = cv2.morphologyEx(resize_img, cv2.MORPH_OPEN, self.morph_kernel)
    # key_points = self.detector.detect(resize_img)
    # keypoints_img = cv2.drawKeypoints(resize_img, key_points, (255, 0, 255),
    #                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # edge_img = cv2.Canny(resize_img, 10, 70)
    _, binary_img = cv2.threshold(resize_img, 0, 255, cv2.THRESH_OTSU)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, morph_kernel)
    contour_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    # contour_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3))
    contours, hierachy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours number: {}'.format(len(contours)))

    if len(contours) > 1:
        print('=> Failed to detect object, retry again.')
        return None
    rotate_rect = cv2.minAreaRect(contours[0])
    rotate_angle = rotate_rect[2]
    corners = cv2.boxPoints(rotate_rect)
    print(rotate_angle)
    cv2.drawContours(contour_img, [np.int0(corners)], 0, (255, 255, 0), 2)

    original_corners = corners * (2 ** 2)
    original_corners[:, 0] = original_corners[:, 0] + x1
    original_corners[:, 1] = original_corners[:, 1] + y1
    cv2.drawContours(blur_img, [np.int0(original_corners)], 0, (255, 255, 0), 2)


    # cv2.drawContours(contour_img, contours, -1, (255, 255, 0), 1)
    # cv2.namedWindow('binary image', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('binary image', binary_img)
    # cv2.namedWindow('roi image', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('roi image', blur_img)
    # cv2.waitKey(2)
    # cv2.destroyWindow('roi image')
    return original_corners


def horizontal_projecting(img):
    w, h = img.shape[0:2]
    point = np.zeros(h, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if img[y][x] > 0:
                point[y] = point[y] + 1
    point = cv2.normalize(point, 0, 100, cv2.NORM_INF).squeeze()
    # print(point)
    # scale = 4
    # hist_img = np.zeros((h * scale, 100, 3), dtype=np.uint8)
    flag = 0
    for i in range(len(point) - 5):
        binValue = point[i]
        if flag == 0 and binValue > 1 and point[i + 3] > 1 and point[i + 5] > 1:
            upper_bound = i
            flag = 1
            continue
        if flag == 1 and binValue <= 1 and point[i + 3] <= 1 and point[i + 5] <= 1:
            bottom_bound = i
            flag = 2
            continue
        if flag == 2 and binValue > 10 and point[i + 3] > 10 and point[i + 5] > 10:
            flag = 1
        # cv2.rectangle(hist_img, pt1=(0, i * scale), pt2=(binValue, (i + 1) * scale),
        #               color=(255, 255, 0), thickness=2)
    # print('upper bound: {}, bottom bound: {}'.format(upper_bound, bottom_bound))
    # cv2.namedWindow('horizontal projecting', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('horizontal projecting', hist_img)
    return upper_bound, bottom_bound


def vertical_projecting(img):
    w, h = img.shape[0:2]
    point = np.zeros(h, dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            if img[y][x] > 0:
                point[x] = point[x] + 1
    point = cv2.normalize(point, 0, 100, cv2.NORM_INF).squeeze()
    # scale = 4
    # hist_img = np.zeros((100, w * scale, 3), dtype=np.uint8)
    flag = 0
    for i in range(len(point) - 5):
        binValue = point[i]
        if flag == 0 and binValue > 1 and point[i + 3] > 1 and point[i + 5] > 1:
            left_bound = i
            flag = 1
            continue
        if flag == 1 and binValue <= 1 and point[i + 3] <= 1 and point[i + 5] <= 1:
            right_bound = i
            flag = 2
            continue
        if flag == 2 and binValue > 10 and point[i + 3] > 10 and point[i + 5] > 10:
            flag = 1
        # cv2.rectangle(hist_img, pt1=(i * scale, 0), pt2=((i + 1) * scale, binValue),
        #               color=(255, 255, 0), thickness=2)
    # print('left bound: {}, right bound: {}'.format(left_bound, right_bound))
    # cv2.namedWindow('vertical projecting', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('vertical projecting', hist_img)
    return left_bound, right_bound


if __name__ == '__main__':
    ################################################################
    # satellite dataset loading
    ################################################################
    dataset = Satellite_dataset('E:\\Satellite_wings')
    # dataset.format_transformation()
    # dataset.dataset_display()

    ################################################################
    # fast_detector class for sequential detection
    ################################################################
    left_detector = Fast_detector(img_size=dataset.img_size)
    right_detector = Fast_detector(img_size=dataset.img_size)

    ################################################################
    # multi-processing pool for parallel detection
    ################################################################
    multi_process = Pool(2)


    for f, (left_img, right_img) in enumerate(dataset):
        print('=> frame: {}/{}'.format(f, dataset.img_num))
        start_time = time.time()
        left_corners, left_center, left_rotation_angle = left_detector.detection(left_img)
        right_corners, right_center, right_rotation_angle = right_detector.detection(right_img)
        end_time = time.time()
        elapsed_time2 = end_time - start_time
        print('=> successfully sequentially detection in {} ms '.format(elapsed_time2 * 1000))

        ################################################################
        # multi-processing detection
        ################################################################
        # start_time = time.time()
        # results = multi_process.map(detection, [left_img, right_img])
        # left_corners, right_corners = results
        # end_time = time.time()
        # elapsed_time1 = end_time - start_time
        # print('=> successfully multi-processing detection in {} ms'.format(elapsed_time1 * 1000))

        ################################################################
        # sequential detection
        ################################################################


        ################################################################
        # detection result displaying
        ################################################################

        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(left_img, [np.int0(left_detector.pre_corners)], 0, (255, 0, 255), 1)
        cv2.drawContours(left_img, [np.int0(left_corners)], 0, (255, 255, 0), 3)

        cv2.drawContours(right_img, [np.int0(right_detector.pre_corners)], 0, (255, 0, 255), 1)
        cv2.drawContours(right_img, [np.int0(right_corners)], 0, (255, 255, 0), 3)

        detected_result = np.vstack((left_img, right_img))
        cv2.namedWindow('detection displaying', cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('detection displaying', detected_result)

        key = cv2.waitKey(1)
        if key == 27:
            exit(0)