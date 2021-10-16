import os
import time
import numpy as np
import cv2
# import imageio
from skimage import transform, img_as_ubyte
from landmarks_detector import LandmarksDetector


origin_video_path = './origin_video/'
compress_video_path = './compress_video/'
origin_img_path = './origin_img/'
os.makedirs(compress_video_path, exist_ok=True)
os.makedirs(origin_img_path, exist_ok=True)

def process_one_img(rgb_img, save_dir, start_num, frame_num):
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(rgb_img), start=1):
        try:
            lm = np.array(face_landmarks)
            lm_chin          = lm[0  : 17]  # left-right
            lm_eyebrow_left  = lm[17 : 22]  # left-right
            lm_eyebrow_right = lm[22 : 27]  # left-right
            lm_nose          = lm[27 : 31]  # top-down
            lm_nostrils      = lm[31 : 36]  # top-down
            lm_eye_left      = lm[36 : 42]  # left-clockwise
            lm_eye_right     = lm[42 : 48]  # left-clockwise
            lm_mouth_outer   = lm[48 : 60]  # left-clockwise
            lm_mouth_inner   = lm[60 : 68]  # left-clockwise

            # Calculate auxiliary vectors.
            eye_left     = np.mean(lm_eye_left, axis=0)
            eye_right    = np.mean(lm_eye_right, axis=0)
            eye_avg      = (eye_left + eye_right) * 0.5
            eye_to_eye   = eye_right - eye_left
            mouth_left   = lm_mouth_outer[0]
            mouth_right  = lm_mouth_outer[6]
            mouth_avg    = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg
            chin_lowest = lm_chin[8]
            eye_to_chin = chin_lowest - eye_avg

            # eye_to_eye_dis = math.sqrt(eye_to_eye[0]*eye_to_eye[0] + eye_to_eye[1]*eye_to_eye[1])
            # eye_to_mouth_dis = math.sqrt(eye_to_mouth[0]*eye_to_mouth[0] + eye_to_mouth[1]*eye_to_mouth[1])
            # eye_to_chin_dis = math.sqrt(eye_to_chin[0]*eye_to_chin[0] + eye_to_chin[1]*eye_to_chin[1])
            eye_to_chin_dis = np.max(eye_to_chin)
            # center = eye_avg + eye_to_chin * 0.315
            center = eye_avg + eye_to_chin * 0.2
            center = center.astype(int)

            height, width, channels = np.shape(rgb_img)[0], np.shape(rgb_img)[1], np.shape(rgb_img)[2]

            half_side_lengh = eye_to_chin_dis*1.4
            half_side_lengh = np.ceil(half_side_lengh)
            half_side_lengh = half_side_lengh.astype(np.int64)

            if (half_side_lengh < 256):
                # print("side too small: ", half_side_lengh)
                return

            # crop cordinate 
            x_left = center[0] - half_side_lengh
            x_right = center[0] + half_side_lengh
            y_up = center[1] - half_side_lengh
            y_down = center[1] + half_side_lengh

            x_left_in_img = np.max([x_left, 0])
            x_right_in_img = np.min([x_right, width-1])
            y_up_in_img = np.max([y_up, 0])
            y_down_in_img = np.min([y_down, height-1])

            x_left_in_out = x_left_in_img - x_left
            x_right_in_out = half_side_lengh*2 - (x_right - x_right_in_img)
            y_up_in_out = y_up_in_img - y_up
            y_down_in_out = half_side_lengh*2 - (y_down - y_down_in_img)

            out_img = np.zeros(shape=(half_side_lengh*2+1, half_side_lengh*2+1, 3), dtype=np.uint8)
            out_img[y_up_in_out:y_down_in_out+1, x_left_in_out:x_right_in_out+1, :] = rgb_img[y_up_in_img:y_down_in_img+1, x_left_in_img:x_right_in_img+1, :]
            out_img = transform.resize(out_img, (512, 512), anti_aliasing=True)
            out_img = img_as_ubyte(out_img)
            # out_img = cv2.resize(out_img, (512, 512))
            # print("img dtype:", out_img.dtype.name)
            img_path = os.path.join(save_dir, "%d_%d_%d.png"%(start_num, frame_num, i))
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, out_img)
            # io.imsave(img_path, out_img)

            #background img
            bg_x_left = x_right_in_img
            bg_x_right = x_right_in_img + 256
            bg_y_up = y_up_in_img
            bg_y_down = y_up_in_img + 256
            if bg_x_right < width+1 and bg_y_down < height+1:
                bg_img = rgb_img[bg_y_up:bg_y_down, bg_x_left:bg_x_right, :]
                # bg_img = transform.resize(bg_img, (512, 512), anti_aliasing=True)
                # bg_img = img_as_ubyte(bg_img)
                # bg_img = cv2.resize(bg_img, (512, 512), interpolation=cv2.INTER_CUBIC)
                bg_img_path = os.path.join(save_dir, "%d_%d_%d_background.png"%(start_num, frame_num, i))
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(bg_img_path, bg_img)
        except:
            print("Exception in face crop!")


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.mp4') or f.endswith('.mov') or f.endswith('MP4'):
                fullname = os.path.join(root, f)
                yield fullname


# compress video to 2fps
video_list = findAllFile(origin_video_path)
for i, file_path in enumerate(video_list):
    print("Deal with::::::::::::::::::::", file_path)
    file_extension = os.path.splitext(file_path)[-1]
    dis_file_path = os.path.join(compress_video_path, str(i) + file_extension)
    shell_str = "ffmpeg -i %s -r 2 %s"%(file_path, dis_file_path)
    os.system(shell_str)
    
# crop face
landmarks_detector = LandmarksDetector("shape_predictor_68_face_landmarks.dat")
compress_video_list = findAllFile(compress_video_path)
for i, file_path in enumerate(compress_video_list):
    print("Deal with-----------------------", file_path)
    time_now = int(time.time())
    # vid = imageio.get_reader(file_path, 'ffmpeg')
    # for k,img in enumerate(vid):
    vid = cv2.VideoCapture(file_path)
    k = 0
    while True:
        k = k + 1
        ret, img = vid.read()
        if img is None:
            break
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        thumb = cv2.resize(img,(np.shape(rgb_img)[1]//6, np.shape(rgb_img)[0]//6))
        dets = landmarks_detector.detector(thumb, 1)
        if (len(dets) == 0):
            # print("no face")
            continue
        # print("yes face")
        save_dir_path = origin_img_path
        # print("img dtype:", rgb_img.dtype.name)
        out_img = process_one_img(rgb_img, save_dir_path, time_now, k)



print("------------------------end", )