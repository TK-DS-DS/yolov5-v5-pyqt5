import os

from PyQt5.QtCore import QTimer

from main_win.simply_win import Ui_Yolov5PyQt # 需要运行的.py文件名
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
import cv2
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from PyQt5 import QtCore, QtGui, QtWidgets

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

class MainWindow(QMainWindow, Ui_Yolov5PyQt):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # self.m_flag = False

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())
        #video 定时器
        self.timer_video = QtCore.QTimer()

        #设置yolov5相关参数
        self.yolov5_init()
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(
            weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

        self.is_save=False
        self.vid_cap = cv2.VideoCapture()
        # self.out = None
        self.vid_writer=None
        self.counter_video_saveimg=0
        self.video_name=""
        self.video_is_pause=0
        # self.camera_out=None

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./mydata/weights')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./mydata/weights/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        #修改设置：模型、置信度、iou阈值
        self.comboBox.currentTextChanged.connect(self.change_model)
        self.conf_thres.valueChanged.connect(lambda x: self.change_val(x, 'conf_thres'))
        self.iou_thres.valueChanged.connect(lambda x: self.change_val(x, 'iou_thres'))
        #选择 输入类型
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.radioButton_save.toggled.connect(self.change_save)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.pushButton_start.clicked.connect(self.show_video_frame)
        self.pushButton_pause.clicked.connect(self.video_pause)
        self.pushButton_next.clicked.connect(self.video_show_next)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        # self.pushButton_start.clicked.connect(self.detect_start())
        # self.parser = argparse.ArgumentParser()

    def yolov5_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='weights/yolov5s.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default='data/images', help='source')
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument(
            '--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)
    # 搜索pt 完成
    def search_pt(self):
        pt_list = os.listdir('./mydata/weights/')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./mydata/weights/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

        self.model_type = self.comboBox.currentText()
        self.opt.weights = "./mydata/weights/%s" % self.model_type
    # 改变model 完成
    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.opt.weights = "./mydata/weights/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)
    # 是否保存结果
    def change_save(self):
        if(self.radioButton_save.isChecked()==True):
            self.is_save=True
            self.statistic_msg("将自动保存检测结果")
        elif (self.radioButton_save.isChecked() == False):
            self.is_save = False
            self.statistic_msg("关闭保存检测结果")

    # 改变val 完成
    def change_val(self, x, flag):
        if flag == 'conf_thres':
            self.opt.conf_thres = round(x,2)
            self.statistic_msg('置信度设置为：'+str(round(x,2)))
        elif flag == 'iou_thres':
            # self.iou_thres.setValue(int(x*100))
            self.opt.iou_thres = round(x,2)
            self.statistic_msg('iou_thres设置为：' +str(round(x,2)))
        else:
            pass
    # 检测图片
    def button_image_open(self):
        name_list = []
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            return

        img = cv2.imread(img_name)
        showimg = img
        print(self.opt.weights)
        print(img_name)

        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # print(pred)

            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], showimg.shape).round()
                    self.listWidget_result.clear()
                    #统计每个类别的数量
                    for c in det[:, -1].unique():
                        # print(c)
                        # c_number = c_number + 1
                        n = (det[:, -1] == c).sum()  # detections per class
                        # s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                        whole_number = str('%g' % n)
                        class_id = str('%s' % self.names[int(c)])
                        # self.listWidget_result.setGeometry(120, 30, 800, 300)  # 设置QListWidget在窗口中的位置与大小
                        self.listWidget_result.addItem(whole_number+' '+class_id)  # 往QListWidget添加内容

                    #画框
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)
            # Save img?
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
            if (self.is_save == True):
                img_path, img_filename = os.path.split(img_name)
                cv2.imwrite('outputs/'+img_filename, showimg)

            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(
                self.result, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(
                self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
            self.label_resutlt.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
    # 检测视频
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        self.video_name=video_name

        if not video_name:
            return

        flag = self.vid_cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # video
            fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not os.path.exists('outputs'):
                os.makedirs('outputs')
            video_path, video_filename = os.path.split(video_name)
            self.vid_writer = cv2.VideoWriter('outputs/'+video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
            # 30ms执行一次帧刷新
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    # 展示视频写入
    def show_video_frame(self):
        if(self.video_is_pause==1):
            self.timer_video.start(30)
            self.video_is_pause=0
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)
            return

        name_list = []

        flag, img = self.vid_cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        self.listWidget_result.clear()
                        # 统计每个类别的数量
                        for c in det[:, -1].unique():
                            # print(c)
                            # c_number = c_number + 1
                            n = (det[:, -1] == c).sum()  # detections per class
                            # s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                            whole_number = str('%g' % n)
                            class_id = str('%s' % self.names[int(c)])
                            # self.listWidget_result.setGeometry(120, 30, 800, 300)  # 设置QListWidget在窗口中的位置与大小

                            self.listWidget_result.addItem(whole_number + ' ' + class_id)  # 往QListWidget添加内容

                        # 画框
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            # print(label)
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

            self.vid_writer.write(showimg)

            if (self.is_save == True):
                img_path, img_filename = os.path.split(self.video_name)
                video_imgsave_path ='outputs/' + img_filename.replace('.','')+'/'
                if not os.path.exists(video_imgsave_path):
                    os.makedirs(video_imgsave_path)
                cv2.imwrite(video_imgsave_path+str(self.counter_video_saveimg)+'.jpg', showimg)
                self.counter_video_saveimg=self.counter_video_saveimg+1

            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label_resutlt.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.timer_video.stop()
            self.vid_cap.release()
            self.vid_writer.release()
            self.label_resutlt.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
    # 下一帧
    def video_show_next(self):
        name_list = []

        flag, img = self.vid_cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        self.listWidget_result.clear()
                        # 统计每个类别的数量
                        for c in det[:, -1].unique():
                            # print(c)
                            # c_number = c_number + 1
                            n = (det[:, -1] == c).sum()  # detections per class
                            # s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                            whole_number = str('%g' % n)
                            class_id = str('%s' % self.names[int(c)])
                            # self.listWidget_result.setGeometry(120, 30, 800, 300)  # 设置QListWidget在窗口中的位置与大小

                            self.listWidget_result.addItem(whole_number + ' ' + class_id)  # 往QListWidget添加内容

                        # 画框
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            # print(label)
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

            self.vid_writer.write(showimg)

            if (self.is_save == True):
                img_path, img_filename = os.path.split(self.video_name)
                video_imgsave_path = 'outputs/' + img_filename.replace('.', '') + '/'
                if not os.path.exists(video_imgsave_path):
                    os.makedirs(video_imgsave_path)
                cv2.imwrite(video_imgsave_path + str(self.counter_video_saveimg) + '.jpg', showimg)
                self.counter_video_saveimg = self.counter_video_saveimg + 1

            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label_resutlt.setPixmap(QtGui.QPixmap.fromImage(showImage))
    # 视频检测暂停
    def video_pause(self):
        self.video_is_pause=1
        self.timer_video.start(3600000)
        self.pushButton_video.setDisabled(False)
        self.pushButton_img.setDisabled(False)
        self.pushButton_camera.setDisabled(False)
    # 摄像头检测
    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.vid_cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.vid_writer = cv2.VideoWriter('outputs/camera.mp4', cv2.VideoWriter_fourcc(
                    *'mp4v'), 20, (int(self.vid_cap.get(3)), int(self.vid_cap.get(4))))
                self.timer_video.start(10)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:
            self.timer_video.stop()
            self.vid_cap.release()
            self.vid_writer.release()
            self.label_resutlt.clear()
            # self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头")
    # 加载模型
    def load_model(self):
        imgsz = self.imgsz
        device = self.device
        # Load model
        model = attempt_load(self.opt.weights, map_location=device)  # load FP32 model
        half = device != 'cpu'
        if half:
            model.half()  # to FP16

        if device!='cpu':
            model(torch.zeros(1,3,imgsz,imgsz).to(device).type_as(next(model.parameters())))

        return model

    # 提示语句 完成
    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        self.qtimer.start(1500)#延时1500ms

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
    # app = QApplication(sys.argv)    # 创建应用程序
    # mainwindow = QMainWindow()      # 创建主窗口
    # ui = Ui_Yolov5PyQt()         # 调用中的主窗口
    # ui.setupUi(mainwindow)          # 向主窗口添加控件
    # mainwindow.show()               # 显示窗口
    # sys.exit(app.exec_())           # 程序执行循环