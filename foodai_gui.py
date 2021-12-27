import sys
import os
import time
import argparse
import numpy as np
import cv2
import json
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
from tool.utils import *

################################################################################################################################################################################################


DEBUG = False

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Ui_MainWindow(object):

    def setupUi(self, MainWindow, context, buffers, image_size):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1124, 862)

        self.context = context
        self.buffers = buffers
        self.image_size = image_size

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(470, 0, 161, 61))

        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semilight")
        font.setPointSize(20)

        self.label.setFont(font)
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(700, 710, 131, 51))
        self.pushButton.setObjectName("pushButton")
        # self.pushButton.clicked.connect(self.CancelFeed)
        self.pushButton.clicked.connect(self.take_shot_main)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(920, 710, 131, 51))
        self.pushButton_2.setObjectName("pushButton_2")

        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(680, 70, 421, 311))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(0)
        
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)


        self.video_stream = QtWidgets.QLabel(self.centralwidget)
        self.video_stream.setGeometry(QtCore.QRect(10, 70, 631, 471))
        self.video_stream.setObjectName("video_stream")
        MainWindow.setCentralWidget(self.centralwidget)

        self.Worker1 = Worker1(self.context, self.buffers, self.image_size)
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1124, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.label.setText(_translate("MainWindow", "foodai v2.0"))
        self.pushButton.setText(_translate("MainWindow", "Start!"))
        self.pushButton_2.setText(_translate("MainWindow", "Pay!"))

        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Name"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Price"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Quantity"))

        self.video_stream.setText(_translate("MainWindow", ""))

    def ImageUpdateSlot(self, Image):
        self.video_stream.setPixmap(QPixmap.fromImage(Image))

    def cancel_feed(self):
        self.Worker1.stop()

    def take_shot_main(self):
        self.table_items, self.img = self.Worker1.take_shot_detect()
        self.load_data_table()
        self.freeze_shot()
    
    def freeze_shot(self):
        height, width, channels = self.img.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(self.img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        self.video_stream.setPixmap(pixmap_image)
        self.video_stream.show()
    	
    def load_data_table(self):
        row = 0
        self.tableWidget.setRowCount(len(self.table_items))
        for item in self.table_items:
            self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(item))
            row = row + 1



class Worker1(QThread):
    def __init__(self, context, buffers, image_size):
        super(QThread, self).__init__()
        self.context = context
        self.buffers = buffers
        self.image_size = image_size
        self.res_total = {}
        self.counter = 0
        self.ThreadActive = True
        self.Capture = cv2.VideoCapture("/dev/video0")

    ImageUpdate = pyqtSignal(QImage)

    def run(self):

        while self.ThreadActive:
            ret, self.frame = self.Capture.read()
            if ret:
                Image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()

    def take_shot_detect(self):
        self.counter = self.counter + 1
        image_src = self.frame
        num_classes = 80
        boxes = detect(self.context, self.buffers, image_src, self.image_size, num_classes)
        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'
        class_names = load_class_names(namesfile)
        img, table_items = plot_boxes_cv2(image_src, boxes[0], self.res_total, self.counter,savename='predictions_trt_' + str(self.counter) + '.jpg', class_names=class_names, color=None)
        return table_items, img

    # self.img_name = "opencv_frame_{}.png".format(self.img_counter)
    # cv2.imwrite(self.img_name, self.frame)
    # print("{} written!".format(self.img_name))
    # self.img_counter += 1


##################################################################################################################################################################################################

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def GiB(val):
    return val * 1 << 30


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.
    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.
    Returns:
        str: Path of data directory.
    Raises:
        FileNotFoundError
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory.",
                        default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Make sure data directory exists.
    if not (os.path.exists(data_path)):
        raise FileNotFoundError(data_path + " does not exist. Please provide the correct data path with the -d option.")

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(
                find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


TRT_LOGGER = trt.Logger()


def main_debug():
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow, context, buffers, image_size)
	MainWindow.show()
	sys.exit(app.exec_())


def main(engine_path, image_size):
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        buffers = allocate_buffers(engine, 1)
        IN_IMAGE_H, IN_IMAGE_W = image_size
        context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))

        #### GUI part ###################################################################################################
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow, context, buffers, image_size)
        MainWindow.show()
        sys.exit(app.exec_())
        ########################################################################################################

        # image_src = cv2.imread(image_path)
        # cap = cv2.VideoCapture("/dev/video0")
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        # cv2.namedWindow("test")

        # res_total = {}
        # res_total_json = json.dumps(res_total, indent = 4)
        # counter = 0
        # while True:
        #	ret, frame = cap.read()
        #	cv2.imshow("test", frame)
        #	k = cv2.waitKey(1)
        #
        #	if k % 256 == 27:
        #		break
        #	elif k % 256 == 32:
        #		counter = counter + 1
        #		image_src = frame
        #		num_classes = 80
        #		boxes = detect(context, buffers, image_src, image_size, num_classes)
        #		if num_classes == 20:
        #			namesfile = 'data/voc.names'
        #		elif num_classes == 80:
        #			namesfile = 'data/coco.names'
        #		else:
        #			namesfile = 'data/names'
        #		class_names = load_class_names(namesfile)
        #		plot_boxes_cv2(image_src, boxes[0], res_total, counter, savename='predictions_trt_'+str(counter)+'.jpg', class_names=class_names, color=None)


def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def detect(context, buffers, image_src, image_size, num_classes):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    ta = time.time()
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    print("Shape of the network input: ", img_in.shape)
    # print(img_in)

    inputs, outputs, bindings, stream = buffers
    print('Length of inputs: ', len(inputs))
    inputs[0].host = img_in

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print('Len of outputs: ', len(trt_outputs))

    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

    tb = time.time()

    print('-----------------------------------')
    print('    TRT inference time: %f' % (tb - ta))
    print('-----------------------------------')

    boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)

    return boxes


if __name__ == '__main__':
    engine_path = sys.argv[1]

    if len(sys.argv) < 4:
        image_size = (416, 416)
    elif len(sys.argv) < 5:
        image_size = (int(sys.argv[3]), int(sys.argv[3]))
    else:
        image_size = (int(sys.argv[3]), int(sys.argv[4]))

    if DEBUG:
        main_debug()
    else:
        main(engine_path, image_size)
