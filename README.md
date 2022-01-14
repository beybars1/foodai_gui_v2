# foodai_gui_v2

Create virtual environment with `requirements.txt`. You may need to install PyQt5.

Download ([gdrive](https://drive.google.com/file/d/1VpNLNr9xFheQu01jb-o2qye0E8Gv2YqW/view?usp=sharing)) and place `own_trt.engine` into project directory

Run with GUI:
```
python3 foodai_gui.py ./own_trt.engine 416 416
```

Run with raw TensorRT:
```
python3 demo_trt.py ./own_trt.engine ./data/dog.jpg 416 416
```
