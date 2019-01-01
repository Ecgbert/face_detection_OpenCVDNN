# face_detection_OpenCVDNN

稍微練習一下OpenCV用於DNN inference的操作
不得不說這真的很方便，通吃各家的model還不用擔心各種依賴真的很爽啊
回到正題

這邊使用的Model是res10_300x300_ssd，有分caffe跟tensorflow版本，其中caffe有分為單精度(FP32)跟半精度(FP16)的model
tensorflow則只有提供量化int8一種
如果你想嘗試其他model，這邊提供連結 https://github.com/opencv/opencv_extra/tree/master/testdata/dnn

單純用於face detection的情況下不要求太嚴苛的話用tf的量化model就可以了

![image](https://github.com/lisssse14/face_detection_OpenCVDNN/blob/master/demo.gif)
