# Face
https://drive.google.com/file/d/1lsQS8hOCquMFKJFhK_z-n03ixWGkjT2P/view

人像卡通化预训练模型：photo2cartoon_weights.pt(20200504更新)，存放在models路径下。
头像分割模型：seg_model_384.pb，存放在utils路径下。
人脸识别预训练模型：model_mobilefacenet.pth，存放在models路径下。（From: InsightFace_Pytorch）
卡通画开源数据：cartoon_data，包含trainB和testB。
人像卡通化onnx模型：photo2cartoon_weights.onnx 谷歌网盘，存放在models路径下。

https://github.com/YadiraF/PRNet/tree/master/Data

python combine.py --photo_path ./images/face04.jpg --save_path ./images/test.jpg
