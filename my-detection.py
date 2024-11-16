#from jetson_inference import detectNet
#from jetson_utils import videoSource,videoOutput
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2",threshold=0.5)
#camera = jetson.utils.videoSource("/dev/video0")
camera = jetson.utils.videoSource("/home/nvidia/jetson-inference/data/images/airplane_0.jpg",argv=['--loop=1'])
display = jetson.utils.videoOutput("/home/nvidia/jetson-inference/data/result/airplane_0.jpg")
while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        print("None")
        continue

    detections = net.Detect(img)
    print(type(detections))
    print(detections[0])
    print(detections[1])
    print(detections[2])
    display.Render(img)
    display.SetStatus("Object Detection | Network{:.0f}FPS".format(net.GetNetworkFPS()))
