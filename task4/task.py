import threading
import time
import logging
import queue
import argparse

import cv2


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")
    
class SensorX(Sensor):
    def __init__(self, delay: float):
        self.delay = delay
        self._data = 0
        self.running = True
        self.queue = queue.Queue()
        self._thread = threading.Thread(target=self.run)
        self._thread.start()
        
    def run(self):
        while self.running:
            self.queue.put(self._data)
            self._data += 1
            time.sleep(self.delay)
    
    def get(self) -> int:
        if not self.queue.empty():
            return self.queue.get()
        else:
            return self._data
        
    def stop(self):
        self.running = False
        self._thread.join()  
            
class SensorCam(Sensor):
    def __init__(self, cam, imgsz = (1280,720)):
        self._cam = cv2.VideoCapture(cam)
        self._imgsz = imgsz
        if self._cam is None or not self._cam.isOpened():
            logging.error("Camera did not open")
            raise RuntimeError("Camera did not open")
  
    def get(self) -> int:
        ret, frame = self._cam.read()
        if not ret:
            logging.error("Failed to read from camera") 
            print("Error: Failed to read from camera")
            return None
        else:
            return cv2.resize(frame, self._imgsz)
        
    def __del__(self):
        if self._cam is not None:
            self._cam.release()
            
class WindowImage:
    def __init__(self, delay: float):
        self.delay = delay
        self.running = True
        
    def show(self, img, sensors_data):
        if img is None:
            self.running = False
        
        for i, data in enumerate(sensors_data):
            text = f"Sensor {i}: {data}"
            cv2.putText(img, text, (50, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
           
        cv2.imshow("Sensor", img)
        
        key = cv2.waitKey(self.delay)
        if key & 0xFF == ord('q') or cv2.getWindowProperty("Sensor", cv2.WND_PROP_VISIBLE) < 1:
            self.running = False
            
    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str)
    parser.add_argument("--resolution", type=str, default="1280x720")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, filename="./log/logs.log", filemode="w")

    width, height = map(int, args.resolution.split('x'))
    camera = SensorCam(cam=args.camera, imgsz=(width, height))
    sensors = [SensorX(delay=0.01), SensorX(delay=0.1), SensorX(delay=1)]
    window = WindowImage(delay=args.fps)
    
    try:
        while window.running:
            frame = camera.get()
            if frame is None:
                break
            sensors_data = [sensor.get() for sensor in sensors]
            window.show(frame, sensors_data)
    finally:
        for sensor in sensors:
            sensor.stop()
        del camera
        del window
        