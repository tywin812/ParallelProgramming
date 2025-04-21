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
        
    def get(self) -> int:
        time.sleep(self.delay)
        self._data += 1
        return self._data
    
class SensorCam(Sensor):
    def __init__(self, cam, imgsz = (1280,720)):
        self._cam = cv2.VideoCapture(cam)
        self._imgsz = imgsz
        if self._cam is None or not self._cam.isOpened():
            logging.error("Camera did not open")
            raise RuntimeError("Camera did not open")

    def get(self) -> int:
        ret, frame = self._cam.read()
        if ret == False:
            logging.error("Failed to read from camera") 
            return None
        else:
            return cv2.resize(frame, self._imgsz)
            
    def __del__(self):
        if self._cam.isOpened():
            self._cam.release()
        
class WindowImage:
    def __init__(self, delay):
        self.delay = delay
        self.running = True
        
    def show(self, img):
        cv2.imshow("Window", img)
    
        key = cv2.waitKey(self.delay)
        
        if key & 0xFF == ord('q'):
            self.running = False
            
    def __del__(self): 
        cv2.destroyAllWindows()
        
def run_sensor(sensor, q: queue.Queue, stop_flag):
    while not stop_flag.is_set():      
        latest_data = sensor.get()
        if q.full():
            q.get()
        q.put(latest_data)
        
def run(sensors, window):
    queues = [queue.Queue(maxsize=1) for _ in range(len(sensors))]
    threads = []
    stop_flag = threading.Event()
    sensors_data= [0] * len(sensors)
    print(len(sensors_data))
    for i, sensor in enumerate(sensors):
        t = threading.Thread(target=run_sensor, args=(sensor, queues[i], stop_flag))
        t.start()
        threads.append(t)
    
    try:
        while window.running:
            sensors_data = [q.get() if not q.empty() else sensors_data[i] for i, q in enumerate(queues)]   
            if sensors_data[3] is not None:
                for i in range(3):
                    text = f"Sensor {i}: {sensors_data[i]}"
                    cv2.putText(sensors_data[3], text, (50, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)    
                window.show(sensors_data[3])
            else:
                break
    finally:
        stop_flag.set()
        for t in threads:
            t.join()
        del sensors[3]
        del window
                 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int)
    parser.add_argument("--resolution", type=str, default="1280x720")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, filename="./log/logs.log", filemode="w")

    width, height = map(int, args.resolution.split('x'))
    sensors = [SensorX(delay=0.01), SensorX(delay=0.1), SensorX(delay=1.0), SensorCam(cam=args.camera, imgsz=(width, height))]
    window = WindowImage(delay=int(1000/args.fps))
    run(sensors, window)
