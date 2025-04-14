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
        
def run_sensor(sensor, q: queue.Queue, stop_flag, last_data, idx):
    while not stop_flag.is_set():      
        latest_data = sensor.get()
        last_data[idx] = latest_data
        if q.full():
            q.get()
        q.put(latest_data)
        
def run_camera(cam, cam_queue: queue.Queue, stop_flag):
    while not stop_flag.is_set():
        frame = cam.get()
        if frame is not None:
            if cam_queue.full():
                cam_queue.get()
            cam_queue.put(frame)
        
def run(cam, sensors, window):
    queues = [queue.Queue(maxsize=1) for _ in range(len(sensors))]
    cam_queue = queue.Queue(maxsize=1)
    threads = []
    stop_flag = threading.Event()
    last_data = [0] * len(sensors)

    for i, sensor in enumerate(sensors):
        t = threading.Thread(target=run_sensor, args=(sensor, queues[i], stop_flag, last_data, i))
        t.start()
        threads.append(t)

    cam_thread = threading.Thread(target=run_camera, args=(cam, cam_queue, stop_flag))
    cam_thread.start()
    threads.append(cam_thread)
    
    try:
        while window.running:
            frame = cam_queue.get()
            sensors_data = [q.get() if not q.empty() else last_data[i] for i, q in enumerate(queues)]
            if frame is not None:
                for i, data in enumerate(sensors_data):
                    text = f"Sensor {i}: {data}"
                    cv2.putText(frame, text, (50, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)    
                window.show(frame)
            else:
                break
    finally:
        stop_flag.set()
        for t in threads:
            t.join()
        del cam
        del window
                 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int)
    parser.add_argument("--resolution", type=str, default="1280x720")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, filename="./log/logs.log", filemode="w")

    width, height = map(int, args.resolution.split('x'))
    camera = SensorCam(cam=args.camera, imgsz=(width, height))
    sensors = [SensorX(delay=0.01), SensorX(delay=0.1), SensorX(delay=1.0)]
    window = WindowImage(delay=int(1000/args.fps))
    run(camera, sensors, window)
