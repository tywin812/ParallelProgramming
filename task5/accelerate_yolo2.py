import argparse
import time
import threading
import queue
from ultralytics import YOLO
import cv2

class VideoProcessor:
    def __init__(self, model_path,video_source, output_path):
        self.model_path= model_path
        self.video_source = video_source
        self.input_queue = queue.Queue()
        self.frame_count = 0
        self.output_path = output_path
        self.output_buffer = {}
        self.out_lock = threading.Lock()
        self.writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'),fps=30.0,frameSize=(640, 480))

    def get_keypoints(self,model, frame):
        result = model.predict(frame, conf = 0.5, imgsz=(640,480))
        return result[0].plot()
    
    def capture_frames(self):
        cap = cv2.VideoCapture(self.video_source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.input_queue.put((self.frame_count, frame))
            self.frame_count += 1
        cap.release()
        
    def process_frames(self):
        model = YOLO(self.model_path)
        while True:
            idx, frame = self.input_queue.get()
            
            if frame is None:
                break
            
            processed_frame = self.get_keypoints(model,frame)
            processed_frame = cv2.resize(processed_frame, (640, 480))
            with self.out_lock:
                self.output_buffer[idx] = processed_frame
                            
    def single_threaded_process(self):
        model = YOLO(self.model_path)
        video = cv2.VideoCapture(self.video_source)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            result = model.predict(frame, conf=0.5, imgsz=(640, 480))
            processed_frame = result[0].plot()
            processed_frame = cv2.resize(processed_frame, (640, 480))
            self.writer.write(processed_frame)
        video.release()

    def multi_threaded_process(self, num_threads):
        reader_thread = threading.Thread(target=self.capture_frames)
        reader_thread.start()
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=self.process_frames)
            thread.start()
            threads.append(thread)
        
        reader_thread.join() 
        
        for _ in range(num_threads):
            self.input_queue.put((None, None))
        for thread in threads:
            thread.join()
            
        for i in range(self.frame_count):
            self.writer.write(self.output_buffer[i])
        
    def run(self, multi_threaded=True, num_threads=4):
        start = time.time()
        
        if multi_threaded:
            self.multi_threaded_process(num_threads)
        else:
            self.single_threaded_process()
            
        end = time.time()
        
        print(f"Mode: {f'Multi thread({num_threads} threads)' if multi_threaded else 'Single thread'}. Finished in {end - start:.2f} sec.")

        
    def __del__(self):
        self.writer.release()


def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return value
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str_or_int)
    parser.add_argument("--mode", type=str, default="multithread")
    parser.add_argument("--output", type=str, default="output.mp4")
    args = parser.parse_args()
    
    videoProcessor = VideoProcessor(model_path="yolov8s-pose.pt", video_source=args.video, output_path=args.output)
    if (args.mode == 'singlethread'):
        videoProcessor.run(multi_threaded=False)
    else:
        videoProcessor.run(multi_threaded=True, num_threads=2)