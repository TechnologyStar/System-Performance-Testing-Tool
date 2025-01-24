import psutil
import numpy as np
import time
import threading
import queue
import socket
import os
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import torch
import sys
import random
import requests

sys.stdout.reconfigure(encoding='utf-8')

class BenchmarkError(Exception):
    """Custom exception class for handling benchmark errors"""
    pass

class SystemBenchmark:
    def __init__(self):
        self.scores = {
            'CPU': 0,
            'GPU': 0,
            'Memory': 0,
            'Disk': 0,
            'Network': 0
        }
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging"""
        logging.basicConfig(
            filename=f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def cpu_benchmark(self):
        """CPU performance test, including multiple complex tasks"""
        try:
            logging.info("Starting CPU performance test...")
            score = 0
            
            # Multithreading stress test
            def cpu_stress_test():
                def worker():
                    while True:
                        _ = sum([random.random() for _ in range(1000)])
                threads = []
                for _ in range(psutil.cpu_count(logical=True)):
                    t = threading.Thread(target=worker)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join(1)  # Set timeout to avoid deadlock
                return time.time()

            # Matrix decomposition (LU decomposition)
            def matrix_decomposition():
                size = 200
                matrix = np.random.rand(size, size)
                start_time = time.time()
                np.linalg.slogdet(matrix)  # Use matrix determinant for LU decomposition
                return time.time() - start_time

            matrix_time = self.matrix_operation()
            prime_time = self.prime_calculation()
            stress_time = cpu_stress_test()
            decomposition_time = matrix_decomposition()
            
            matrix_score = int(10000 / (matrix_time * 100))
            prime_score = int(10000 / (prime_time * 100))
            stress_score = int(10000 / (stress_time * 10))
            decomposition_score = int(10000 / (decomposition_time * 10))
            
            score = (matrix_score + prime_score + stress_score + decomposition_score) // 4
            self.scores['CPU'] = min(10000, max(0, score))
            logging.info(f"CPU performance test completed, score: {self.scores['CPU']}")
            
        except Exception as e:
            logging.error(f"CPU performance test failed: {str(e)}")
            raise BenchmarkError(f"CPU performance test error: {str(e)}")

    def matrix_operation(self):
        size = 200
        matrix1 = np.random.rand(size, size)
        matrix2 = np.random.rand(size, size)
        start_time = time.time()
        for _ in range(20):
            result = np.dot(matrix1, matrix2)
        return time.time() - start_time

    def prime_calculation(self):
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        count = 0
        start_time = time.time()
        for num in range(2, 10000):
            if is_prime(num):
                count += 1
        return time.time() - start_time

    def gpu_benchmark(self):
        """GPU performance test, including deep learning tasks"""
        try:
            logging.info("Starting GPU performance test...")
            
            if not torch.cuda.is_available():
                logging.warning("No GPU detected")
                self.scores['GPU'] = 0
                return
            
            device = torch.device("cuda")
            torch.cuda.synchronize()

            # Matrix multiplication test
            def matrix_multiplication():
                size = 4096
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                start_time = time.time()
                for _ in range(10):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                return time.time() - start_time

            # Deep learning model training
            def model_training():
                model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Flatten(),
                    torch.nn.Linear(128*32*32, 10)
                ).to(device)
                
                input_data = torch.randn(32, 3, 32, 32, device=device)  # Simulate 32 images
                target = torch.randint(0, 10, (32,), device=device)

                optimizer = torch.optim.Adam(model.parameters())
                criterion = torch.nn.CrossEntropyLoss()

                start_time = time.time()
                for _ in range(5):  # Train 5 mini-batches
                    optimizer.zero_grad()
                    output = model(input_data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                return time.time() - start_time

            matrix_time = matrix_multiplication()
            training_time = model_training()
            
            matrix_score = int(10000 / (matrix_time * 10))
            training_score = int(10000 / (training_time * 10))
            
            final_score = (matrix_score + training_score) // 2
            self.scores['GPU'] = min(10000, max(0, final_score))
            logging.info(f"GPU performance test completed, score: {self.scores['GPU']}")
            
        except Exception as e:
            logging.error(f"GPU performance test failed: {str(e)}")
            raise BenchmarkError(f"GPU performance test error: {str(e)}")

    def memory_benchmark(self):
        """Memory performance test, including random access and concurrent access"""
        try:
            logging.info("Starting memory performance test...")
            
            # Random memory access test
            def memory_access_test():
                size = 10000000
                data = np.random.rand(size)
                random_indices = np.random.randint(0, size, size // 10)
                start_time = time.time()
                for idx in random_indices:
                    _ = data[idx]
                return time.time() - start_time

            # Concurrent memory access
            def concurrent_memory_access():
                def worker():
                    data = np.random.rand(1000000)
                    _ = sum(data)
                threads = []
                for _ in range(4):  # Start 4 threads for memory access
                    t = threading.Thread(target=worker)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
                return time.time()

            access_time = memory_access_test()
            concurrent_time = concurrent_memory_access()
            access_score = int(10000 / (access_time * 100))
            concurrent_score = int(10000 / (concurrent_time * 10))
            
            self.scores['Memory'] = min(10000, max(0, (access_score + concurrent_score) // 2))
            logging.info(f"Memory performance test completed, score: {self.scores['Memory']}")
            
        except Exception as e:
            logging.error(f"Memory performance test failed: {str(e)}")
            raise BenchmarkError(f"Memory performance test error: {str(e)}")

    def disk_benchmark(self):
        """Disk performance test, including random read/write operations"""
        try:
            logging.info("Starting disk performance test...")
            
            # Random read/write test
            def random_read_write_test():
                file_name = 'test_file.dat'
                with open(file_name, 'wb') as f:
                    f.write(os.urandom(10 * 1024 * 1024))  # 10 MB

                start_time = time.time()
                with open(file_name, 'rb') as f:
                    data = f.read()
                read_time = time.time() - start_time

                start_time = time.time()
                with open(file_name, 'wb') as f:
                    f.write(os.urandom(10 * 1024 * 1024))  # 10 MB
                write_time = time.time() - start_time

                os.remove(file_name)
                return read_time, write_time

            read_time, write_time = random_read_write_test()
            read_score = int(10000 / (read_time * 10))
            write_score = int(10000 / (write_time * 10))

            self.scores['Disk'] = min(10000, max(0, (read_score + write_score) // 2))
            logging.info(f"Disk performance test completed, score: {self.scores['Disk']}")
            
        except Exception as e:
            logging.error(f"Disk performance test failed: {str(e)}")
            raise BenchmarkError(f"Disk performance test error: {str(e)}")

    def network_benchmark(self):
        """Network performance test, including download/upload tests"""
        try:
            
            logging.info("Starting network performance test...")
            
            def network_speed_test():
                # Define test site list (mixed domestic and international)
                test_sites = [
                    'https://www.baidu.com',
                    'https://www.qq.com',
                    'https://www.taobao.com',
                    'https://www.jd.com',
                    'https://www.sina.com.cn',
                    'https://www.google.com',
                    'https://www.youtube.com',
                    'https://www.facebook.com',
                    'https://www.amazon.com',
                    'https://www.microsoft.com',
                    'https://www.twitter.com',
                    'https://www.instagram.com',
                    'https://www.linkedin.com',
                    'https://www.reddit.com',
                    'https://www.wikipedia.org',
                    'https://www.netflix.com',
                    'https://www.apple.com',
                    'https://www.github.com',
                    'https://www.stackoverflow.com',
                    'https://www.medium.com',
                    'https://www.ebay.com',
                    'https://www.paypal.com',
                    'https://www.dropbox.com',
                    'https://www.slack.com',
                    'https://www.trello.com',
                    'https://www.zoom.us',
                    'https://www.quora.com',
                    'https://www.pinterest.com',
                    'https://www.tumblr.com',
                    'https://www.wordpress.com'
                ]
                        
                response_times = []
                timeout = 5  # Set timeout to 5 seconds
                
                for site in test_sites:
                    try:
                        start_time = time.time()
                        response = requests.get(site, timeout=timeout)
                        if response.status_code == 200:
                            end_time = time.time()
                            response_time = end_time - start_time
                            response_times.append(response_time)
                            logging.info(f"Site {site} response time: {response_time:.3f} seconds")
                    except:
                        logging.warning(f"Site {site} access failed")
                        continue
                
                if not response_times:
                    raise BenchmarkError("All site tests failed")
                    
                # Select the fastest 3 response times to calculate the average
                fastest_times = sorted(response_times)[:3]
                avg_time = sum(fastest_times) / len(fastest_times)
                return avg_time

            speed = network_speed_test()
            # Convert response time to score (the shorter the response time, the higher the score)
            # Assume 0.5 seconds response time gets full score of 10000, 10 seconds or more gets 0
            self.scores['Network'] = min(10000, max(0, int(10000 * (0.5 / speed))))
            logging.info(f"Network performance test completed, average response time: {speed:.3f} seconds, score: {self.scores['Network']}")
            
        except Exception as e:
            logging.error(f"Network performance test failed: {str(e)}")
            raise BenchmarkError(f"Network performance test error: {str(e)}")

    def run_benchmark(self):
        """Run all performance tests"""
        try:
            print("System performance test starting...\n")
            
            test_functions = [
                (self.cpu_benchmark, "CPU performance test"),
                (self.gpu_benchmark, "GPU performance test"),
                (self.memory_benchmark, "Memory performance test"),
                (self.disk_benchmark, "Disk performance test"),
                (self.network_benchmark, "Network performance test")
            ]
            
            for func, name in test_functions:
                print(f"Running {name}...")
                try:
                    func()
                    print(f"{name} completed!")
                except BenchmarkError as e:
                    print(f"{name} failed: {str(e)}")
                print()
            
            # Calculate total score
            total_score = sum(self.scores.values()) / len(self.scores)
            print("\n" + "="*50)
            print("Test Results".center(46))
            print("="*50)
            print(f"CPU performance score:    {self.scores['CPU']:>7,d}")
            print(f"GPU performance score:    {self.scores['GPU']:>7,d}")
            print(f"Memory performance score: {self.scores['Memory']:>7,d}")
            print(f"Disk performance score:   {self.scores['Disk']:>7,d}")
            print(f"Network performance score:{self.scores['Network']:>7,d}")
            print("-"*50)
            print(f"Overall performance score:{int(total_score):>7,d}")
            print("="*50)
            
            # Output performance level
            level = "S" if total_score > 8000 else "A" if total_score > 6000 else "B" if total_score > 4000 else "C" if total_score > 2000 else "D"
            print(f"\nSystem performance level: {level}")
            
            logging.info(f"Test completed, total score: {int(total_score)}, level: {level}")
            
        except Exception as e:
            logging.error(f"Serious error occurred during the test: {str(e)}")
            print(f"Serious error occurred during the test: {str(e)}")
            raise

if __name__ == "__main__":
    benchmark = SystemBenchmark()
    benchmark.run_benchmark()
