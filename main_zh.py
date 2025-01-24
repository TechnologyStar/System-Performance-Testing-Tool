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
    """自定义异常类，用于处理跑分过程中的错误"""
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
        """设置日志记录"""
        logging.basicConfig(
            filename=f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def cpu_benchmark(self):
        """CPU性能测试，包含多个复杂任务"""
        try:
            logging.info("开始CPU性能测试...")
            score = 0
            
            # 多线程压力测试
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
                    t.join(1)  # 设置超时，避免死锁
                return time.time()

            # 矩阵分解（LU分解）
            def matrix_decomposition():
                size = 200
                matrix = np.random.rand(size, size)
                start_time = time.time()
                np.linalg.slogdet(matrix)  # 使用矩阵的行列式来进行LU分解
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
            logging.info(f"CPU性能测试完成，得分：{self.scores['CPU']}")
            
        except Exception as e:
            logging.error(f"CPU性能测试失败: {str(e)}")
            raise BenchmarkError(f"CPU性能测试出错: {str(e)}")

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
        """GPU性能测试，包含深度学习任务"""
        try:
            logging.info("开始GPU性能测试...")
            
            if not torch.cuda.is_available():
                logging.warning("未检测到GPU")
                self.scores['GPU'] = 0
                return
            
            device = torch.device("cuda")
            torch.cuda.synchronize()

            # 矩阵乘法测试
            def matrix_multiplication():
                size = 4096
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                start_time = time.time()
                for _ in range(10):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                return time.time() - start_time

            # 深度学习模型训练
            def model_training():
                model = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Flatten(),
                    torch.nn.Linear(128*32*32, 10)
                ).to(device)
                
                input_data = torch.randn(32, 3, 32, 32, device=device)  # 模拟32张图片
                target = torch.randint(0, 10, (32,), device=device)

                optimizer = torch.optim.Adam(model.parameters())
                criterion = torch.nn.CrossEntropyLoss()

                start_time = time.time()
                for _ in range(5):  # 训练5个mini-batch
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
            logging.info(f"GPU性能测试完成，得分：{self.scores['GPU']}")
            
        except Exception as e:
            logging.error(f"GPU性能测试失败: {str(e)}")
            raise BenchmarkError(f"GPU性能测试出错: {str(e)}")

    def memory_benchmark(self):
        """内存性能测试，包含随机访问与并发访问"""
        try:
            logging.info("开始内存性能测试...")
            
            # 随机访问内存测试
            def memory_access_test():
                size = 10000000
                data = np.random.rand(size)
                random_indices = np.random.randint(0, size, size // 10)
                start_time = time.time()
                for idx in random_indices:
                    _ = data[idx]
                return time.time() - start_time

            # 并发内存访问
            def concurrent_memory_access():
                def worker():
                    data = np.random.rand(1000000)
                    _ = sum(data)
                threads = []
                for _ in range(4):  # 启动4个线程进行内存访问
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
            logging.info(f"内存性能测试完成，得分：{self.scores['Memory']}")
            
        except Exception as e:
            logging.error(f"内存性能测试失败: {str(e)}")
            raise BenchmarkError(f"内存性能测试出错: {str(e)}")

    def disk_benchmark(self):
        """磁盘性能测试，包含随机读写操作"""
        try:
            logging.info("开始磁盘性能测试...")
            
            # 随机读写测试
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
            logging.info(f"磁盘性能测试完成，得分：{self.scores['Disk']}")
            
        except Exception as e:
            logging.error(f"磁盘性能测试失败: {str(e)}")
            raise BenchmarkError(f"磁盘性能测试出错: {str(e)}")

    def network_benchmark(self):
        """网络性能测试，包含下载上传测试"""
        try:
            
            logging.info("开始网络性能测试...")
            
            def network_speed_test():
                # 定义测试网站列表（国内外混合）
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
                timeout = 5  # 设置超时时间为5秒
                
                for site in test_sites:
                    try:
                        start_time = time.time()
                        response = requests.get(site, timeout=timeout)
                        if response.status_code == 200:
                            end_time = time.time()
                            response_time = end_time - start_time
                            response_times.append(response_time)
                            logging.info(f"站点 {site} 响应时间: {response_time:.3f}秒")
                    except:
                        logging.warning(f"站点 {site} 访问失败")
                        continue
                
                if not response_times:
                    raise BenchmarkError("所有网站测试都失败")
                    
                # 选择最快的3个响应时间计算平均值
                fastest_times = sorted(response_times)[:3]
                avg_time = sum(fastest_times) / len(fastest_times)
                return avg_time

            speed = network_speed_test()
            # 将响应时间转换为得分（响应时间越短，得分越高）
            # 假设0.5秒响应时间得满分10000分，10秒以上得0分
            self.scores['Network'] = min(10000, max(0, int(10000 * (0.5 / speed))))
            logging.info(f"网络性能测试完成，平均响应时间：{speed:.3f}秒，得分：{self.scores['Network']}")
            
        except Exception as e:
            logging.error(f"网络性能测试失败: {str(e)}")
            raise BenchmarkError(f"网络性能测试出错: {str(e)}")

    def run_benchmark(self):
        """运行所有性能测试"""
        try:
            print("系统性能测试开始...\n")
            
            test_functions = [
                (self.cpu_benchmark, "CPU性能测试"),
                (self.gpu_benchmark, "GPU性能测试"),
                (self.memory_benchmark, "内存性能测试"),
                (self.disk_benchmark, "磁盘性能测试"),
                (self.network_benchmark, "网络性能测试")
            ]
            
            for func, name in test_functions:
                print(f"正在执行{name}...")
                try:
                    func()
                    print(f"{name}完成！")
                except BenchmarkError as e:
                    print(f"{name}失败: {str(e)}")
                print()
            
            # 计算总分
            total_score = sum(self.scores.values()) / len(self.scores)
            print("\n" + "="*50)
            print("测试结果".center(46))
            print("="*50)
            print(f"CPU性能得分：    {self.scores['CPU']:>7,d}")
            print(f"GPU性能得分：    {self.scores['GPU']:>7,d}")
            print(f"内存性能得分：   {self.scores['Memory']:>7,d}")
            print(f"磁盘性能得分：   {self.scores['Disk']:>7,d}")
            print(f"网络性能得分：   {self.scores['Network']:>7,d}")
            print("-"*50)
            print(f"总体性能得分：   {int(total_score):>7,d}")
            print("="*50)
            
            # 输出性能等级
            level = "S" if total_score > 8000 else "A" if total_score > 6000 else "B" if total_score > 4000 else "C" if total_score > 2000 else "D"
            print(f"\n系统性能等级：{level}")
            
            logging.info(f"测试完成，总分：{int(total_score)}，等级：{level}")
            
        except Exception as e:
            logging.error(f"测试过程出现严重错误: {str(e)}")
            print(f"测试过程出现严重错误: {str(e)}")
            raise

if __name__ == "__main__":
    benchmark = SystemBenchmark()
    benchmark.run_benchmark()
