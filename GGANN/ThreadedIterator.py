import queue
import threading

"""
多线程任务执行
"""


class ThreadIterator:
    def __init__(self, original_iterator, max_queue_size:int=2 ):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self._thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self._queue.put(element, block=True)  # 阻塞任务列表
        self._queue.put(None, block=True)

    def __iter__(self):
        next_element = self._queue.get(block=True)
        while next_element is not  None:
            yield next_element
            next_element = self._queue.get(block=True)
        self._thread.join()