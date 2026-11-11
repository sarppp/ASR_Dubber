import threading
import time
import unittest

from batch_dubbing.scheduler import run_videos


class SchedulingTests(unittest.TestCase):
    def test_two_video_limit_and_complete_chains(self):
        lock = threading.Lock()
        active = 0
        maximum = 0
        records = []

        def process(video):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            for stage in ('gpu', 'cpu', 'package'):
                with lock:
                    records.append((video, stage))
                time.sleep(0.005)
            with lock:
                active -= 1
            return 'done'

        report = run_videos(range(5), process)
        self.assertEqual(maximum, 2)
        self.assertEqual(len(report['results']), 5)
        for video in range(5):
            self.assertEqual([s for v, s in records if v == video], ['gpu', 'cpu', 'package'])

    def test_stops_admitting_new_work_after_failure(self):
        def process(video):
            if video == 0:
                raise RuntimeError('failed')
            time.sleep(0.01)
            return 'done'
        report = run_videos(range(8), process)
        self.assertIn(0, report['failures'])
        self.assertLessEqual(len(report['results']), 1)

    def test_rejects_excess_concurrency(self):
        with self.assertRaises(ValueError):
            run_videos([], lambda x: x, max_parallel=3)


if __name__ == '__main__':
    unittest.main()
