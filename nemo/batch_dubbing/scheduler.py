"""Bound video-level concurrency while keeping each video in one slot."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def run_videos(videos, process_video, *, max_parallel=2):
    """Run complete video workflows with at most two simultaneously active.

    Stop admitting new videos on failure and wait for already-running work.
    `process_video` must return only after that video's whole workflow ends.
    This function itself does not launch Modal jobs.
    """
    if not 1 <= max_parallel <= 2:
        raise ValueError('Video concurrency must be one or two')
    pending = iter(videos)
    running, results, failures = {}, {}, {}
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        def submit_next():
            try:
                video = next(pending)
            except StopIteration:
                return False
            if video in running.values() or video in results or video in failures:
                raise ValueError('Duplicate video identifier')
            running[pool.submit(process_video, video)] = video
            return True

        for _ in range(max_parallel):
            if not submit_next():
                break
        while running:
            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                video = running.pop(future)
                try:
                    results[video] = future.result()
                except Exception as exc:
                    failures[video] = str(exc)
            if not failures:
                for _ in completed:
                    if not submit_next():
                        break
    return {'results': results, 'failures': failures}
