"""Small stdout telemetry stream: lifecycle and resource samples, never environment values."""
import datetime,json,os,threading,time,subprocess

def emit(kind,**fields):print('MC_EVENT '+json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':kind,**fields},default=str),flush=True)
def sampler(stop):
 import psutil
 process=psutil.Process();process.cpu_percent()
 while not stop.wait(5):
  sample={'rss_bytes':process.memory_info().rss,'cpu_percent':process.cpu_percent()}
  try:
   data=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).strip();sample['gpu_csv']=data
  except (FileNotFoundError,subprocess.SubprocessError):pass
  emit('resources',**sample)
def start():
 stop=threading.Event();threading.Thread(target=sampler,args=(stop,),daemon=True).start();return stop
