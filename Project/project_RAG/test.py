import subprocess

def get_gpu_info():
    result = subprocess.run(['wmic', 'path', 'win32_videocontroller', 'get', 'caption'], capture_output=True, text=True)
    return result.stdout

gpu_info = get_gpu_info()
print(gpu_info)