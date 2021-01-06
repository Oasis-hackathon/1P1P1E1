from tensorflow.python.client import device_lib


local_device_protos = device_lib.list_local_devices()
for x in local_device_protos:
    print(x.name)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

device = get_available_gpus()

print(device)