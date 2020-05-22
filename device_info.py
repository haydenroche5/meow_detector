import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    print('##################')
    print('DEVICE {}'.format(i))
    print('##################')
    print(p.get_device_info_by_index(i))
