# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, uos, gc

#sensor.reset()                         # Reset and initialize the sensor.
#sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
#sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
#sensor.set_windowing((240, 240))       # Set 240x240 window.
#sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')


print(net.ram())
print(net.len())
print(net.run())

clock = time.clock()

"""
while(True):
    clock.tick()

    print("test")
    img = image.Image("cell0039.PNG")

    for obj in tf.classify(net, img):
        # This combines the labels and confidence values into a list of tuples
        # and then sorts that list by the confidence values.
        sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
        for i in range(2):
            print("%s = %f" % (sorted_list[i][0], sorted_list[i][1]))
    print(clock.fps(), "fps")
"""
