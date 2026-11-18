import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

import time
start = time.perf_counter()

# import tracemalloc
# tracemalloc.start()

from vispy import scene
from vispy import app
from vispy.visuals.transforms import STTransform
from variables import *
from physicsx import run_simulation
phistory = run_simulation(p, v, cno, t, dt, l, rc, k, r0, g, cell, weight)


canvas = scene.SceneCanvas(keys='interactive', 
                  show=True, 
                  size=(800, 600))

view = canvas.central_widget.add_view()

view.camera = 'turntable'

view.camera.center = (l/2, l/2, l/2)

view.camera.distance = 5*l


axis = scene.visuals.XYZAxis(parent=view.scene)
axis.transform = STTransform(scale=(l, l, l))


markers = scene.visuals.Markers(parent=view.scene)

markers.set_data(pos=phistory[0], face_color='blue', size=15)


frame_index = 0

def update(event):
    global frame_index
    markers.set_data(pos=phistory[frame_index], face_color='blue', size=15)
    frame_index += 1
    if frame_index >= len(phistory):
        frame_index = 0

end = time.perf_counter()

print(end - start, "seconds")

timer = app.Timer(interval=1/60.0)

timer.connect(update)

timer.start()

# current, peak = tracemalloc.get_traced_memory()
# tracemalloc.stop()

# print(peak / 1e9, "GB")

app.run()