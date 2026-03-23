import genesis as gs

print("OK")

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.build()

for _ in range(100):
    scene.step()

print("Done")