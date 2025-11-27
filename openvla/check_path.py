from omni.usd import get_context
stage = get_context().get_stage()

print("\n[DEBUG] Listing all prims under /World/Franka:\n")
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "Franka" in path or "panda" in path.lower():
        print("  ->", path)
