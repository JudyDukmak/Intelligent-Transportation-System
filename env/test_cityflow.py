import cityflow
import inspect

CONFIG_PATH = "config/config.json"

def list_public_callables(obj):
    names = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            names.append(name)
    return sorted(names)

eng = cityflow.Engine(CONFIG_PATH, thread_num=1)

print("Engine created OK")
print("Engine type:", type(eng))

print("\n=== Public callable methods on eng ===")
for name in list_public_callables(eng):
    print("-", name)

print("\n=== Signatures for common methods (if present) ===")
for name in ["next_step", "get_current_time", "reset", "set_tl_phase", "get_vehicle_count", "get_all_info"]:
    if hasattr(eng, name):
        fn = getattr(eng, name)
        print(f"\n--- {name} ---")
        try:
            print("signature:", inspect.signature(fn))
        except Exception:
            print("signature: (unavailable)")
        doc = getattr(fn, "__doc__", None)
        if doc:
            first = doc.strip().splitlines()[0]
            if first:
                print("doc:", first)

if hasattr(eng, "get_current_time"):
    print("\nCurrent time:", eng.get_current_time())

# Step a few times
for _ in range(5):
    if hasattr(eng, "next_step"):
        eng.next_step()
    else:
        raise RuntimeError("eng.next_step() not found; see method list above.")

if hasattr(eng, "get_current_time"):
    print("After 5 steps:", eng.get_current_time())
print("Done")



