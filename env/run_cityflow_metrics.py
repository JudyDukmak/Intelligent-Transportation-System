import cityflow

eng = cityflow.Engine("config/config.json", thread_num=1)

for t in range(50):   # 50 simulation steps
    eng.next_step()

    # Example signals you can log each step (or every N steps)
    if t <51:
        print(
            "t=", eng.get_current_time(),
            "\nvehicles=", eng.get_vehicle_count(),
            "\n--------------------------------",
            "\navg_travel_time=", eng.get_average_travel_time(),
            "\n--------------------------------",
            "\nlane_vehicle_count=", eng.get_lane_vehicle_count(),
            "\n--------------------------------",
            "\nlane_vehicles=", eng.get_lane_vehicles(),
            "\n--------------------------------",
            "\nlane_waiting_vehicle_count=", eng.get_lane_waiting_vehicle_count(),
            "\n--------------------------------",
            "\nvehicles=", eng.get_vehicles(),
            "\n--------------------------------",
            
        )