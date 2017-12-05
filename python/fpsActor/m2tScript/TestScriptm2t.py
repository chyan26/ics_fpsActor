import mpsClient

mps = mpsClient.MPSClient("140.109.177.86", 4201)

mps.run_diagnostic()

mps.set_power_or_reset(1, 1, [1, 1, 1, 1, 1, 1])

mps.set_hardstop_orientation({
    'Module_Id':[1],
    'Positioner_Id':[1],
    'HardStop_Ori':[0],
})

mps.export_database_to_xml_file('test', {
    'Module_Id':[1],
    'Positioner_Id':[1],
})

mps.import_database_from_xml_file('test')

import pickle
with open('data.pkl') as f:
	data = pickle.load(f)

mps.set_database_data(data)

// command counter problem
data = mps.get_database_data({
    'Module_Id':[1],
    'Positioner_Id':[1],
})

mps.move_positioner_with_delay({
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Theta_Joint1':[1.0],
    'Delay_Theta_Joint1':[1.0],
    'Phi_Joint2':[1.0],
    'Delay_Phi_Joint2':[1.0],
})

mps.move_positioner_interval_duration({
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Theta_Joint1':[1.0],
    'Theta_Joint1_Interval':[1.0],
    'Theta_Joint1_Duration':[1.0],
    'Phi_Joint2':[1.0],
    'Phi_Joint2_Interval':[1.0],
    'Phi_Joint2_Duration':[1.0],
})

mps.move_positioner({
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Theta_Joint1':[1.0],
    'Phi_Joint2':[1.0],
})

mps.set_current_position({
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Current_Position_X':[0.0],
    'Current_Position_Y':[0.0],
    'fixed_arm':[False],
})

// command counter problem
telemetry = mps.get_telemetry_data({
    'Module_Id':[1],
    'Positioner_Id':[1],
})

mps.mps_software(shutdown=False, restart=False, save_database=False)

mps.calibrate_motor_frequencies({
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Theta_Joint1_Start_Frequency':[0.0],
    'Theta_Joint1_End_Frequency':[10.0],
    'Phi_Joint2_Start_Frequency':[0.0],
    'Phi_Joint2_End_Frequency':[10.0],
    'Move_Theta':[True],
    'Move_Phi':[True],
})

mps.move_to_target(1, 1, {
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Current_Position_X':[0.0],
    'Current_Position_Y':[0.0],
    'Target_Position_X':[0.0],
    'Target_Position_Y':[0.0],
    'X_axes_Uncertainty':[1.0],
    'Y_axes_Uncertainty':[1.0],
    'Joint1_Delay_Count':[0],
    'Joint2_Delay_Count':[0],
    'fixed_arm':[False],
    'target_latched':[False],
})

mps.go_home_all()
