from mpsClient import MPSClient, MPSUnits, MPSTypes
import numpy as np


# Connect to MPS server
mps = MPSClient("10.1.164.251", 4201)
mps = MPSClient("140.109.177.86", 4201)


# Run_Diagnostic
mps.run_diagnostic()


# Set_Power_or_Reset FPGA sectors
mps.set_power_or_reset(1, 1, [1, 1, 1, 1, 1, 1])


# Move_Positioner Command
# The following two methods are the same for target assignment
# method 1, use numpy structure
targets = np.empty([2], dtype=MPSTypes.move_positioner)
targets[0] = (1, 1, 5.0, 1.0)
targets[1] = (1, 2, 3.0, 2.0)
# method 2, use dictionary
targets = {
    'Module_Id':[1, 1],
    'Positioner_Id':[1, 2],
    'Theta_Joint1':[5.0, 3.0],
    'Phi_Joint2':[1.0, 2.0],
}
#
mps.move_positioner(
    targets,
    unit=MPSUnits.STEP,
    use_fast_map=True
)


# Set_HardStop_Orientation
mps.set_hardstop_orientation({
    'Module_Id':[1, 1],
    'Positioner_Id':[1, 2],
    'HardStop_Ori':[1, 0],
})
mps.set_hardstop_orientation(np.array([
    (1, 1, 1),
    (1, 2, 0),
    ], dtype=MPSTypes.set_hardstop_orientation
))


# Mps_Software
mps.mps_software(shutdown=False, restart=False, save_database=False)


# Export_Database_To_Xml_File
mps.export_database_to_xml_file('test', {
    'Module_Id':[1, 1],
    'Positioner_Id':[1, 2],
})
mps.export_database_to_xml_file(
    'test',
    np.array([(1, 1), (1, 2)], dtype=MPSTypes.export_database_to_xml_file)
)


# Import_Database_From_Xml_File
mps.import_database_from_xml_file('test')


# Set_Database_Data Command
import pickle
with open('data.pkl', 'rb') as f:
	data = pickle.load(f)
mps.set_database_data(data)


# Get_Database_Data Command
(res, errMsg, data) = mps.get_database_data({
    'Module_Id':[1],
    'Positioner_Id':[1],
})
(res, errMsg, data) = mps.get_database_data(
    np.array([(1, 1), (1, 2)], dtype=MPSTypes.get_database_data)
)


# Mps_Move_Positioner_With_Delay
targets = {
    'Module_Id':[1, 1],
    'Positioner_Id':[1, 2],
    'Theta_Joint1':[1.0, 3.0],
    'Delay_Theta_Joint1':[1.0, 5.0],
    'Phi_Joint2':[1.0, 6.0],
    'Delay_Phi_Joint2':[1.0, 7.0],
}
targets = np.array([
    (1, 1, 1.0, 1.0, 1.0, 1.0),
    (1, 2, 3.0, 5.0, 6.0, 7.0),
    ], dtype=MPSTypes.move_positioner_with_delay
)
mps.move_positioner_with_delay(
    targets,
    unit=MPSUnits.STEP,
    use_fast_map=True
)


# Move_Positioner_Interval_Duration Command
targets = {
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Theta_Joint1':[1.0],
    'Theta_Joint1_Interval':[1.0],
    'Theta_Joint1_Duration':[1.0],
    'Phi_Joint2':[1.0],
    'Phi_Joint2_Interval':[1.0],
    'Phi_Joint2_Duration':[1.0],
}
targets = np.array([
    (1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ], dtype=MPSTypes.move_positioner_interval_duration
)
mps.move_positioner_interval_duration(
    targets,
    unit=MPSUnits.STEP,
    use_fast_map=False
)


# Set_Current_Position
mps.set_current_position({
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Current_Position_X':[0.0],
    'Current_Position_Y':[0.0],
    'fixed_arm':[False],
})
mps.set_current_position(np.array(
    [(1, 1, 0.0, 0.0, False)],
    dtype=MPSTypes.set_current_position
))


# Get_Telemetry_Data Command
(res, errMsg, timing, telemetry) = mps.get_telemetry_data({
    'Module_Id':[1],
    'Positioner_Id':[1],
})
(res, errMsg, timing, telemetry) = mps.get_telemetry_data(np.array(
    [(1, 1), (1, 2)],
    dtype=MPSTypes.get_telemetry_data
))


# Calibrate_Motor_Frequencies Command
targets = {
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Theta_Joint1_Start_Frequency':[0.0],
    'Theta_Joint1_End_Frequency':[10.0],
    'Phi_Joint2_Start_Frequency':[0.0],
    'Phi_Joint2_End_Frequency':[10.0],
    'Move_Theta':[True],
    'Move_Phi':[True],
}
targets = np.array(
    [(1, 1, 0.0, 10.0, 0.0, 10.0, True, True)],
    dtype=MPSTypes.calibrate_motor_frequencies
)
mps.calibrate_motor_frequencies(
    targets,
    update_database = True,
    save_database = True
)


# Move_To_Target Command
targets = {
    'Module_Id':[1],
    'Positioner_Id':[1],
    'Current_Position_X':[0.0],
    'Current_Position_Y':[0.0],
    'Target_Position_X':[10.0],
    'Target_Position_Y':[10.0],
    'X_axes_Uncertainty':[1.0],
    'Y_axes_Uncertainty':[1.0],
    'Joint1_Delay_Count':[0.0],
    'Joint2_Delay_Count':[0.0],
    'fixed_arm':[False],
    'target_latched':[False],
}
targets = np.array(
    [(1, 1, 0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 0.0, 0.0, False, False)],
    dtype=MPSTypes.move_to_target
)
# sequence_number, iteration_number, target_number, targets
mps.move_to_target(1, 0, 25, targets)


# Not used anymore
mps.go_home_all()
