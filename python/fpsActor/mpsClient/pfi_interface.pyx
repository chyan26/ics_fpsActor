"""PFI - MPS interface"""

#
# Designed By: ChihYi Wen
# ASIAA - 2015
#

import time

cdef:
	unsigned long command_header_counter

command_header_counter = 0
header_size = sizeof(command_header)

def set_command_header_counter(counter):
	global command_header_counter
	command_header_counter = counter

def get_command_header_counter():
	global command_header_counter
	return command_header_counter


def pack_go_home_all(obstacle_avoidance, enable_blind_move, j1_use_fast_map, j2_use_fast_map):
	global command_header_counter
	cdef go_home_all_command Go_Home_All
	cdef char *cp

	Go_Home_All.Command_Header.Command_Id = Go_Home_All_ID
	Go_Home_All.Command_Header.Message_Size = sizeof(command_header) + sizeof(go_home_all_msg_record)

	now = time.time()
	Go_Home_All.Command_Header.Time_Stamp1 = int(now)
	Go_Home_All.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Go_Home_All.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1

	Go_Home_All.Command_Header.Flags = BIT_Step_Count

	Go_Home_All.Msg_Record.Flags = 0x0

	if obstacle_avoidance:
		Go_Home_All.Msg_Record.Flags |= BIT_Obstacle_Avoidance
		if enable_blind_move:
			Go_Home_All.Msg_Record.Flags |= BIT_Enable_Blind_Move

	if j1_use_fast_map:
		Go_Home_All.Msg_Record.Flags |= BIT_Theta_Use_Fast_Map


	if j2_use_fast_map:
		Go_Home_All.Msg_Record.Flags |= BIT_Phi_Use_Fast_Map

	cp = <char *> &Go_Home_All
	return cp[:sizeof(go_home_all_command)]


def parse_msg_header_response(resp):
	cdef:
		command_header *pHeader
		int cmd_id, cmd_counter, body_size
		unsigned char[:] buf

	buf = bytearray(resp)
	pHeader = <command_header *> &buf[0]
	cmd_id = pHeader.Command_Id
	cmd_counter = pHeader.Command_Counter
	body_size = pHeader.Message_Size - sizeof(command_header)

	print "Command_Id=%d, Message_Size=%d, Time_Stamp=%d.%03d," \
	      " Command_Counter=%d, Flags=%x" \
	      % (cmd_id, pHeader.Message_Size, pHeader.Time_Stamp1,
	         pHeader.Time_Stamp2, cmd_counter, pHeader.Flags)

	if not isValid_Mps_Response(cmd_id):
		print "Parse_Header_Response() ==>> MSG_HEADER_ID_ERROR"
		return (-1, 0, 0)
	elif cmd_counter + 1 != command_header_counter:
		print "Parse_Mps_Response() ==>> Command_Counter mismatched"
		return (-1, 0, 0)
	else:
		return (cmd_id, cmd_counter, body_size)


cdef isValid_Mps_Response(Response_Id):
	if Response_Id == Command_Response_ID:
		return True
	elif Response_Id == Send_Database_Data_ID:
		return True
	elif Response_Id == Send_Telemetry_Data_ID:
		return True
	else:
		return False

def isCommand_Response(Response_Id):
	if Response_Id == Command_Response_ID:
		return True
	else:
		return False

def isSend_Database_Data(Response_Id):
	if Response_Id == Send_Database_Data_ID:
		return True
	else:
		return False

def isSend_Telemetry_Data(Response_Id):
	if Response_Id == Send_Telemetry_Data_ID:
		return True
	else:
		return False


def parse_command_response(resp):
	cdef:
		command_response_error *pData
		int status, sz
		char *errStr
		unsigned char[:] buf

	buf = bytearray(resp)
	pData = <command_response_error *> &buf[0];
	status = pData.StatusNumber
	sz = pData.Error_String_Size
	errStr = pData.Error_String
	print "Parse_Command_Response() ==>> StatusNumber=%d, Error_String_Size=%d" \
	      % (status, sz)
	print "Parse_Command_Response() ==>> Error_String='%s'" % errStr
	return (status, errStr[:sz])


def parse_send_database_data(resp):
	cdef uint32_t *pUsi
	cdef uint32_t sz
	cdef unsigned char *cp
	cdef unsigned char[:] buf

	buf = bytearray(resp)
	cp = &buf[0]
	pUsi = <uint32_t *> cp
	sz = pUsi[0]
	print "Parse_Send_Database_Data() ==>> Messgae_size=%d" % sz
	cp += sizeof(sz)
	return cp[:sz]


def parse_send_telemetry_data(resp):
	cdef int index = 0, i, sz
	cdef execution_time_data *p_exec_time
	cdef number_of_records *p_number_of_records
	cdef send_telemetry_data_record *p_telemetry_data_record
	cdef unsigned char[:] buf

	buf = bytearray(resp)
	p_exec_time = <execution_time_data *> &buf[index]
	exec_time = {}
	exec_time['Mps_Command_Receive_Time'] = p_exec_time.Mps_Command_Receive_Time
	exec_time['FPGA_Command_Send_Time'] = p_exec_time.FPGA_Command_Send_Time
	exec_time['FPGA_Command_Response1_Time'] = p_exec_time.FPGA_Command_Response1_Time
	exec_time['FPGA_Command_Response2_Time'] = p_exec_time.FPGA_Command_Response2_Time
	exec_time['Mps_Command_Response_Time'] = p_exec_time.Mps_Command_Response_Time
	index += sizeof(execution_time_data)

	p_number_of_records = <number_of_records *> &buf[index]
	sz = p_number_of_records.Number_Of_Records
	print "Parse_Send_Telemetry_Data() ==>> Number_Of_Records=%d" % sz
	index += sizeof(number_of_records)

	data = {
		'Module_Id': [],
		'Positioner_Id': [],
		'Flags': [],
		'Target_Number': [],
		'Iteration_number': [],
		'Target_X': [],
		'Target_Y': [],
		'Target_Joint1_Angle': [],
		'Target_Joint2_Angle': [],
		'Target_Joint1_Quadrant': [],
		'Target_Joint2_Quadrant': [],
		'Current_X': [],
		'Current_Y': [],
		'Current_Join1_Angle': [],
		'Current_Join2_Angle': [],
		'Current_Joint1_Quadrant': [],
		'Current_Joint2_Quadrant': [],
		'Joint1_Step': [],
		'Joint2_Step': [],
		'Join1_Step_Delay': [],
		'Join2_Step_Delay': [],
		'Target_Changed': [],
		'Target_Changed_X': [],
		'Target_Changed_Y': [],
		'HardStop_Ori': [],
		'Joint1_from_Angle': [],
		'Joint1_to_Angle': [],
		'FPGA_Board_Number': [],
		'FPGA_Temp1': [],
		'FPGA_Temp2': [],
		'FPGA_Voltage': [],
		'FPGA_Join1_Frequency': [],
		'FPGA_Join1_Current': [],
		'FPGA_Join2_Frequency': [],
		'FPGA_Join2_Current': []
	}

	print "Parse_Send_Telemetry_Data() ==>> : \n"
	for i in range(sz):
		p_telemetry_data_record = <send_telemetry_data_record *> &buf[index];
		index += sizeof(send_telemetry_data_record);

		print "  Module_Id=%d, Positioner_Id=%d, Iteration_number=%d\n" % ( \
				p_telemetry_data_record.Module_Id, \
				p_telemetry_data_record.Positioner_Id, \
				p_telemetry_data_record.Iteration_number)
		data['Module_Id'].append(p_telemetry_data_record.Module_Id)
		data['Positioner_Id'].append(p_telemetry_data_record.Positioner_Id)
		data['Flags'].append(p_telemetry_data_record.Flags)
		data['Target_Number'].append(p_telemetry_data_record.Target_Number)
		data['Iteration_number'].append(p_telemetry_data_record.Iteration_number)
		data['Target_X'].append(p_telemetry_data_record.Target_X)
		data['Target_Y'].append(p_telemetry_data_record.Target_Y)
		data['Target_Joint1_Angle'].append(p_telemetry_data_record.Target_Joint1_Angle)
		data['Target_Joint2_Angle'].append(p_telemetry_data_record.Target_Joint2_Angle)
		data['Target_Joint1_Quadrant'].append(p_telemetry_data_record.Target_Joint1_Quadrant)
		data['Target_Joint2_Quadrant'].append(p_telemetry_data_record.Target_Joint2_Quadrant)
		data['Current_X'].append(p_telemetry_data_record.Current_X)
		data['Current_Y'].append(p_telemetry_data_record.Current_Y)
		data['Current_Join1_Angle'].append(p_telemetry_data_record.Current_Join1_Angle)
		data['Current_Join2_Angle'].append(p_telemetry_data_record.Current_Join2_Angle)
		data['Current_Joint1_Quadrant'].append(p_telemetry_data_record.Current_Joint1_Quadrant)
		data['Current_Joint2_Quadrant'].append(p_telemetry_data_record.Current_Joint2_Quadrant)
		data['Joint1_Step'].append(p_telemetry_data_record.Joint1_Step)
		data['Joint2_Step'].append(p_telemetry_data_record.Joint2_Step)
		data['Join1_Step_Delay'].append(p_telemetry_data_record.Join1_Step_Delay)
		data['Join2_Step_Delay'].append(p_telemetry_data_record.Join2_Step_Delay)
		data['Target_Changed'].append(p_telemetry_data_record.Target_Changed)
		data['Target_Changed_X'].append(p_telemetry_data_record.Target_Changed_X)
		data['Target_Changed_Y'].append(p_telemetry_data_record.Target_Changed_Y)
		data['HardStop_Ori'].append(p_telemetry_data_record.HardStop_Ori)
		data['Joint1_from_Angle'].append(p_telemetry_data_record.Joint1_from_Angle)
		data['Joint1_to_Angle'].append(p_telemetry_data_record.Joint1_to_Angle)
		data['FPGA_Board_Number'].append(p_telemetry_data_record.FPGA_Board_Number)
		data['FPGA_Temp1'].append(p_telemetry_data_record.FPGA_Temp1)
		data['FPGA_Temp2'].append(p_telemetry_data_record.FPGA_Temp2)
		data['FPGA_Voltage'].append(p_telemetry_data_record.FPGA_Voltage)
		data['FPGA_Join1_Frequency'].append(p_telemetry_data_record.FPGA_Join1_Frequency)
		data['FPGA_Join1_Current'].append(p_telemetry_data_record.FPGA_Join1_Current)
		data['FPGA_Join2_Frequency'].append(p_telemetry_data_record.FPGA_Join2_Frequency)
		data['FPGA_Join2_Current'].append(p_telemetry_data_record.FPGA_Join2_Current)

	return (exec_time, data)


def pack_move_to_target(sequence_number, iteration_number, positions, obstacle_avoidance, enable_blind_move):
	global command_header_counter
	cdef move_to_target_command Move_To_Target
	cdef char *cp
	cdef int cmd_size, npos, i

	Move_To_Target.Command_Header.Command_Id = Move_To_Target_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(move_to_target_msg_header) + \
		sizeof(move_to_target_msg_record) * npos
	Move_To_Target.Command_Header.Message_Size = cmd_size

	now = time.time()
	Move_To_Target.Command_Header.Time_Stamp1 = int(now)
	Move_To_Target.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Move_To_Target.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Move_To_Target.Command_Header.Flags = 0x0

	Move_To_Target.Msg_Header.Sequnce_Number = sequence_number
	Move_To_Target.Msg_Header.Iteration_Number = iteration_number
	Move_To_Target.Msg_Header.Target_Number = npos
	Move_To_Target.Msg_Header.Number_Of_Records = npos

	Move_To_Target.Msg_Header.Flags = 0x0
	if obstacle_avoidance:
		Move_To_Target.Msg_Header.Flags |= BIT_Obstacle_Avoidance
		if enable_blind_move:
			Move_To_Target.Msg_Header.Flags |= BIT_Enable_Blind_Move

	for i in range(npos):
		Move_To_Target.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Move_To_Target.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]
		Move_To_Target.Msg_Record[i].Current_Position_X = positions['Current_Position_X'][i]
		Move_To_Target.Msg_Record[i].Current_Position_Y = positions['Current_Position_Y'][i]
		Move_To_Target.Msg_Record[i].Target_Position_X = positions['Target_Position_X'][i]
		Move_To_Target.Msg_Record[i].Target_Position_Y = positions['Target_Position_Y'][i]
		Move_To_Target.Msg_Record[i].X_axes_Uncertainty = positions['X_axes_Uncertainty'][i]
		Move_To_Target.Msg_Record[i].Y_axes_Uncertainty = positions['Y_axes_Uncertainty'][i]
		Move_To_Target.Msg_Record[i].Joint1_Delay_Count = positions['Joint1_Delay_Count'][i]
		Move_To_Target.Msg_Record[i].Joint2_Delay_Count = positions['Joint2_Delay_Count'][i]
		Move_To_Target.Msg_Record[i].Flags = 0x0;

		if positions['fixed_arm'][i]:
			Move_To_Target.Msg_Record[i].Flags |= BIT_Fixed_Positioner

		if positions['target_latched'][i]:
			Move_To_Target.Msg_Record[i].Flags |= BIT_Target_Latched

	cp = <char *> &Move_To_Target
	return cp[:cmd_size]


def pack_calibrate_motor_frequencies(targets):
	global command_header_counter
	cdef calibrate_motor_frequencies_command Calibrate
	cdef char *cp
	cdef int cmd_size, npos, i

	Calibrate.Command_Header.Command_Id = Calibrate_Motor_Frequencies_ID
	npos = len(targets['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(calibrate_motor_frequencies_msg_record) * npos
	Calibrate.Command_Header.Message_Size = cmd_size

	now = time.time()
	Calibrate.Command_Header.Time_Stamp1 = int(now)
	Calibrate.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Calibrate.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Calibrate.Command_Header.Flags = 0x0

	Calibrate.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Calibrate.Msg_Record[i].Module_Id = targets['Module_Id'][i]
		Calibrate.Msg_Record[i].Positioner_Id = targets['Positioner_Id'][i]
		Calibrate.Msg_Record[i].Theta_Joint1_Start_Frequency = targets['Theta_Joint1_Start_Frequency'][i]
		Calibrate.Msg_Record[i].Theta_Joint1_End_Frequency = targets['Theta_Joint1_End_Frequency'][i]
		Calibrate.Msg_Record[i].Phi_Joint2_Start_Frequency = targets['Phi_Joint2_Start_Frequency'][i]
		Calibrate.Msg_Record[i].Phi_Joint2_End_Frequency = targets['Phi_Joint2_End_Frequency'][i]

		Calibrate.Msg_Record[i].Flags = 0
		if targets['Move_Theta'][i]:
			Calibrate.Msg_Record[i].Flags |= BIT_Move_Theta
		if targets['Move_Phi'][i]:
			Calibrate.Msg_Record[i].Flags |= BIT_Move_Phi

	cp = <char *> &Calibrate
	return cp[:cmd_size]


def pack_mps_software(shutdown, restart, save_database):
	global command_header_counter
	cdef mps_software_command Mps_Software_Command
	cdef char *cp

	Mps_Software_Command.Command_Header.Command_Id = Mps_Software_ID
	Mps_Software_Command.Command_Header.Message_Size = sizeof(mps_software_command)

	now = time.time()
	Mps_Software_Command.Command_Header.Time_Stamp1 = int(now)
	Mps_Software_Command.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Mps_Software_Command.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Mps_Software_Command.Command_Header.Flags = 0x0

	Mps_Software_Command.Msg_Record.Command_Flags = 0x0
	if save_database:
		Mps_Software_Command.Msg_Record.Command_Flags |= BIT_Save_The_Database;
	if shutdown:
		Mps_Software_Command.Msg_Record.Command_Flags |= BIT_Shut_Down;
	if restart:
		Mps_Software_Command.Msg_Record.Command_Flags |= BIT_Re_Start;

	cp = <char *> &Mps_Software_Command
	return cp[:sizeof(mps_software_command)]


def pack_get_telemetry_data(positions):
	global command_header_counter
	cdef get_telemetry_data_command Get_Telemetry
	cdef char *cp
	cdef int cmd_size, npos, i

	Get_Telemetry.Command_Header.Command_Id = Get_Telemetry_Data_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(get_telemetry_data_record) * npos
	Get_Telemetry.Command_Header.Message_Size = cmd_size

	now = time.time()
	Get_Telemetry.Command_Header.Time_Stamp1 = int(now)
	Get_Telemetry.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Get_Telemetry.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Get_Telemetry.Command_Header.Flags = 0x0

	Get_Telemetry.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Get_Telemetry.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Get_Telemetry.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]

	cp = <char *> &Get_Telemetry
	return cp[:cmd_size]


def pack_set_current_position(positions):
	global command_header_counter
	cdef set_current_position_data_command Set_Current_Position
	cdef char *cp
	cdef int cmd_size, npos, i

	Set_Current_Position.Command_Header.Command_Id = Set_Current_Position_Data_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(current_positioner_msg_record) * npos
	Set_Current_Position.Command_Header.Message_Size = cmd_size

	now = time.time()
	Set_Current_Position.Command_Header.Time_Stamp1 = int(now)
	Set_Current_Position.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Set_Current_Position.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Set_Current_Position.Command_Header.Flags = 0x0

	Set_Current_Position.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Set_Current_Position.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Set_Current_Position.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]
		Set_Current_Position.Msg_Record[i].Position_X = positions['Current_Position_X'][i]
		Set_Current_Position.Msg_Record[i].Position_Y = positions['Current_Position_Y'][i]

		Set_Current_Position.Msg_Record[i].Flags = 0x0;
		if positions['fixed_arm'][i]:
			Set_Current_Position.Msg_Record[i].Flags |= BIT_Fixed_Positioner

	cp = <char *> &Set_Current_Position
	return cp[:cmd_size]


def pack_move_positioner(positions):
	global command_header_counter
	cdef move_positioner_command Move_Positioner_command
	cdef char *cp
	cdef int cmd_size, npos, i

	Move_Positioner_command.Command_Header.Command_Id = Move_Positoner_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(move_positioner_msg_record) * npos
	Move_Positioner_command.Command_Header.Message_Size = cmd_size

	now = time.time()
	Move_Positioner_command.Command_Header.Time_Stamp1 = int(now)
	Move_Positioner_command.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Move_Positioner_command.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Move_Positioner_command.Command_Header.Flags = BIT_Step_Count

	Move_Positioner_command.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Move_Positioner_command.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Move_Positioner_command.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]
		Move_Positioner_command.Msg_Record[i].Theta_Joint1 = positions['Theta_Joint1'][i]
		Move_Positioner_command.Msg_Record[i].Phi_Joint2 = positions['Phi_Joint2'][i]
		Move_Positioner_command.Msg_Record[i].Flags = (BIT_Theta_Use_Fast_Map | BIT_Phi_Use_Fast_Map)

	cp = <char *> &Move_Positioner_command
	return cp[:cmd_size]


def pack_move_positioner_interval_duration(positions):
	global command_header_counter
	cdef move_positioner_int_dur_command Move_Positioner_Int_Dur
	cdef char *cp
	cdef int cmd_size, npos, i

	Move_Positioner_Int_Dur.Command_Header.Command_Id = Move_Positoner_Interval_Duration_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(move_positioner_int_dur_msg_record) * npos
	Move_Positioner_Int_Dur.Command_Header.Message_Size = cmd_size

	now = time.time()
	Move_Positioner_Int_Dur.Command_Header.Time_Stamp1 = int(now)
	Move_Positioner_Int_Dur.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Move_Positioner_Int_Dur.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Move_Positioner_Int_Dur.Command_Header.Flags = BIT_Step_Count

	Move_Positioner_Int_Dur.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Move_Positioner_Int_Dur.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Theta_Joint1 = positions['Theta_Joint1'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Theta_Joint1_Interval = positions['Theta_Joint1_Interval'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Theta_Joint1_Duration = positions['Theta_Joint1_Duration'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Phi_Joint2 = positions['Phi_Joint2'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Phi_Joint2_Interval = positions['Phi_Joint2_Interval'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Phi_Joint2_Duration = positions['Phi_Joint2_Duration'][i]
		Move_Positioner_Int_Dur.Msg_Record[i].Flags = 0

	cp = <char *> &Move_Positioner_Int_Dur
	return cp[:cmd_size]


def pack_move_positioner_with_delay(positions):
	global command_header_counter
	cdef move_positioner_with_delay_command Move_Positioner_With_Delay
	cdef char *cp
	cdef int cmd_size, npos, i

	Move_Positioner_With_Delay.Command_Header.Command_Id = Move_Positoner_With_Delay_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(move_positioner_with_delay_msg_record) * npos
	Move_Positioner_With_Delay.Command_Header.Message_Size = cmd_size

	now = time.time()
	Move_Positioner_With_Delay.Command_Header.Time_Stamp1 = int(now)
	Move_Positioner_With_Delay.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Move_Positioner_With_Delay.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Move_Positioner_With_Delay.Command_Header.Flags = BIT_Step_Count

	Move_Positioner_With_Delay.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Move_Positioner_With_Delay.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Move_Positioner_With_Delay.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]
		Move_Positioner_With_Delay.Msg_Record[i].Theta_Joint1 = positions['Theta_Joint1'][i]
		Move_Positioner_With_Delay.Msg_Record[i].Phi_Joint2 = positions['Phi_Joint2'][i]
		Move_Positioner_With_Delay.Msg_Record[i].Delay_Theta_Joint1 = positions['Delay_Theta_Joint1'][i]
		Move_Positioner_With_Delay.Msg_Record[i].Delay_Phi_Joint2 = positions['Delay_Phi_Joint2'][i]
		Move_Positioner_With_Delay.Msg_Record[i].Flags = 0

	cp = <char *> &Move_Positioner_With_Delay
	return cp[:cmd_size]


def pack_get_database_data(positions):
	global command_header_counter
	cdef get_database_data_command Get_Database_Data
	cdef char *cp
	cdef int cmd_size, npos, i

	Get_Database_Data.Command_Header.Command_Id = Get_Database_Data_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(get_database_data_record) * npos
	Get_Database_Data.Command_Header.Message_Size = cmd_size

	now = time.time()
	Get_Database_Data.Command_Header.Time_Stamp1 = int(now)
	Get_Database_Data.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Get_Database_Data.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Get_Database_Data.Command_Header.Flags = 0x0

	Get_Database_Data.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Get_Database_Data.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Get_Database_Data.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]

	cp = <char *> &Get_Database_Data
	return cp[:cmd_size]


def pack_set_database_data(xml_data, savedatabase):
	global command_header_counter
	cdef set_database_data_command Set_Database_Data
	cdef char *cp
	cdef int cmd_size, i

	Set_Database_Data.Command_Header.Command_Id = Set_Database_Data_ID
	cmd_size = sizeof(command_header) + sizeof(uint32_t) + len(xml_data)
	Set_Database_Data.Command_Header.Message_Size = cmd_size

	now = time.time()
	Set_Database_Data.Command_Header.Time_Stamp1 = int(now)
	Set_Database_Data.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Set_Database_Data.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	if savedatabase:
		Set_Database_Data.Command_Header.Flags = BIT_Save_The_Database
	else:
		Set_Database_Data.Command_Header.Flags = 0x0

	Set_Database_Data.Msg_Record.Xml_File_Size = len(xml_data)
	for i in range(len(xml_data)):
		Set_Database_Data.Msg_Record.Xml_File_Data[i] = ord(xml_data[i])

	cp = <char *> &Set_Database_Data
	return cp[:cmd_size]


def pack_import_database_from_xml_file(xml_data, savedatabase):
	global command_header_counter
	cdef import_database_from_xml_file_command Import_Xml
	cdef char *cp
	cdef int cmd_size, i

	Import_Xml.Command_Header.Command_Id = Import_Database_From_Xml_File_ID
	cmd_size = sizeof(command_header) + sizeof(uint32_t) + len(xml_data)
	Import_Xml.Command_Header.Message_Size = cmd_size

	now = time.time()
	Import_Xml.Command_Header.Time_Stamp1 = int(now)
	Import_Xml.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Import_Xml.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	if savedatabase:
		Import_Xml.Command_Header.Flags = BIT_Save_The_Database
	else:
		Import_Xml.Command_Header.Flags = 0x0

	Import_Xml.FileInfo.Name_Length = len(xml_data)
	for i in range(len(xml_data)):
		Import_Xml.FileInfo.Name[i] = ord(xml_data[i])

	cp = <char *> &Import_Xml
	return cp[:cmd_size]


def pack_export_database_to_xml_file(xml_data, positions):
	global command_header_counter
	cdef export_database_to_xml_file_command Export_Xml
	cdef char *cp
	cdef int cmd_size, npos, i, p1, p2

	Export_Xml.Command_Header.Command_Id = Export_Database_To_Xml_File_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + sizeof(uint32_t) \
	           + len(xml_data) + sizeof(export_database_to_xml_file_record) * npos
	Export_Xml.Command_Header.Message_Size = cmd_size

	now = time.time()
	Export_Xml.Command_Header.Time_Stamp1 = int(now)
	Export_Xml.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Export_Xml.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Export_Xml.Command_Header.Flags = BIT_Degree

	Export_Xml.Msg_Header.Number_Of_Records = npos

	Export_Xml.FileInfo.Name_Length = len(xml_data)
	for i in range(len(xml_data)):
		Export_Xml.FileInfo.Name[i] = ord(xml_data[i])

	for i in range(npos):
		Export_Xml.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Export_Xml.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]

	# Move positioner info to the end of XML filename
	p1 = sizeof(command_header) + sizeof(number_of_records) + sizeof(uint32_t) \
	     + len(xml_data)
	p2 = sizeof(command_header) + sizeof(number_of_records) + sizeof(xml_file_info)
	cp = <char *> &Export_Xml
	for i in range(sizeof(export_database_to_xml_file_record) * npos):
		cp[p1 + i] = cp[p2 + i]

	return cp[:cmd_size]


def pack_set_hardstop_orientation(positions):
	global command_header_counter
	cdef set_hardstop_orientation_data_command Set_HardStop_Ori
	cdef char *cp
	cdef int cmd_size, npos, i

	Set_HardStop_Ori.Command_Header.Command_Id = Set_HardStop_Orientation_ID
	npos = len(positions['Module_Id'])
	cmd_size = sizeof(command_header) + sizeof(number_of_records) + \
		sizeof(hardstop_orientation_msg_record) * npos
	Set_HardStop_Ori.Command_Header.Message_Size = cmd_size

	now = time.time()
	Set_HardStop_Ori.Command_Header.Time_Stamp1 = int(now)
	Set_HardStop_Ori.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Set_HardStop_Ori.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Set_HardStop_Ori.Command_Header.Flags = 0x0

	Set_HardStop_Ori.Msg_Header.Number_Of_Records = npos

	for i in range(npos):
		Set_HardStop_Ori.Msg_Record[i].Module_Id = positions['Module_Id'][i]
		Set_HardStop_Ori.Msg_Record[i].Positioner_Id = positions['Positioner_Id'][i]
		Set_HardStop_Ori.Msg_Record[i].HardStop_Ori = positions['HardStop_Ori'][i]

	cp = <char *> &Set_HardStop_Ori
	return cp[:cmd_size]


def pack_set_power_or_reset(cmd, set_motor_freq, sectors):
	global command_header_counter
	cdef set_sectors_power_or_reset_data_command Set_Power_or_Reset
	cdef char *cp
	cdef int cmd_size, i

	Set_Power_or_Reset.Command_Header.Command_Id = Set_Power_or_Reset_ID
	cmd_size = sizeof(set_sectors_power_or_reset_data_command)
	Set_Power_or_Reset.Command_Header.Message_Size = cmd_size

	now = time.time()
	Set_Power_or_Reset.Command_Header.Time_Stamp1 = int(now)
	Set_Power_or_Reset.Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Set_Power_or_Reset.Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1
	Set_Power_or_Reset.Command_Header.Flags = 0x0

	Set_Power_or_Reset.Msg_Record.Command_Type = cmd
	Set_Power_or_Reset.Msg_Record.Enable_Set_Motor_Frequencis = set_motor_freq
	for i in range(NUM_Power_Sectors):
		Set_Power_or_Reset.Msg_Record.Sectors[i] = sectors[i]

	cp = <char *> &Set_Power_or_Reset
	return cp[:cmd_size]


def pack_run_diagnostic():
	global command_header_counter
	cdef command_header Command_Header
	cdef char *cp

	Command_Header.Command_Id = Run_Diagnostic_ID
	Command_Header.Message_Size = sizeof(command_header)

	now = time.time()
	Command_Header.Time_Stamp1 = int(now)
	Command_Header.Time_Stamp2 = int((now - int(now)) * 1000)
	Command_Header.Command_Counter = command_header_counter
	command_header_counter += 1

	Command_Header.Flags = 0

	cp = <char *> &Command_Header
	return cp[:sizeof(command_header)]
