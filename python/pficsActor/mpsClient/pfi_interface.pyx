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


def parse_msg_header_response(unsigned char[:] buf):
	cdef:
		command_header *pHeader
		int cmd_id, cmd_counter, body_size

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


def parse_command_response(unsigned char[:] buf):
	cdef:
		command_response_error *pData
		int status, sz
		char *errStr

	pData = <command_response_error *> &buf[0];
	status = pData.StatusNumber
	sz = pData.Error_String_Size
	errStr = pData.Error_String
	print "Parse_Command_Response() ==>> StatusNumber=%d, Error_String_Size=%d" \
	      % (status, sz)
	print "Parse_Command_Response() ==>> Error_String='%s'" % errStr
	return (status, errStr[:sz])


def parse_send_database_data(unsigned char[:] buf):
	cdef uint32_t *pUsi
	cdef uint32_t sz
	cdef unsigned char *cp

	cp = &buf[0]
	pUsi = <uint32_t *> cp
	sz = pUsi[0]
	print "Parse_Send_Database_Data() ==>> Messgae_size=%d" % sz
	cp += sizeof(sz)
	return cp[:sz]


def parse_send_telemetry_data(unsigned char[:] buf):
	cdef int index = 0, i, sz
	cdef number_of_records *p_number_of_records
	cdef send_telemetry_data_record *p_telemetry_data_record

	p_number_of_records = <number_of_records *> &buf[index]
	sz = p_number_of_records.Number_Of_Records
	print "Parse_Send_Telemetry_Data() ==>> Number_Of_Records=%d" % sz

	index += sizeof(number_of_records)

	data = {
		'Module_Id': [],
		'Positioner_Id': [],
		'Iteration_number': [],
		'Target_X': [],
		'Target_Y': [],
		'Current_X': [],
		'Current_Y': [],
		'Target_Joint1_Angle': [],
		'Target_Joint2_Angle': [],
		'Target_Joint1_Quadrant': [],
		'Target_Joint2_Quadrant': [],
		'Current_Join1_Angle': [],
		'Current_Join2_Angle': [],
		'Joint1_Step': [],
		'Joint2_Step': [],
		'Current_Joint1_Quadrant': [],
		'Current_Joint2_Quadrant': [],
		'Seconds': [],
		'Milli_Seconds': []
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
		data['Iteration_number'].append(p_telemetry_data_record.Iteration_number)
		data['Target_X'].append(p_telemetry_data_record.Target_X)
		data['Target_Y'].append(p_telemetry_data_record.Target_Y)
		data['Current_X'].append(p_telemetry_data_record.Current_X)
		data['Current_Y'].append(p_telemetry_data_record.Current_Y)
		data['Target_Joint1_Angle'].append(p_telemetry_data_record.Target_Joint1_Angle)
		data['Target_Joint2_Angle'].append(p_telemetry_data_record.Target_Joint2_Angle)
		data['Target_Joint1_Quadrant'].append(p_telemetry_data_record.Target_Joint1_Quadrant)
		data['Target_Joint2_Quadrant'].append(p_telemetry_data_record.Target_Joint2_Quadrant)
		data['Current_Join1_Angle'].append(p_telemetry_data_record.Current_Join1_Angle)
		data['Current_Join2_Angle'].append(p_telemetry_data_record.Current_Join2_Angle)
		data['Joint1_Step'].append(p_telemetry_data_record.Joint1_Step)
		data['Joint2_Step'].append(p_telemetry_data_record.Joint2_Step)
		data['Current_Joint1_Quadrant'].append(p_telemetry_data_record.Current_Joint1_Quadrant)
		data['Current_Joint2_Quadrant'].append(p_telemetry_data_record.Current_Joint2_Quadrant)
		data['Seconds'].append(p_telemetry_data_record.Seconds)
		data['Milli_Seconds'].append(p_telemetry_data_record.Milli_Seconds)

	return data


def pack_move_to_target(sequence_number, iteration_number, positions, obstacle_avoidance, enable_blind_move):
	global command_header_counter
	cdef move_to_target_command Move_To_Target
	cdef char *cp
	cdef int cmd_size, npos, i

	Move_To_Target.Command_Header.Command_Id = Move_To_Target_ID
	npos = len(positions.Module_Id)
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
	npos = len(targets.Module_Id)
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
	Mps_Software_Command.Command_Header.Message_Size = header_size

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
	npos = len(positions.Module_Id)
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
	npos = len(positions.Module_Id)
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
	npos = len(positions.Module_Id)
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
	npos = len(positions.Module_Id)
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
	npos = len(positions.Module_Id)
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
