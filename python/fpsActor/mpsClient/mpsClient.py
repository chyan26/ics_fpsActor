import socket
import pfi_interface as pfi


class MPSError(Exception):
	"""Exception for MPSClient"""
	pass


class MPSClient:

	def __init__(self, host, port, command_header_counter=0):
		self.host = host
		self.port = port
		pfi.set_command_header_counter(command_header_counter)
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.connect((host, port))

	def send_command(data):
		self.sock.send(data)

	def get_response(size):
		data = bytearray(size)
		self.sock.recv_into(data, size)
		return data

	def get_command_header_counter():
		return pfi.get_command_header_counter()

	def go_home_all(obstacle_avoidance=True, enable_blind_move=False, j1_use_fast_map=False, j2_use_fast_map=False):
		"""GO HOME ALL command"""
		# send command
		cmd_buffer = pfi.pack_go_home_all(obstacle_avoidance, enable_blind_move, j1_use_fast_map, j2_use_fast_map)
		send_command(cmd_buffer)
		# get command response
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Message body error %s", errStr)
		# get MPS status
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Send_Telemetry_Data_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		telemetry = pfi.parse_send_telemetry_data(response_message)
		return telemetry

	def move_to_target(sequence_number, iteration_number, targets, obstacle_avoidance=True, enable_blind_move=False):
		"""Move To Target command"""
		# send command
		cmd_buffer = pfi.pack_move_to_target(sequence_number, iteration_number, targets, obstacle_avoidance, enable_blind_move)
		send_command(cmd_buffer)
		# get command response
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)
		# get MPS status
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Send_Telemetry_Data_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		telemetry = pfi.parse_send_telemetry_data(response_message)
		return telemetry

	def calibrate_motor_frequencies(targets):
		"""Calibrate Motor Frequencies command"""
		# send command
		cmd_buffer = pfi.pack_calibrate_motor_frequencies(targets)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def mps_software(shutdown=False, restart=False, save_database=False):
		"""MPS software command"""
		# send command
		cmd_buffer = pfi.pack_mps_software(shutdown, restart, save_database)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def get_telemetry_data(targets):
		"""Get telemetry data command"""
		# send command
		cmd_buffer = pfi.pack_get_telemetry_data(targets)
		send_command(cmd_buffer)
		# get command response
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)
		# get MPS status
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Send_Telemetry_Data_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		telemetry = pfi.parse_send_telemetry_data(response_message)
		return telemetry

	def set_current_position(targets):
		"""Set current position command"""
		# send command
		cmd_buffer = pfi.pack_set_current_position(targets)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def move_positioner(targets):
		"""Move Positioner command"""
		# send command
		cmd_buffer = pfi.pack_move_positioner(targets)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def move_positioner_interval_duration(targets):
		"""Move Positioner Interval Duration command"""
		# send command
		cmd_buffer = pfi.pack_move_positioner_interval_duration(targets)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def move_positioner_with_delay(targets):
		"""Move Positioner With Delay command"""
		# send command
		cmd_buffer = pfi.pack_move_positioner_with_delay(targets)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def get_database_data(targets):
		"""Get Database Data command"""
		# send command
		cmd_buffer = pfi.pack_get_database_data(targets)
		send_command(cmd_buffer)
		# get command response
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)
		# get Database Data
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Send_Database_Data_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		data = pfi.parse_send_database_data(response_message)
		return data

	def set_database_data(xml_data, save_database=False):
		"""Set Database Data command"""
		# send command
		cmd_buffer = pfi.pack_set_database_data(xml_data, save_database)
		send_command(cmd_buffer)
		# get command response
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)
		# get Database Data
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Send_Database_Data_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		data = pfi.parse_send_database_data(response_message)
		return data

	def import_database_from_xml_file(xml_data, save_database=False):
		"""Import Database from XML File command"""
		# send command
		cmd_buffer = pfi.pack_import_database_from_xml_file(xml_data, save_database)
		send_command(cmd_buffer)
		# get command response
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)
		# get Database Data
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)

	def export_database_to_xml_file(xml_data, targets):
		"""Export Database to XML File command"""
		# send command
		cmd_buffer = pfi.pack_export_database_to_xml_file(xml_data, targets)
		send_command(cmd_buffer)
		# get command response
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)
		# get Database Data
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error %s", errStr)

	def set_hardstop_orientation(orientations):
		"""Set_HardStop_Orientation"""
		# send command
		cmd_buffer = pfi.pack_set_hardstop_orientation(orientations)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def set_power_or_reset(cmd, set_motor_freq, sectors):
		"""Set_Power_or_Reset"""
		# send command
		cmd_buffer = pfi.pack_set_power_or_reset(cmd, set_motor_freq, sectors)
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)

	def run_diagnostic():
		"""Run_Diagnostic"""
		# send command
		cmd_buffer = pfi.pack_run_diagnostic()
		send_command(cmd_buffer)
		# get command response (validate command)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
		# get command response (Command is done)
		response_header = get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id != Command_Response_ID:
			raise MPSError("Message header ID error: %d" % cmd_id)
		response_message = get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != 0:
			raise MPSError("Command error: %s", errStr)
