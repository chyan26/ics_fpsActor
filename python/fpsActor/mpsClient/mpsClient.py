from builtins import object
import socket
import enum

import pfi_interface as pfi
import logging


class MPSUnits(enum.IntEnum):
	RAD = pfi.unit_radians
	DEG = pfi.unit_degree
	STEP = pfi.unit_step


class MPSTypes(object):
	move_positioner = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
		('Theta_Joint1', 'f4'),
		('Phi_Joint2', 'f4'),
	]
	set_hardstop_orientation = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
		('HardStop_Ori', 'i2'),
	]
	export_database_to_xml_file = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
	]
	get_database_data = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
	]
	move_positioner_with_delay = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
		('Theta_Joint1', 'f4'),
		('Delay_Theta_Joint1', 'f4'),
		('Phi_Joint2', 'f4'),
		('Delay_Phi_Joint2', 'f4'),
	]
	move_positioner_interval_duration = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
		('Theta_Joint1', 'f4'),
		('Theta_Joint1_Interval', 'f4'),
		('Theta_Joint1_Duration', 'f4'),
		('Phi_Joint2', 'f4'),
		('Phi_Joint2_Interval', 'f4'),
		('Phi_Joint2_Duration', 'f4'),
	]
	set_current_position = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
		('Current_Position_X', 'f4'),
		('Current_Position_Y', 'f4'),
		('fixed_arm', '?'),
	]
	get_telemetry_data = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
	]
	calibrate_motor_frequencies = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
		('Theta_Joint1_Start_Frequency', 'f4'),
		('Theta_Joint1_End_Frequency', 'f4'),
		('Phi_Joint2_Start_Frequency', 'f4'),
		('Phi_Joint2_End_Frequency', 'f4'),
		('Move_Theta', '?'),
		('Move_Phi', '?'),
	]
	move_to_target = [
		('Module_Id', 'i2'),
		('Positioner_Id', 'i2'),
		('Current_Position_X', 'f4'),
		('Current_Position_Y', 'f4'),
		('Target_Position_X', 'f4'),
		('Target_Position_Y', 'f4'),
		('X_axes_Uncertainty', 'f4'),
		('Y_axes_Uncertainty', 'f4'),
		('Joint1_Delay_Count', 'f4'),
		('Joint2_Delay_Count', 'f4'),
		('fixed_arm', '?'),
		('target_latched', '?'),
	]


class MPSClient(object):

	def __init__(self, host, port, command_header_counter=0):
		self.host = host
		self.port = port
		pfi.set_command_header_counter(command_header_counter)
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.connect((host, port))

	def _send_command(self, data):
		self.sock.send(data)

	def _get_response(self, size):
		data = bytearray(size)
		self.sock.recv_into(data, size, socket.MSG_WAITALL)
		return data

	def reconnect(self):
		self.sock.close()
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.connect((self.host, self.port))

	def get_command_header_counter(self):
		return pfi.get_command_header_counter()

	def _get_command_response(self, cmd):
		self._send_command(cmd)
		# get command response (validate command)
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched")
		elif not pfi.isCommandValidate_Response(cmd_id):
			return (1, "Message header ID error: %d" % cmd_id)
		response_message = self._get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != pfi.status_ok:
			return (1, errStr)
		# get command response (Command is done)
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched")
		elif not pfi.isCommand_Response(cmd_id):
			return (1, "Message header ID error: %d" % cmd_id)
		response_message = self._get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != pfi.cmd_ok:
			return (1, errStr)
		return (0, errStr)

	def _get_database_data(self, cmd):
		self._send_command(cmd)
		# get command response (validate command)
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched", None)
		elif not pfi.isCommandValidate_Response(cmd_id):
			return (1, "Message header ID error: %d" % cmd_id, None)
		response_message = self._get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != pfi.status_ok:
			return (1, errStr, None)
		# get database data
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched", None)
		elif pfi.isSend_Database_Data(cmd_id):
			response_message = self._get_response(body_size)
			data = pfi.parse_send_database_data(response_message)
		elif pfi.isCommand_Response(cmd_id):
			response_message = self._get_response(body_size)
			status, errStr = pfi.parse_command_response(response_message)
			if status != pfi.cmd_ok:
				return (1, errStr, None)
			else:
				return (1, errStr, "Empty database data???")
		else:
			return (1, "Message header ID error: %d" % cmd_id, None)
		# get command response (Command is done)
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched") + data
		elif not pfi.isCommand_Response(cmd_id):
			return (1, "Message header ID error: %d" % cmd_id) + data
		response_message = self._get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != pfi.cmd_ok:
			return (1, errStr, data)
		else:
			return (0, errStr, data)

	def _get_telemetry_data(self, cmd):
		self._send_command(cmd)
		# get command response (validate command)
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched", None, None)
		elif not pfi.isCommandValidate_Response(cmd_id):
			return (1, "Message header ID error: %d" % cmd_id, None, None)
		response_message = self._get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != pfi.status_ok:
			return (1, errStr, None, None)
		# get telemetry response
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched", None, None)
		elif pfi.isSend_Telemetry_Data(cmd_id):
			response_message = self._get_response(body_size)
			telemetry = pfi.parse_send_telemetry_data(response_message)
		elif pfi.isCommand_Response(cmd_id):
			response_message = self._get_response(body_size)
			status, errStr = pfi.parse_command_response(response_message)
			if status != pfi.cmd_ok:
				return (1, errStr, None, None)
			else:
				return (1, errStr, None, "Empty telemetry data???")
		else:
			return (1, "Message header ID error: %d" % cmd_id, None, None)
		# get command response (Command is done)
		response_header = self._get_response(pfi.header_size)
		cmd_id, cmd_counter, body_size = pfi.parse_msg_header_response(response_header)
		if cmd_id < 0:
			return (1, "Command counter mismatched") + telemetry
		elif not pfi.isCommand_Response(cmd_id):
			return (1, "Message header ID error: %d" % cmd_id) + telemetry
		response_message = self._get_response(body_size)
		status, errStr = pfi.parse_command_response(response_message)
		if status != pfi.cmd_ok:
			return (1, errStr) + telemetry
		else:
			return (0, errStr) + telemetry

	def go_home_all(self, obstacle_avoidance=True, enable_blind_move=False, j1_use_fast_map=False, j2_use_fast_map=False):
		"""GO HOME ALL command"""
		# send command
		cmd_buffer = pfi.pack_go_home_all(obstacle_avoidance, enable_blind_move, j1_use_fast_map, j2_use_fast_map)
		return self._get_telemetry_data(cmd_buffer)

	def move_to_target(self, sequence_number, iteration_number, target_number, targets):
		"""Move To Target command"""
		# send command
		cmd_buffer = pfi.pack_move_to_target(sequence_number, iteration_number, target_number, targets)
		return self._get_telemetry_data(cmd_buffer)

	def calibrate_motor_frequencies(self, targets, update_database, save_database):
		"""Calibrate Motor Frequencies command"""
		# send command
		cmd_buffer = pfi.pack_calibrate_motor_frequencies(targets, update_database, save_database)
		return self._get_command_response(cmd_buffer)

	def mps_software(self, shutdown=False, restart=False, save_database=False):
		"""MPS software command"""
		# send command
		cmd_buffer = pfi.pack_mps_software(shutdown, restart, save_database)
		return self._get_command_response(cmd_buffer)

	def get_telemetry_data(self, targets):
		"""Get telemetry data command"""
		# send command
		cmd_buffer = pfi.pack_get_telemetry_data(targets)
		return self._get_telemetry_data(cmd_buffer)

	def set_current_position(self, targets):
		"""Set current position command"""
		# send command
		cmd_buffer = pfi.pack_set_current_position(targets)
		return self._get_command_response(cmd_buffer)

	def move_positioner(self, targets, unit, use_fast_map=False):
		"""Move Positioner command"""
		# send command
		cmd_buffer = pfi.pack_move_positioner(targets, unit, use_fast_map)
		return self._get_command_response(cmd_buffer)

	def move_positioner_interval_duration(self, targets, unit, use_fast_map=False):
		"""Move Positioner Interval Duration command"""
		# send command
		cmd_buffer = pfi.pack_move_positioner_interval_duration(targets, unit, use_fast_map)
		return self._get_command_response(cmd_buffer)

	def move_positioner_with_delay(self, targets, unit, use_fast_map=False):
		"""Move Positioner With Delay command"""
		# send command
		cmd_buffer = pfi.pack_move_positioner_with_delay(targets, unit, use_fast_map)
		return self._get_command_response(cmd_buffer)

	def get_database_data(self, targets):
		"""Get Database Data command"""
		# send command
		cmd_buffer = pfi.pack_get_database_data(targets)
		return self._get_database_data(cmd_buffer)

	def set_database_data(self, xml_data, save_database=False):
		"""Set Database Data command"""
		# send command
		cmd_buffer = pfi.pack_set_database_data(xml_data, save_database)
		return self._get_command_response(cmd_buffer)

	def import_database_from_xml_file(self, xml_data, save_database=False):
		"""Import Database from XML File command"""
		# send command
		cmd_buffer = pfi.pack_import_database_from_xml_file(xml_data, save_database)
		return self._get_command_response(cmd_buffer)

	def export_database_to_xml_file(self, xml_data, targets):
		"""Export Database to XML File command"""
		# send command
		cmd_buffer = pfi.pack_export_database_to_xml_file(xml_data, targets)
		return self._get_command_response(cmd_buffer)

	def set_hardstop_orientation(self, orientations):
		"""Set_HardStop_Orientation"""
		# send command
		cmd_buffer = pfi.pack_set_hardstop_orientation(orientations)
		return self._get_command_response(cmd_buffer)

	def set_power_or_reset(self, cmd, set_motor_freq, sectors):
		"""Set_Power_or_Reset"""
		# send command
		# cmd: 1 = Power on or off, 2 = reset sector
		# set_motor_freq: 1 = Enable, 0 = Disable
		cmd_buffer = pfi.pack_set_power_or_reset(cmd, set_motor_freq, sectors)
		return self._get_command_response(cmd_buffer)

	def run_diagnostic(self):
		"""Run_Diagnostic"""
		# send command
		cmd_buffer = pfi.pack_run_diagnostic()
		return self._get_command_response(cmd_buffer)
