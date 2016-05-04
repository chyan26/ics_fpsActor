"""PFI - MPS interface"""

# 
# Designed By: ChihYi Wen
# ASIAA - 2015
#

from libc.stdint cimport uint32_t

cdef extern from "pfi_interface_defs.h" nogil:

	enum:
		STATUS_OK
		CMD_EXEC_OK
		CMD_EXEC_FAILED
		e100	#error
		e101	#error
		e102	#error
		e103	#error
		e104	#error

		MAX_RECORD_SIZE

		#Move_To_Target command bit positions
		MOVE_THETA
		MOVE_PHI
		OBTSTACLE_AVOIDNACE

		MAX_XML_FILE_NAME_SIZE
		MAX_ERROR_STRING_SIZE
		SET_DATABASE_DATA_MAX_SIZE
		CAL_COL_SIZE	#must match define in pfi_arm_ctrl.h

		#Go_Home_All flags bit positions
		Go_Home_All_OBSTACLE_AVOIDANCE_BIT
		Go_Home_All_USE_FAST_MAP_BIT

		#Flag Bits
		BIT_Radians
		BIT_Degree
		BIT_Step_Count
		BIT_Move_Theta
		BIT_Move_Phi
		BIT_Target_Latched
		BIT_Fixed_Positioner
		BIT_Obstacle_Avoidance
		BIT_Theta_Use_Fast_Map
		BIT_Phi_Use_Fast_Map
		BIT_Save_The_Database
		BIT_Shut_Down
		BIT_Re_Start
		BIT_Enable_Blind_Move
		BIT_Delay_Theta_Joint1
		BIT_Delay_Phi_Joint2

		#Pfi command to MPS IDS
		Move_To_Target_ID
		Go_Home_All_ID
		Go_Home_Positioner_ID
		Move_Positoner_ID
		Calibrate_Motor_Frequencies_ID
		Get_Status_Command_ID
		Get_Database_Data_ID
		Set_Database_Data_ID
		Set_Database_Field_ID
		Get_Telemetry_Data_ID
		Export_Database_To_Xml_File_ID
		Import_Database_From_Xml_File_ID
		Set_Current_Position_Data_ID
		Move_Positoner_Interval_Duration_ID
		Mps_Software_ID
		Move_Positoner_With_Delay_ID

		#MPS response to Pfi IDS
		Command_Response_ID
		Send_Database_Data_ID
		Send_Telemetry_Data_ID

	#Command Header
	struct command_header:
		uint32_t Command_Id
		uint32_t Message_Size
		uint32_t Time_Stamp1 #seconds
		uint32_t Time_Stamp2 #milliseconds
		uint32_t Command_Counter
		uint32_t Flags

	struct number_of_records:
		uint32_t Number_Of_Records
		#uint32_t Flags

	struct xml_file_info:
		uint32_t Name_Length
		char Name[MAX_XML_FILE_NAME_SIZE]

	#Move_To_Target Command
	struct move_to_target_msg_header:
		uint32_t Sequnce_Number
		uint32_t Iteration_Number
		uint32_t Target_Number
		uint32_t Number_Of_Records
		uint32_t Flags

	struct move_to_target_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		double Current_Position_X
		double Current_Position_Y
		double Target_Position_X
		double Target_Position_Y
		double X_axes_Uncertainty
		double Y_axes_Uncertainty
		double Joint1_Delay_Count
		double Joint2_Delay_Count
		uint32_t Flags

	struct move_to_target_command:
		command_header Command_Header
		move_to_target_msg_header Msg_Header
		move_to_target_msg_record Msg_Record[MAX_RECORD_SIZE]
		uint32_t Msg_Record_Count

	#Go_Home_All Command
	struct go_home_all_msg_record:
		uint32_t Flags #bit1: Enable Obstacle avoidance

	struct go_home_all_command:
		command_header Command_Header
		go_home_all_msg_record Msg_Record

	#Go_Home_Positioner Command
	struct go_home_positioner_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		uint32_t Flags

	struct go_home_positioner_command:
		command_header Command_Header
		number_of_records Msg_Header
		go_home_positioner_msg_record Msg_Record[MAX_RECORD_SIZE]

	#Move_Positioner Command

	struct move_positioner_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		double Theta_Joint1
		double Phi_Joint2
		uint32_t Flags

	struct move_positioner_command:
		command_header Command_Header
		number_of_records Msg_Header
		move_positioner_msg_record Msg_Record[MAX_RECORD_SIZE]

	#Move_Positioner_Interval_Duration Command

	struct move_positioner_int_dur_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		double Theta_Joint1
		double Theta_Joint1_Interval
		double Theta_Joint1_Duration
		double Phi_Joint2
		double Phi_Joint2_Interval
		double Phi_Joint2_Duration
		uint32_t Flags

	struct move_positioner_int_dur_command:
		command_header Command_Header
		number_of_records Msg_Header
		move_positioner_int_dur_msg_record Msg_Record[MAX_RECORD_SIZE]

	#Calibrate_Motor_Frequencies Command

	struct calibrate_motor_frequencies_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		double Theta_Joint1_Start_Frequency
		double Theta_Joint1_End_Frequency
		double Phi_Joint2_Start_Frequency
		double Phi_Joint2_End_Frequency
		uint32_t Flags

	struct calibrate_motor_frequencies_command:
		command_header Command_Header
		number_of_records Msg_Header
		calibrate_motor_frequencies_msg_record Msg_Record[MAX_RECORD_SIZE]

	#Get_Status Command

	struct get_status_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id

	struct get_status_command:
		command_header Command_Header
		number_of_records Msg_Header
		get_status_msg_record Msg_Record[MAX_RECORD_SIZE]

	#Set_Database_Data Command
	#later - later

	struct number_of_database_items:
		uint32_t Number_Of_Items
		uint32_t Save_To_Database #1=save, 2=do not save

	#Set_Database_Field Command

	struct set_database_field_msg_header:
		uint32_t Module_Id
		uint32_t Positioner_Id
		uint32_t Number_Of_Set_Fields

	struct set_database_field_id:
		uint32_t Field_Id
		#Field data, Depends on the field

	#Get_Database_Data

	struct get_database_data_record:
		uint32_t Module_Id
		uint32_t Positioner_Id

	struct get_database_data_command:
		command_header Command_Header
		number_of_records Msg_Header
		get_database_data_record Msg_Record[MAX_RECORD_SIZE]

	#Set_Database_Data

	struct set_database_data_record:
		uint32_t Xml_File_Size
		char Xml_File_Data[SET_DATABASE_DATA_MAX_SIZE]

	struct set_database_data_command:
		command_header Command_Header
		set_database_data_record Msg_Record

	#Get_Telemetry_Data

	struct get_telemetry_data_record:
		uint32_t Module_Id
		uint32_t Positioner_Id

	struct get_telemetry_data_command:
		command_header Command_Header
		number_of_records Msg_Header
		get_telemetry_data_record Msg_Record[MAX_RECORD_SIZE]

	#export_Database_To_Xml_File

	struct export_database_to_xml_file_record:
		uint32_t Module_Id
		uint32_t Positioner_Id

	struct export_database_to_xml_file_command:
		command_header Command_Header
		number_of_records Msg_Header
		xml_file_info FileInfo
		export_database_to_xml_file_record Msg_Record[MAX_RECORD_SIZE]

	#import_Database_From_Xml_File

	struct import_database_from_xml_file_command:
		command_header Command_Header
		xml_file_info FileInfo

	#Move_Positioner with delay Command

	struct move_positioner_with_delay_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		double Theta_Joint1
		double Phi_Joint2
		double Delay_Theta_Joint1
		double Delay_Phi_Joint2
		uint32_t Flags

	struct move_positioner_with_delay_command:
		command_header Command_Header
		number_of_records Msg_Header
		move_positioner_with_delay_msg_record Msg_Record[MAX_RECORD_SIZE]

	#Set_Current_Position_Data_ID

	struct current_positioner_msg_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		double Postion_X
		double Postion_Y
		uint32_t Flags

	struct set_current_position_data_command:
		command_header Command_Header
		number_of_records Msg_Header
		current_positioner_msg_record Msg_Record[MAX_RECORD_SIZE]

	#

	struct mps_software_record:
		uint32_t Command_Flags

	struct mps_software_command:
		command_header Command_Header
		mps_software_record Msg_Record



	#Command_Response Response

	struct command_response_error:
		uint32_t StatusNumber
		uint32_t Error_String_Size
		char Error_String[MAX_ERROR_STRING_SIZE] #NULL terminated text


	struct command_response:
		command_header Response_Header
		command_response_error MSG_Body

	#Send_Telemetry_Data

	struct send_telemetry_data_record:
		uint32_t Module_Id
		uint32_t Positioner_Id
		uint32_t Iteration_number
		double Target_X
		double Target_Y
		double Current_X
		double Current_Y
		double Target_Joint1_Angle
		double Target_Joint2_Angle
		uint32_t Target_Joint1_Quadrant
		uint32_t Target_Joint2_Quadrant
		double Current_Join1_Angle
		double Current_Join2_Angle
		uint32_t Joint1_Step
		uint32_t Joint2_Step
		uint32_t Current_Joint1_Quadrant
		uint32_t Current_Joint2_Quadrant
		uint32_t Seconds
		uint32_t Milli_Seconds

	struct send_telemetry_data_command:
		command_header Command_Header
		number_of_records Msg_Record_Count
		send_telemetry_data_record Msg_Record[MAX_RECORD_SIZE]


	#DataBase Ids

	enum:
		Positioner_serial_number_Id

		Base_Position_X_Id
		Base_Position_Y_Id
		Base_Orientation_Id

		Theta_Joint1_CW_Angle_limit_Id
		Theta_Joint1_CCW_Angle_limit_Id

		Phi_Joint2_CW_Angle_limit_Id
		Phi_Joint2_CCW_Angle_limit_Id

		Theta_Joint1_transition_angle_Id
		Phi_Joint2_transition_angle_Id

		Theta_Joint1_link_length_Id
		Phi_Joint2_link_length_Id

		Phi_Joint2_Base_Radius_Id
		Phi_Joint2_Tip_Radius_Id

		Length_uncertainty_Id

		Theta_Joint1_Frequency_Id
		Phi_Joint2_Frequency_Id

		Fast_Theta_Joint1_CW_Region_Count_Id
		Fast_Theta_Joint1_CW_Regions_Id
		Fast_Theta_Joint1_CW_Step_Sizes_Id

		Fast_Theta_Joint1_CCW_Region_Count_Id
		Fast_Theta_Joint1_CCW_Regions_Id
		Fast_Theta_Joint1_CCW_Step_Sizes_Id

		Fast_Phi_Joint2_CW_Region_Count_Id
		Fast_Phi_Joint2_CW_Regions_Id
		Fast_Phi_Joint2_CW_Step_Sizes_Id

		Fast_Phi_Joint2_CCW_Region_Count_Id
		Fast_Phi_Joint2_CCW_Regions_Id
		Fast_Phi_Joint2_CCW_Step_Sizes_Id

		Slow_Theta_Joint1_CW_Region_Count_Id
		Slow_Theta_Joint1_CW_Regions_Id
		Slow_Theta_Joint1_CW_Step_Sizes_Id

		Slow_Theta_Joint1_CCW_Region_Count_Id
		Slow_Theta_Joint1_CCW_Regions_Id
		Slow_Theta_Joint1_CCW_Step_Sizes_Id

		Slow_Phi_Joint2_CW_Region_Count_Id
		Slow_Phi_Joint2_CW_Regions_Id
		Slow_Phi_Joint2_CW_Step_Sizes_Id

		Slow_Phi_Joint2_CCW_Region_Count_Id
		Slow_Phi_Joint2_CCW_Regions_Id
		Slow_Phi_Joint2_CCW_Step_Sizes_Id


	struct database_info:
		uint32_t Positioner_serial_number

		double Base_Position_X
		double Base_Position_Y
		double Base_Orientation

		double Theta_Joint1_CW_Angle_limit
		double Theta_Joint1_CCW_Angle_limit

		double Phi_Joint2_CW_Angle_limit
		double Phi_Joint2_CCW_Angle_limit

		double Theta_Joint1_transition_angle
		double Phi_Joint2_transition_angle

		double Theta_Joint1_link_length
		double Phi_Joint2_link_length

		double Phi_Joint2_Base_Radius
		double Phi_Joint2_Tip_Radius

		double Length_uncertainty

		double Theta_Joint1_Frequency
		double Phi_Joint2_Frequency

		uint32_t Fast_Theta_Joint1_CW_Region_Count
		double Fast_Theta_Joint1_CW_Regions[CAL_COL_SIZE]
		double Fast_Theta_Joint1_CW_Step_Sizes[CAL_COL_SIZE]

		uint32_t Fast_Theta_Joint1_CCW_Region_Count
		double Fast_Theta_Joint1_CCW_Regions[CAL_COL_SIZE]
		double Fast_Theta_Joint1_CCW_Step_Sizes[CAL_COL_SIZE]

		uint32_t Fast_Phi_Joint2_CW_Region_Count
		double Fast_Phi_Joint2_CW_Regions[CAL_COL_SIZE]
		double Fast_Phi_Joint2_CW_Step_Sizes[CAL_COL_SIZE]

		uint32_t Fast_Phi_Joint2_CCW_Region_Count
		double Fast_Phi_Joint2_CCW_Regions[CAL_COL_SIZE]
		double Fast_Phi_Joint2_CCW_Step_Sizes[CAL_COL_SIZE]

		uint32_t Slow_Theta_Joint1_CW_Region_Count
		double Slow_Theta_Joint1_CW_Regions[CAL_COL_SIZE]
		double Slow_Theta_Joint1_CW_Step_Sizes[CAL_COL_SIZE]

		uint32_t Slow_Theta_Joint1_CCW_Region_Count
		double Slow_Theta_Joint1_CCW_Regions[CAL_COL_SIZE]
		double Slow_Theta_Joint1_CCW_Step_Sizes[CAL_COL_SIZE]

		uint32_t Slow_Phi_Joint2_CW_Region_Count
		double Slow_Phi_Joint2_CW_Regions[CAL_COL_SIZE]
		double Slow_Phi_Joint2_CW_Step_Sizes[CAL_COL_SIZE]

		uint32_t Slow_Phi_Joint2_CCW_Region_Count
		double Slow_Phi_Joint2_CCW_Regions[CAL_COL_SIZE]
		double Slow_Phi_Joint2_CCW_Step_Sizes[CAL_COL_SIZE]

