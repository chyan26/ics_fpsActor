/*************************************************************
* Designed By: Hrand Aghazarian
* JPL - 2015
* File: pfi_interface_defs.h
* Version: 1.0
*************************************************************/

/*-----------------------------------------------------------*/
/*------ The following three lines are added by ChihYi ------*/
/*-----------------------------------------------------------*/

typedef uint32_t UInt32;
typedef double Float64;
typedef char UChar;


/*-----------------------------------------------------------*/
/*------------------------ Includes -------------------------*/
/*-----------------------------------------------------------*/




/*-----------------------------------------------------------*/
/*------------------------ Defines --------------------------*/
/*-----------------------------------------------------------*/


#define STATUS_OK		(0)
#define CMD_EXEC_OK		(1)
#define CMD_EXEC_FAILED	(2)
#define e100			(100)//error
#define e101			(101)//error
#define e102			(102)//error
#define e103			(103)//error
#define e104			(104)//error






//////////////////////////////////////////////////////


//
#define MAX_RECORD_SIZE	(2394)



//Move_To_Target command bit positions
#define MOVE_THETA				(0)//joint 1
#define MOVE_PHI				(1)//joint 2
#define OBTSTACLE_AVOIDNACE		(2)


#define	MAX_XML_FILE_NAME_SIZE	(1024*10)

#define	MAX_ERROR_STRING_SIZE	(1024*64)


#define SET_DATABASE_DATA_MAX_SIZE	(1024*1024*6)


#define CAL_COL_SIZE	(200)//must match define in pfi_arm_ctrl.h


//Go_Home_All flags bit positions
#define Go_Home_All_OBSTACLE_AVOIDANCE_BIT	(0) //
#define Go_Home_All_USE_FAST_MAP_BIT		(1) //





//Flag Bits
#define BIT_Radians				0x00000001//1
#define BIT_Degree				0x00000002//2
#define BIT_Step_Count			0x00000004//3
#define BIT_Move_Theta			0x00000008//4
#define BIT_Move_Phi			0x00000010//5
#define BIT_Target_Latched		0x00000020//6
#define BIT_Fixed_Positioner	0x00000040//7
#define BIT_Obstacle_Avoidance	0x00000080//8
#define BIT_Theta_Use_Fast_Map	0x00000100//9
#define BIT_Phi_Use_Fast_Map	0x00000200//10
#define BIT_Save_The_Database	0x00000400//11
#define BIT_Shut_Down			0x00000800//12
#define BIT_Re_Start			0x00001000//13
#define BIT_Enable_Blind_Move	0x00002000//14

#define BIT_Delay_Theta_Joint1	0x00004000//15 - REV_D
#define BIT_Delay_Phi_Joint2	0x00008000//16 - REV_D

#define BIT_Target_Changed		0x00010000//17
#define BIT_Theta_Joint1_Delayed	0x00020000//18
#define BIT_Phi_Joint2_Delayed	0x00040000//19
#define BIT_Arm_Collided		0x00080000//20
#define BIT_Arm_Disabled		0x00100000//21
#define BIT_Kinematics_Failed	0x00200000//22
#define BIT_DB_Invalid			0x00400000//23
#define BIT_Get_Fresh_HK_Data	0x00800000//24








/////////////////////
//Pfi command to MPS IDS
#define Move_To_Target_ID					(200)
#define Go_Home_All_ID						(201)//not used anymore
#define Go_Home_Positioner_ID				(202)
#define Move_Positoner_ID					(203)
#define Calibrate_Motor_Frequencies_ID		(204)
#define Get_Status_Command_ID				(205)
#define Get_Database_Data_ID				(206)
#define Set_Database_Data_ID				(207)
#define Set_Database_Field_ID				(208)
#define Get_Telemetry_Data_ID				(209)
#define Export_Database_To_Xml_File_ID		(210)
#define Import_Database_From_Xml_File_ID	(211)
#define Set_Current_Position_Data_ID		(212)
#define Move_Positoner_Interval_Duration_ID	(213)
#define Mps_Software_ID						(214)
#define Move_Positoner_With_Delay_ID		(215)
#define Set_HardStop_Orientation_ID			(216)

#define Set_Power_or_Reset_ID				(217)
#define Run_Diagnostic_ID					(218)


/////////////////////
//MPS response to Pfi IDS
#define Command_Response_ID		(300)
#define Send_Database_Data_ID	(301)
#define Send_Telemetry_Data_ID	(302)
#define Send_Diagnostic_Result_Telemetry_Data_ID	(303)











/*-----------------------------------------------------------*/
/*----------------------- Variables -------------------------*/
/*-----------------------------------------------------------*/

#pragma pack(4)

//common types

//////////////////////////////////////////////////////////////
//Command Header


struct command_header
{
	UInt32 Command_Id;
	UInt32 Message_Size;
	UInt32 Time_Stamp1;//seconds
	UInt32 Time_Stamp2;//milliseconds
	UInt32 Command_Counter;
	UInt32 Flags;
};

struct number_of_records
{
	UInt32	Number_Of_Records;
//	UInt32	Flags;
};



struct xml_file_info
{
	UInt32	Name_Length;
	UChar	Name[MAX_XML_FILE_NAME_SIZE];
};


//////////////////////////////////////////////////////////////
//Move_To_Target Command


struct move_to_target_msg_header
{
	UInt32 Sequnce_Number;
	UInt32 Iteration_Number;
	UInt32 Target_Number;
	UInt32 Number_Of_Records;
	UInt32 Flags;
};

struct move_to_target_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	Float64	Current_Position_X;
	Float64	Current_Position_Y;
	Float64 Target_Position_X;
	Float64 Target_Position_Y;
	Float64 X_axes_Uncertainty;
	Float64 Y_axes_Uncertainty;
	Float64	Joint1_Delay_Count;
	Float64	Joint2_Delay_Count;

	UInt32	Flags;
};

struct move_to_target_command
{
	struct command_header				Command_Header;
	struct move_to_target_msg_header	Msg_Header;
	struct move_to_target_msg_record	Msg_Record[MAX_RECORD_SIZE];
	UInt32								Msg_Record_Count;
};


/////////////////////////////////////////////////////////////
//Go_Home_All Command

//not used anymore
struct go_home_all_msg_record
{
	UInt32	Flags;//bit1: Enable Obstacle avoidance
};

//not used anymore
struct go_home_all_command
{
	struct command_header			Command_Header;
	struct go_home_all_msg_record	Msg_Record;
};



/////////////////////////////////////////////////////////////
//Go_Home_Positioner Command
//not used anymore
struct go_home_positioner_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	UInt32	Flags;
};
//not used anymore
struct go_home_positioner_command
{
	struct command_header					Command_Header;
	struct number_of_records				Msg_Header;
	struct go_home_positioner_msg_record	Msg_Record[MAX_RECORD_SIZE];
};




/////////////////////////////////////////////////////////////
//Move_Positioner Command


struct move_positioner_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	Float64	Theta_Joint1;
	Float64	Phi_Joint2;
	UInt32	Flags;
};


struct move_positioner_command
{
	struct command_header				Command_Header;
	struct number_of_records			Msg_Header;
	struct move_positioner_msg_record	Msg_Record[MAX_RECORD_SIZE];
};





/////////////////////////////////////////////////////////////
//Move_Positioner_Interval_Duration Command



struct move_positioner_int_dur_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	Float64	Theta_Joint1;
	Float64	Theta_Joint1_Interval;
	Float64	Theta_Joint1_Duration;
	Float64	Phi_Joint2;
	Float64	Phi_Joint2_Interval;
	Float64	Phi_Joint2_Duration;
	UInt32	Flags;
};


struct move_positioner_int_dur_command
{
	struct command_header						Command_Header;
	struct number_of_records					Msg_Header;
	struct move_positioner_int_dur_msg_record	Msg_Record[MAX_RECORD_SIZE];
};




/////////////////////////////////////////////////////////////
//Calibrate_Motor_Frequencies Command




struct calibrate_motor_frequencies_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	Float64	Theta_Joint1_Start_Frequency;
	Float64	Theta_Joint1_End_Frequency;
	Float64	Phi_Joint2_Start_Frequency;
	Float64	Phi_Joint2_End_Frequency;
	UInt32	Flags;

};


struct calibrate_motor_frequencies_command
{
	struct command_header							Command_Header;
	struct number_of_records						Msg_Header;
	struct calibrate_motor_frequencies_msg_record	Msg_Record[MAX_RECORD_SIZE];
};



/////////////////////////////////////////////////////////////
//Get_Status Command



struct get_status_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
};


struct get_status_command
{
	struct command_header			Command_Header;
	struct number_of_records		Msg_Header;
	struct get_status_msg_record	Msg_Record[MAX_RECORD_SIZE];
};



////////////////////////////////////////////////////////////
//Set_Database_Data Command


//  later - later



struct number_of_database_items
{
	UInt32	Number_Of_Items;
	UInt32	Save_To_Database; //1=save, 2=do not save
};



////////////////////////////////////////////////////////////
//Set_Database_Field Command


struct set_database_field_msg_header
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	UInt32	Number_Of_Set_Fields;
};


struct set_database_field_id
{
	UInt32	Field_Id;
	//Field data, Depends on the field
};




////////////////////////////////////////////////////////////
//Get_Database_Data


struct get_database_data_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
};

struct get_database_data_command
{
	struct command_header			Command_Header;
	struct number_of_records		Msg_Header;
	struct get_database_data_record	Msg_Record[MAX_RECORD_SIZE];
};


////////////////////////////////////////////////////////////
//Set_Database_Data


struct set_database_data_record
{
	UInt32	Xml_File_Size;
	UChar	Xml_File_Data[SET_DATABASE_DATA_MAX_SIZE];
};

struct set_database_data_command
{
	struct command_header				Command_Header;
	struct set_database_data_record		Msg_Record;
};





////////////////////////////////////////////////////////////
//Get_Telemetry_Data



struct get_telemetry_data_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
};


struct get_telemetry_data_command
{
	struct command_header				Command_Header;
	struct number_of_records			Msg_Header;
	struct get_telemetry_data_record	Msg_Record[MAX_RECORD_SIZE];
};





////////////////////////////////////////////////////////////
//export_Database_To_Xml_File


struct export_database_to_xml_file_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
};


struct export_database_to_xml_file_command
{
	struct command_header						Command_Header;
	struct number_of_records					Msg_Header;
	struct xml_file_info						FileInfo;
	struct export_database_to_xml_file_record	Msg_Record[MAX_RECORD_SIZE];
};



////////////////////////////////////////////////////////////
//import_Database_From_Xml_File

struct import_database_from_xml_file_command
{
	struct command_header						Command_Header;
	struct xml_file_info						FileInfo;
};




/////////////////////////////////////////////////////////////
//Move_Positioner with delay Command


struct move_positioner_with_delay_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	Float64	Theta_Joint1;
	Float64	Phi_Joint2;
	Float64	Delay_Theta_Joint1;
	Float64	Delay_Phi_Joint2;
	UInt32	Flags;
};


struct move_positioner_with_delay_command
{
	struct command_header							Command_Header;
	struct number_of_records						Msg_Header;
	struct move_positioner_with_delay_msg_record	Msg_Record[MAX_RECORD_SIZE];
};




////////////////////////////////////////////////////////////
//Set_Current_Position_Data_ID




struct current_positioner_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	Float64	Position_X;
	Float64	Position_Y;
	UInt32	Flags;
};


struct set_current_position_data_command
{
	struct command_header					Command_Header;
	struct number_of_records				Msg_Header;
	struct current_positioner_msg_record	Msg_Record[MAX_RECORD_SIZE];
};




////////////////////////////////////////////////////////////
struct mps_software_record
{
	UInt32	Command_Flags;
};


struct mps_software_command
{
	struct command_header		Command_Header;
	struct mps_software_record	Msg_Record;
};






////////////////////////////////////////////////////////////
//Get_Telemetry_Data_ID


////////////////////////////////////////////////////////////
//Set_HardStop_Orientation


struct hardstop_orientation_msg_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	UInt32	HardStop_Ori;//ccw=0, cw=1

};


struct set_hardstop_orientation_data_command
{
	struct command_header					Command_Header;
	struct number_of_records				Msg_Header;
	struct hardstop_orientation_msg_record	Msg_Record[MAX_RECORD_SIZE];
};




////////////////////////////////////////////////////////////
//Turn_Sectors_Power_On_or_Off_ID
#define PowerOn				(1)
#define PowerReset			(2)
#define NUM_Power_Sectors	(6)


struct set_sectors_power_or_reset_msg_record
{
	UInt32	Command_Type;
	UInt32	Enable_Set_Motor_Frequencis;
	UInt32	Sectors[NUM_Power_Sectors];
};


struct set_sectors_power_or_reset_data_command
{
	struct command_header							Command_Header;
	struct set_sectors_power_or_reset_msg_record	Msg_Record;
};


////////////////////////////////////////////////////////////


struct diagnostic_command
{
	struct command_header	Command_Header;
};




///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
//Execution Time data structure


struct execution_time_data
{
	Float64		Mps_Command_Receive_Time;

	Float64		FPGA_Command_Send_Time;
	Float64		FPGA_Command_Response1_Time;
	Float64		FPGA_Command_Response2_Time;

	Float64		Mps_Command_Response_Time;

};


extern struct execution_time_data	Exection_Time_Data;



///////////////////////////////////////////////////////////////
//Command_Response Response


struct command_response_error
{
	UInt32	StatusNumber;
	UInt32	Error_String_Size;
	UChar	Error_String[MAX_ERROR_STRING_SIZE];//NULL terminated text
};


struct command_response
{
	struct command_header			Response_Header;
	struct command_response_error	MSG_Body;
};


////////////////////////////////////////////////////////////
//Send_Telemetry_Data


struct send_telemetry_data_record
{
	UInt32	Module_Id;
	UInt32	Positioner_Id;
	UInt32	Flags;

	UInt32	Target_Number;
	UInt32	Iteration_number;

	Float64	Target_X;
	Float64	Target_Y;
	Float64	Target_Joint1_Angle;
	Float64	Target_Joint2_Angle;
	UInt32	Target_Joint1_Quadrant;
	UInt32	Target_Joint2_Quadrant;

	Float64	Current_X;
	Float64	Current_Y;
	Float64	Current_Join1_Angle;
	Float64	Current_Join2_Angle;
	UInt32	Current_Joint1_Quadrant;
	UInt32	Current_Joint2_Quadrant;
	UInt32	Joint1_Step;
	UInt32	Joint2_Step;

	Float64	Join1_Step_Delay;
	Float64	Join2_Step_Delay;

	UInt32	Target_Changed;
	Float64	Target_Changed_X;
	Float64	Target_Changed_Y;

	UInt32	HardStop_Ori;
	Float64	Joint1_from_Angle;
	Float64	Joint1_to_Angle;

	//FPGA data from recent HK
	UInt32	FPGA_Board_Number;
	Float64	FPGA_Temp1;
	Float64	FPGA_Temp2;
	Float64	FPGA_Voltage;

	Float64	FPGA_Join1_Frequency;
	Float64	FPGA_Join1_Current;
	Float64	FPGA_Join2_Frequency;
	Float64	FPGA_Join2_Current;

};


struct send_telemetry_data_command
{
	struct command_header				Command_Header;
	struct execution_time_data			Execution_Time_Data;
	struct number_of_records			Msg_Record_Count;
	struct send_telemetry_data_record	Msg_Record[MAX_RECORD_SIZE];
};



//////////////////////////////////////////////////////////////////////////////
//Send_Diagnostic_Result_Telemetry_Data_ID

#define NumberOfSectors (6)
struct send_diagnostic_telemetry_record
{
	UInt32	Response_Code;
	UInt32	Detail_Message;
	UInt32	Sectors[NumberOfSectors];
};

struct send_diagnostic_telemetry_data
{
	struct command_header					Command_Header;
	struct execution_time_data				Execution_Time_Data;
	struct send_diagnostic_telemetry_record	Msg_Record;
};








//////////////////////////////////////////////////////////////////////////////
//DataBase Ids


#define	Positioner_serial_number_Id			(1)

#define	Base_Position_X_Id					(2)
#define	Base_Position_Y_Id					(3)
#define	Base_Orientation_Id					(4)

#define	Theta_Joint1_CW_Angle_limit_Id		(5)
#define	Theta_Joint1_CCW_Angle_limit_Id		(6)

#define	Phi_Joint2_CW_Angle_limit_Id		(7)
#define	Phi_Joint2_CCW_Angle_limit_Id		(8)

#define	Theta_Joint1_transition_angle_Id	(9)
#define	Phi_Joint2_transition_angle_Id		(10)

#define Theta_Joint1_link_length_Id			(11)
#define Phi_Joint2_link_length_Id			(12)

#define	Phi_Joint2_Base_Radius_Id			(13)
#define	Phi_Joint2_Tip_Radius_Id			(14)

#define Length_uncertainty_Id				(15)

#define	Theta_Joint1_Frequency_Id			(16)
#define Phi_Joint2_Frequency_Id				(17)



#define Fast_Theta_Joint1_CW_Region_Count_Id	(18)
#define Fast_Theta_Joint1_CW_Regions_Id			(19)
#define	Fast_Theta_Joint1_CW_Step_Sizes_Id		(20)

#define	Fast_Theta_Joint1_CCW_Region_Count_Id	(21)
#define Fast_Theta_Joint1_CCW_Regions_Id		(22)
#define Fast_Theta_Joint1_CCW_Step_Sizes_Id		(23)

#define Fast_Phi_Joint2_CW_Region_Count_Id		(24)
#define Fast_Phi_Joint2_CW_Regions_Id			(25)
#define Fast_Phi_Joint2_CW_Step_Sizes_Id		(26)

#define	Fast_Phi_Joint2_CCW_Region_Count_Id		(27)
#define Fast_Phi_Joint2_CCW_Regions_Id			(28)
#define Fast_Phi_Joint2_CCW_Step_Sizes_Id		(29)



#define Slow_Theta_Joint1_CW_Region_Count_Id	(30)
#define Slow_Theta_Joint1_CW_Regions_Id			(31)
#define	Slow_Theta_Joint1_CW_Step_Sizes_Id		(32)

#define	Slow_Theta_Joint1_CCW_Region_Count_Id	(33)
#define Slow_Theta_Joint1_CCW_Regions_Id		(34)
#define Slow_Theta_Joint1_CCW_Step_Sizes_Id		(35)

#define Slow_Phi_Joint2_CW_Region_Count_Id		(36)
#define Slow_Phi_Joint2_CW_Regions_Id			(37)
#define Slow_Phi_Joint2_CW_Step_Sizes_Id		(38)

#define	Slow_Phi_Joint2_CCW_Region_Count_Id		(39)
#define Slow_Phi_Joint2_CCW_Regions_Id			(40)
#define Slow_Phi_Joint2_CCW_Step_Sizes_Id		(41)





struct database_info
{
	UInt32	Positioner_serial_number;

	Float64	Base_Position_X;
	Float64	Base_Position_Y;
	Float64	Base_Orientation;

	Float64	Theta_Joint1_CW_Angle_limit;
	Float64	Theta_Joint1_CCW_Angle_limit;

	Float64	Phi_Joint2_CW_Angle_limit;
	Float64	Phi_Joint2_CCW_Angle_limit;

	Float64	Theta_Joint1_transition_angle;
	Float64	Phi_Joint2_transition_angle;

	Float64 Theta_Joint1_link_length;
	Float64 Phi_Joint2_link_length;

	Float64	Phi_Joint2_Base_Radius;
	Float64	Phi_Joint2_Tip_Radius;

	Float64 Length_uncertainty;

	Float64	Theta_Joint1_Frequency;
	Float64 Phi_Joint2_Frequency;



	UInt32  Fast_Theta_Joint1_CW_Region_Count;
	Float64 Fast_Theta_Joint1_CW_Regions[CAL_COL_SIZE];
	Float64	Fast_Theta_Joint1_CW_Step_Sizes[CAL_COL_SIZE];

	UInt32	Fast_Theta_Joint1_CCW_Region_Count;
	Float64 Fast_Theta_Joint1_CCW_Regions[CAL_COL_SIZE];
	Float64 Fast_Theta_Joint1_CCW_Step_Sizes[CAL_COL_SIZE];

	UInt32  Fast_Phi_Joint2_CW_Region_Count;
	Float64 Fast_Phi_Joint2_CW_Regions[CAL_COL_SIZE];
	Float64 Fast_Phi_Joint2_CW_Step_Sizes[CAL_COL_SIZE];

	UInt32	Fast_Phi_Joint2_CCW_Region_Count;
	Float64 Fast_Phi_Joint2_CCW_Regions[CAL_COL_SIZE];
	Float64 Fast_Phi_Joint2_CCW_Step_Sizes[CAL_COL_SIZE];



	UInt32 Slow_Theta_Joint1_CW_Region_Count;
	Float64 Slow_Theta_Joint1_CW_Regions[CAL_COL_SIZE];
	Float64	Slow_Theta_Joint1_CW_Step_Sizes[CAL_COL_SIZE];

	UInt32	Slow_Theta_Joint1_CCW_Region_Count;
	Float64 Slow_Theta_Joint1_CCW_Regions[CAL_COL_SIZE];
	Float64 Slow_Theta_Joint1_CCW_Step_Sizes[CAL_COL_SIZE];

	UInt32 Slow_Phi_Joint2_CW_Region_Count;
	Float64 Slow_Phi_Joint2_CW_Regions[CAL_COL_SIZE];
	Float64 Slow_Phi_Joint2_CW_Step_Sizes[CAL_COL_SIZE];

	UInt32	Slow_Phi_Joint2_CCW_Region_Count;
	Float64 Slow_Phi_Joint2_CCW_Regions[CAL_COL_SIZE];
	Float64 Slow_Phi_Joint2_CCW_Step_Sizes[CAL_COL_SIZE];


};









/*-----------------------------------------------------------*/
/*----------------------- Functions -------------------------*/
/*-----------------------------------------------------------*/
