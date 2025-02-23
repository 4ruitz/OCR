import snap7
from snap7.util import set_int

# IP address of the Siemens PLC
plc_ip = '192.168.0.1'  # Replace with your PLC's IP address

# Define the DB number and starting byte offset (DB1, for example)
db_number = 1
start_offset = 0  # Start from byte 0 in DB1

# Integer value to send
int_value = 12345  # Replace with your desired integer value

def send_integer_to_plc(plc_ip, db_number, start_offset, int_value):
    # Connect to the PLC
    plc = snap7.client.Client()
    plc.connect(plc_ip, 0, 1)  # 0 and 1 are rack and slot for S7-1200
    
    # Write the integer to the datablock (DB)
    data = bytearray(2)  # Integers take 2 bytes
    set_int(data, 0, int_value)
    
    # Write to the datablock
    plc.db_write(db_number, start_offset, data)
    
    # Disconnect from the PLC
    plc.disconnect()
    print(f"Integer {int_value} written to DB{db_number} at offset {start_offset}.")

# Call the function
send_integer_to_plc(plc_ip, db_number, start_offset, int_value)
