import torch
import serial
import numpy as np
from preprocess import preprocess
# 加载训练好的模型
model = torch.load('.\BEST_MODELS\xxx')
model.eval()

def preprocess(data):

    data = preprocess(data)
    return data

def receive_eeg_data(ser):
    # upload data
    if ser.in_waiting:
        data = ser.readline()  # a sample

        # Assuming PGA = 24, VREF = 2.4V, and a hypothetical raw data sequence
        PGA = 24
        VREF = 2.4
        EEGVoltage = calculate_raw(PGA, VREF, data)

        return EEGVoltage
    return None



def calculate_raw(PGA, VREF, EEG_raw):
    """
    Calculate the original EEG voltage value based on the ADS1299 data format.
    :param PGA: Programmable Gain Amplifier value.
    :param VREF: Reference Voltage.
    :param EEG_raw: Raw EEG data, expected to be a 3-byte sequence.
    :return: EEG voltage in microvolts.
    """
    # Extract bytes A, B, C from the raw data
    A, B, C = EEG_raw

    # Combine the bytes into a single 24-bit number
    VIN = A * 65536 + B * 256 + C

    # Calculate the scaling factor
    k = VREF / PGA / (2 ** 23 - 1) * 10 ** 6

    # Determine and calculate the voltage based on the sign
    if A <= 127:  # Positive voltage
        EEGVoltage = k * VIN
    else:  # Negative voltage
        VIN = ~VIN & 0xFFFFFF  # Two's complement
        VIN = VIN & 0x7FFFFF  # Mask to 23 bits
        EEGVoltage = -k * VIN

    return EEGVoltage


def classify_eeg_data(ser):
    while True:
        eeg_data = receive_eeg_data(ser)
        if eeg_data is not None:
            eeg_data = preprocess(eeg_data)
            eeg_data_tensor = torch.from_numpy(eeg_data).float()
            with torch.no_grad():
                prediction = model(eeg_data_tensor)

            # result
            print("分类结果: ", prediction)

# serial setting
serial_port = '/dev/ttyUSB0'  # number of the port
baud_rate = 9600  # Baud rate

# initial serial
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# run the model
classify_eeg_data(ser)

# close serial
ser.close()

