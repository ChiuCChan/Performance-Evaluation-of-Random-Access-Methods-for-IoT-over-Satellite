# Performance-Evaluation-of-Random-Access-Methods-for-IoT-over-Satellite

Please cite the code which is part of our publication:

Chan, C.C.; Al Homssi, B.; Al-Hourani, A. Performance Evaluation of Random Access Methods for IoT-over-Satellite. Remote Sens. 2022, 14, 4232. 

Link on MDPI: [https://www.mdpi.com/2072-4292/14/17/4232]
# Requirement

This emulation framework is using the LoRaMatlab library that we developed.

https://www.mathworks.com/matlabcentral/fileexchange/81166-loramatlab

# How to use?

1. Open the "SatelliteOnDutyCalculation.m" to obtain the on-duty table of satellites with respect to the local observer.

2. User can change the constellation parameter {N, inc, P, F, h, WalkerDelta in [True, False]}	
   User can also change the satellite antenna beamwidth (beamwidth) to alter the coverage of satellites and therefore to limit the number of avaliable satellites.
   
3. Open "TheEmulator_Case#.m" to emulate the LoRa-over-Satellite communication. 
	Case 1: Random Access
	Case 2: Scheduled Access (when there is any avaliable satellite links)
	Case 3: Scheduled Access with Packet Reptition.
 	Case 4: Round-Robin Scheduled Access (Interference-free)
   User can change the SF, 
   
4. The NB-IoT-over-Satellite emulation script is also provided. It requires the MATLAB LTE toolbox to be pre-installed. 

# Output

Packet error rate: PER

# Contribute

The emulation framework is programmed by Chiu Chun Chan.

The LoRa emulator is programmed by Bassel Al Homssi.

The Satellite-to-Ground pathloss function is programmed by Akram Al-Hourani.

# License

Electronic & Telecommunication department, RMIT University, hereby disclaims all copyright interest in the program "Performance-Evaluation-of-Random-Access-Methods-for-IoT-over-Satellite" written by Chiu Chun Chan, Bassel Al Homssi, Akram Al-Hourani.
