import simpy
import numpy as np
import random
import pandas as pd
import os

def gNB(env, mnPower, timeResolution=31.25, totalNetworkSlices=1, printLogs=False):
	
	# gNB Configuration.
	global totalDev
	global fixedR
	global totalRegisteredDevices
	global totalEnergyGNB
	global collisionList
	global collisionListBand24
	global commonDownlinkBW
	global commonUplinkBW
	global downlinkRRC
	global uplinkRRC
	global downlinkBW
	global uplinkBW
	global randomValueForAccess

	# Default random value for the ACB algorithm.
	randomValueForAccess = 0.8

	# All pending RARs that need to be sent by the gNB to the requesters.
	RACHPendingList = []

	# All pending RRCs that need to be sent by the gNB to the requesters.
	RRCPendingList = []

	# 5000us = 5ms Mminimum periodicity of SIB1 {5ms, 10ms, 20ms, 40ms, 80ms, 160ms}.
	SIB1Periodicity = 5*1000
	
	global numerologySlot
	# 66.67us -> slot time for 15kHz numerology.
	numerologySlot = 66.67
	# Timestamp to be accountable to add 1 to the current slot.
	numerologySlotTimer = 0

	# Minimum ra-responseWindow interval {sl1, sl2, sl4, sl8, sl10, sl20, sl40, sl80}.
	TTLRar = numerologySlot

	# Initializing the resource allocation table.
	resourceAllocationTable = np.array(('RA-RNTI', 'T/C-RNTI', 'RB_dl', 'SC_dl', 'L_dl', \
										'S_dl', 'RB_ul', 'SC_ul', 'L_ul', 'S_ul', 'TTL'))
	resourceAllocationTable = np.reshape(resourceAllocationTable, \
								(1, len(resourceAllocationTable)))
	
	# Global to be used by every device (14 symbols).
	global symbolCounterMN
	# This counter is used to count 14 symbols and restart again, this is a time-domain counter.
	symbolCounterMN = 0

	# Global to be used by every device (80 slots).
	global slotCounterMN
	# This counter is used to count 80 slots and restart again, this is a time-domain counter.
	slotCounterMN = 0

	# Timestamp to be accountable to send or not the SIB1 message.
	SIB1Timer = 0

	# gNB Tasks.
	# Executes the gNB tasks while there are no registered devices.
	while totalRegisteredDevices < totalDev + fixedR:
		# Stores the list of TC-RNTI identifiers received from the requesters.
		# This is used to update the TC-RNTI identifiers list after every time
		# step in case some TC-RNTI expired and the list changed.
		newTC_RNTIList = [int(tc.split('|')[-1]) for tc in RACHPendingList]
		
		# Sends SIB1 every 5ms.
		if env.now - SIB1Timer >= SIB1Periodicity:
			# Timestamp which was sent the last SIB1 message.
			SIB1Timer = env.now

			# Prints the timestamp which the gNB sends SIB1.
			if printLogs == True: print(env.now, 'us: gNB -> Sending SIB1')

			# Add 'mnPower' to the total energy spent in the mobile network band by the gNB.
			totalEnergyGNB += mnPower

			# Sending 64 preambles within SIB1.
			commonDownlinkBW[0] = [i for i in range(64)]

		# Checks common uplink bandwidth to find Random-Access requests.
		# This PRACH requests are only part of the traditional Random-Access procedure.
		randomValueForAccessC = 0
		for channelIndex in range(len(commonUplinkBW)):
			
			# If there is a transmission in the common uplink bandwidth,
			# the gNB receives the message.
			if commonUplinkBW[channelIndex] != None:
				randomValueForAccessC += 1
				# Random-Access request data has the format: RA-RNTI|preamble.
				# the '0' identifier lets the gNB knows that this requester
				# data was acquired by using the traditional Random-Access procedure.
				RA_RNTI = commonUplinkBW[channelIndex].split('|')[0] + '_0'

				TC_RNTI_List = []
				# Saving all TC_RNTI (from the resource allocation table) in a list to compare later.
				for lookup in range(1, resourceAllocationTable.shape[0]):
					TC_RNTI_List.append(resourceAllocationTable[lookup, 1])
				
				# Assigning a new TC-RNTI (that it was not assigned before) to the new requester (the
				# requester that sent the RAR message).
				TC_RNTI_Status = False
				while TC_RNTI_Status == False:
					TC_RNTI = random.randint(0, 65535) # random number in interval {0, 65535}.

					# If the computed TC-RNTI is not in any list, it can be assigned to the new requester.
					if str(TC_RNTI) not in TC_RNTI_List and TC_RNTI not in newTC_RNTIList:
						TC_RNTI_Status = True
						newTC_RNTIList.append(TC_RNTI)

				# Assigns a Time-To-Live time to the new computed TC-RNTI.
				# If this TTL expires, the TC-RNTI is erased.
				TTL = env.now

				# Adding the new device to the resource allocation table.
				# However, it still has not resources assigned.
				new_requester_row = np.array((RA_RNTI, TC_RNTI, \
					'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', TTL))
				resourceAllocationTable = np.vstack([resourceAllocationTable, new_requester_row])
				
				# The gNB will send back the RAR to the new requester in the same channel the gNB received
				# the PRACH request (MSG2). The message contains the RA-RNTI to identify the requester that sent
				# the PRACH request and the message also contains the computed TC-RNTI to be assigned to
				# the requester.
				RACHPendingList.append(RA_RNTI + '|' + commonUplinkBW[channelIndex].split('|')[1] + '|' + str(TC_RNTI))

		# This means that the gNB found a PRACH request from a new requester.		
		if randomValueForAccessC > 0:
			# Adaptates the randomValueForAccess in the interval 0.2-0.8, 
			# if there are 64 received preambles the randomValueForAccess is equal to 0.2, 
			# if there are 0 received preambles the randomValueForAccess is equal to 0.8.
			# The randomValueForAccess is the value sent by the gNB in SIB1 to perform
			# the ACB algorithm and reduce collisions between requesters during access.
			randomValueForAccess = 1-(0.6/64*randomValueForAccessC + 0.2)

			# Prints the timestamp when the gNB receives at least one PRACH request, the number of MSG2
			# received, and the random value that will be used for the ACB algorithm.
			print(env.now, 'gNB -> ', randomValueForAccessC, \
				' Preambles Received - New Random Access Value=' + str(randomValueForAccess))

		# Checks uplink to find forwarded message by a relay from a requester.
		# This is only available for the Framework RAA procedure.
		# If there are registered relays in the resource allocation table...
		if resourceAllocationTable.shape[0] > 1:
			
			# For each possible relay...
			data_to_concatenate_with_RAT = []
			for lookup in range(1, resourceAllocationTable.shape[0]):
				
				# If this device has resources allocated for uplink... 
				if resourceAllocationTable[lookup, 7] != 'None':
					# Gets SC_ul (uplink subcarrier) where the gNB will look up for relay transmission.
					subcarrierToLookUpInUplink = int(resourceAllocationTable[lookup, 7].split('-')[0])
					
					# If there is data from this relay...
					if len(uplinkBW) > 0 and uplinkBW[subcarrierToLookUpInUplink] != None:
						# Data has the format: RA-RNTI|SC_dl|SC_ul|NoSym_dl|NoSym_ul.

						# Saves only data not repeated since more than one relay could forward the same
						# message from the same requester.
						if uplinkBW[subcarrierToLookUpInUplink] not in data_to_concatenate_with_RAT:
							# Saves every relay transmission to concatenate with resource allocation table 
							# at the end of this iteration.
							data_to_concatenate_with_RAT.append(uplinkBW[subcarrierToLookUpInUplink])
			
			# If there are transmissions from relays saved...
			if len(data_to_concatenate_with_RAT) > 0:
				# Every new data from requester to include at the end of the resource allocation table.
				for data in data_to_concatenate_with_RAT:
					
					# Looks if the current requester identified by the current RA-RNTI is not
					# in the resource allocation table.
					for lookup in range(1, resourceAllocationTable.shape[0]):
						# If the requester (RA-RNTI) is in the resource allocation table, skips.
						if data.split('|')[0] == resourceAllocationTable[lookup, 0]:
							break
					# If this RA-RNTI is not in resource allocation table includes 
					# this requester data in resource allocation table.
					else:
						# Saves the requester in the resource allocation table and adds
						# the '1' identifier to let the gNB knows that this requester
						# data was acquired via relay (by using the Framework RAA).
						RA_RNTI = data.split('|')[0] + '_1'
						
						TC_RNTI_List = []
						# Saving all TC_RNTI (from the resource allocation table) in a list to compare later.
						for lookup in range(1, resourceAllocationTable.shape[0]):
							TC_RNTI_List.append(resourceAllocationTable[lookup, 1])

						# Assigning a new TC-RNTI (that it was not assigned before) to the new requester (the
						# requester that sent the message via relay).	
						TC_RNTI_Status = False
						while TC_RNTI_Status == False:
							TC_RNTI = random.randint(0, 65535) # random number in interval {0, 65535}

							# If the computed TC-RNTI is not in any list, it can be assigned to the new requester.
							if str(TC_RNTI) not in TC_RNTI_List and TC_RNTI not in newTC_RNTIList:
								TC_RNTI_Status = True
								newTC_RNTIList.append(TC_RNTI)

						# The gNB extracts the new requester resource's necessities sent via relay.
						RB_dl = str(int(int(data.split('|')[1])/12)) # Resources Blocks needed by the new requester for downlink.
						RB_ul = str(int(int(data.split('|')[2])/12)) # Resources Blocks needed by the new requester for uplink.
						L_dl = data.split('|')[3] # Number of symbols needed by the new requester for downlink.
						L_ul = data.split('|')[4] # Number of symbols needed by the new requester for uplink.

						# Assigns a Time-To-Live time to the new computed TC-RNTI.
						# If this TTL expires, the TC-RNTI is erased.
						TTL = env.now

						# Adding the new device to the resource allocation table.
						# However, it still has not resources assigned.
						# The gNB only saves what the requester needs. In the future
						# the gNB will assign resources for this requester based in the
						# saved requester necessities.
						new_requester_row = np.array((RA_RNTI, TC_RNTI, RB_dl, \
							'None', L_dl, 'None', RB_ul, 'None', L_ul, 'None', TTL))
						resourceAllocationTable = np.vstack([resourceAllocationTable, new_requester_row])
						
						# The gNB will send back the RAR to the new requester. The message contains 
						# the RA-RNTI to identify the requester that sent the PRACH request and the 
						# message also contains the computed TC-RNTI to be assigned to the requester.
						RACHPendingList.append(RA_RNTI + '|' + str(TC_RNTI))

			# Checks if TTL has timed up for every requester in the resource allocation table 
			# when a RRC Connection Request is expected.
			rowToDelete = []
			for lookup in range(1, resourceAllocationTable.shape[0]):

				# If TTL has timed up for this requester includes the requester in the removing list.
				if resourceAllocationTable[lookup, 3] == 'None' and \
				env.now - float(resourceAllocationTable[lookup, 10]) >= TTLRar:
					rowToDelete.append(lookup)
			
			# If there are requesters to remove from the resource allocation table, removes them.
			if len(rowToDelete) > 0:
				for listIndex, row in enumerate(rowToDelete):
					resourceAllocationTable = np.delete(resourceAllocationTable, row - listIndex, 0)

		# Checks if there are empty spaces in commonDownlinkBW to send pending RARs.
		for channelIndex, channel in enumerate(commonDownlinkBW):
			# If there is an available channel that was not used for another transmission, an RAR
			# can be sent in that channel.
			if channelIndex > 0 and channel == None:

				# If there are pending RARs...
				if len(RACHPendingList) > 0:
					# Sends the RAR through the channel.
					commonDownlinkBW[channelIndex] = RACHPendingList.pop(0)

					# Prints the timestamp when the gNB sends the RAR, and the channel where the RAR was sent.
					if printLogs == True: print(env.now, 'us: gNB -> Sending RAR to:', commonDownlinkBW[channelIndex])

					# Add 'mnPower' to the total energy spent in the mobile network band by the gNB.
					totalEnergyGNB += mnPower

		# Checks RRC uplink to find RRC Connection Requests.
		for channelIndex in range(len(uplinkRRC)):

			# If there is an occupied channel in the RRC uplink bandwidth,
			# there is an available RRC Connection Request (MSG3).
			if uplinkRRC[channelIndex] != None:
				# RRC request data has the format: TC-RNTI|40-bitValue
				TC_RNTI = uplinkRRC[channelIndex].split('|')[0]			
				# Looking for this TC_RNTI in resource allocation table to check it is in the table.
				for lookup in range(1, resourceAllocationTable.shape[0]):
					
					# If the TC-RNTI sent in MSG3 is the resource allocation table, the gNB should
					# send back an RRC Connection Setup message containing the allocated resources
					# for the requester that sent MSG3 (in case two requesters send an MSG3 with
					# the same TC-RNTI, only the requester that sent MSG3 the first is the one that
					# will have assigned resources, it is identified by its 40-bit identifier).
					if TC_RNTI == resourceAllocationTable[lookup, 1]:
						# This means that the requester that sent MSG3, executed the 
						# traditional Random-Access procedure.
						if resourceAllocationTable[lookup, 0] != 'empty' and \
						resourceAllocationTable[lookup, 0].split('_')[1] == '0' and \
						resourceAllocationTable[lookup, 2] == 'None':

							# 1 RB will be allocated for the requester.
							
							# The TTL is restarted for this requester because
							# the requester sent an MSG3 before TTL expiration.
							TTL = env.now

							RB_dl_offered = 'None'
							RB_ul_offered = 'None'
							SC_dl_offered = ''
							SC_ul_offered = ''
							downlink_resources_found = False
							uplink_resources_found = False

							# Downlink lookup. Looks for last device with allocated physical resources
							# and allocates the resources next to that device's resources.
							for rowIndex in range(downlinkRB.shape[0]):

								# If there are no resources found yet, continues looking
								# for resources.
								if downlink_resources_found == False:

									# Counter for the symbols that are assigned.
									symbolCounter = 0
									for columnIndex in range(downlinkRB.shape[1]):
										# If the current symbol is not assigned yet, assigns
										# the symbol for this requester.
										if downlinkRB[rowIndex, columnIndex] == 0.0:
											# Add 1 to the number of assigned symbols.
											symbolCounter += 1

									# If 14 symbols were assigned, it means that this requester
									# has found an entire RB in the mobile network for it.
									if symbolCounter == 14:
										downlink_resources_found = True

										# Subcarrier index offered to the requester for downlink.
										SC_dl_offered = rowIndex

							# Fills the space occupied by the resources assigned for downlink 
							# to the requester with the requester TC-RNTI in the downlink resource's grid.
							for i in range(14):
								downlinkRB[SC_dl_offered, i] = TC_RNTI
							
							# Uplink lookup. Looks for last device with allocated physical resources
							# and allocates the resources next to that device's resources.
							for rowIndex in range(uplinkRB.shape[0]):

								# If there are no resources found yet, continues looking
								# for resources.
								if uplink_resources_found == False:

									# Counter for the symbols that are assigned.
									symbolCounter = 0

									# If the current symbol is not assigned yet, assigns
									# the symbol for this requester.
									for columnIndex in range(uplinkRB.shape[1]):
											if uplinkRB[rowIndex, columnIndex] == 0.0:
												# Add 1 to the number of assigned symbols.
												symbolCounter += 1

									# If 14 symbols were assigned, it means that this requester
									# has found an entire RB in the mobile network for it.
									if symbolCounter == 14:
										uplink_resources_found = True

										# Subcarrier index offered to the requester for uplink.
										SC_ul_offered = rowIndex

							# Fills the space occupied by the resources assigned for uplink 
							# to the requester with the requester TC-RNTI in the uplink resource's grid.
							for i in range(14):
								uplinkRB[SC_ul_offered, i] = TC_RNTI

							# Assigning the subcarrier range to the requester for downlink and uplink.
							SC_dl_offered = str(SC_dl_offered*12) + '-' + str(SC_dl_offered*12 + 11)
							SC_ul_offered = str(SC_ul_offered*12) + '-' + str(SC_ul_offered*12 + 11)
							# Assigning the number of symbols offered to the requester.
							L_dl_offered = '14'
							L_ul_offered = '14'
							# Setting the start index of the symbols in the subcarrier.
							S_dl_offered = '0'
							S_ul_offered = '0'

							# Allocating physical resources for this device.
							resourceAllocationTable[lookup, 2] = RB_dl_offered # Resources Blocks assigned to the new requester for downlink.
							resourceAllocationTable[lookup, 3] = SC_dl_offered # Subcarrier assigned to the new requester for downlink.
							resourceAllocationTable[lookup, 4] = L_dl_offered # Number of symbols assigned to the new requester for downlink.
							resourceAllocationTable[lookup, 5] = S_dl_offered # Symbol's index assigned to the new requester for downlink.
							resourceAllocationTable[lookup, 6] = RB_ul_offered # Resources Blocks assigned to the new requester for uplink.
							resourceAllocationTable[lookup, 7] = SC_ul_offered # Subcarrier assigned to the new requester for uplink.
							resourceAllocationTable[lookup, 8] = L_ul_offered # Number of symbols assigned to the new requester for uplink.
							resourceAllocationTable[lookup, 9] = S_ul_offered # Symbol's index assigned to the new requester for uplink.
							resourceAllocationTable[lookup, 10] = TTL # New TTL.
							
							# RRC Connection Setup (MSG4) messages that will be send to the requester.
							RRCPendingList.append('TC-RNTI=' + uplinkRRC[channelIndex].split('|')[0] + \
								'|40-bitValue=' + uplinkRRC[channelIndex].split('|')[1] + \
								'|SC_dl=' + SC_dl_offered + \
								'|K0=0|S=' + S_dl_offered + \
								'|L=' + resourceAllocationTable[lookup, 4] + \
								'|SC_ul=' + SC_ul_offered + \
								'|K2=0|S=' + S_ul_offered + \
								'|L=' + resourceAllocationTable[lookup, 8])						
							
							# Clears RA-RNTI from resource allocation table after resource assigment.
							resourceAllocationTable[lookup, 0] = 'empty'

						# This means that the requester that sent MSG3, executed the 
						# Framework RAA procedure.
						elif resourceAllocationTable[lookup, 0] != 'empty' and \
						resourceAllocationTable[lookup, 0].split('_')[1] == '1' and \
						resourceAllocationTable[lookup, 3] == 'None':

							# The TTL is restarted for this requester because
							# the requester sent an MSG3 before TTL expiration.
							TTL = env.now
							
							# Looks for all devices with allocated physical resources to check if 
							# this device resource's requirements can be allocated in the same frequency resources.

							RB_dl_offered = 'None'
							RB_ul_offered = 'None'
							# Resources needed by the requester.
							RB_dl_wanted = int(resourceAllocationTable[lookup, 2])
							RB_ul_wanted = int(resourceAllocationTable[lookup, 6])
							L_dl_wanted = int(resourceAllocationTable[lookup, 4])
							L_ul_wanted = int(resourceAllocationTable[lookup, 8])
							downlink_resources_found = False
							uplink_resources_found = False
							
							# Downlink lookup. Looks for last device with allocated physical resources
							# and allocates the resources next to that device's resources.
							availableResources = 0
							for rowIndex in range(downlinkRB.shape[0]):
								for columnIndex in range(downlinkRB.shape[1]):

									# If there are no resources found yet, continues looking
									# for resources.
									if downlink_resources_found == False:

										# If the block requested by the requester fits into the resource's grid,
										# the block will be assigned to the requester.
										if rowIndex+RB_dl_wanted <= downlinkRB.shape[0] and \
										columnIndex+L_dl_wanted <= downlinkRB.shape[1]:

											# Block of resources requested by the requester for downlink.
											block = downlinkRB[rowIndex:rowIndex+RB_dl_wanted, columnIndex:columnIndex+L_dl_wanted]
											for R in range(block.shape[0]):
												for S in range(block.shape[1]):
													if block[R, S] == 0.0:
														availableResources += 1

										# If the block is empty, the resources for the requester have been found.
										if availableResources == RB_dl_wanted * L_dl_wanted:
											downlink_resources_found = True

											# Subcarriers offered to the requester for downlink.
											SC_dl_offered = str(rowIndex*12) + '-' + str(rowIndex*12 + RB_dl_wanted*12 - 1)

											# First subcarrier index offered to the requester for downlink.
											SC_index = rowIndex

											# Symbol index offered to the requester for downlink within the offered subcarriers.
											S_dl_offered = str(columnIndex)
											S_index = columnIndex
										# If the block is not empty, looks for another empty block next time.
										else:
											availableResources = 0

							# Fills the space occupied by the resources assigned for downlink 
							# to the requester with the requester TC-RNTI in the downlink resource's grid.
							for R in range(RB_dl_wanted):
								for S in range(L_dl_wanted):
									downlinkRB[SC_index + R, S_index + S] = TC_RNTI

							# Uplink lookup. Looks for last device with allocated physical resources
							# and allocates the resources next to that device's resources.
							availableResources = 0
							for rowIndex in range(uplinkRB.shape[0]):
								for columnIndex in range(uplinkRB.shape[1]):

									# If there are no resources found yet, continues looking
									# for resources.
									if uplink_resources_found == False:

										# If the block requested by the requester fits into the resource's grid,
										# the block will be assigned to the requester.
										if rowIndex+RB_ul_wanted <= uplinkRB.shape[0] and \
										columnIndex+L_ul_wanted <= uplinkRB.shape[1]:

											# Block of resources requested by the requster for uplink.
											block = uplinkRB[rowIndex:rowIndex+RB_ul_wanted, columnIndex:columnIndex+L_ul_wanted]
											for R in range(block.shape[0]):
												for S in range(block.shape[1]):
													if block[R, S] == 0.0:
														availableResources += 1

										# If the block is empty, the resources for the requester have been found.
										if availableResources == RB_ul_wanted * L_ul_wanted:
											uplink_resources_found = True

											# Subcarriers offered to the requester for uplink.
											SC_ul_offered = str(rowIndex*12) + '-' + str(rowIndex*12 + RB_ul_wanted*12 - 1)

											# First subcarrier index offered to the requester for uplink.
											SC_index = rowIndex

											# Symbol index offered to the requester for uplink within the offered subcarriers.
											S_ul_offered = str(columnIndex)
											S_index = columnIndex
										# If the block is not empty, looks for another empty block next time.
										else:
											availableResources = 0

							# Fills the space occupied by the resources assigned for uplink 
							# to the requester with the requester TC-RNTI in the uplink resource's grid.
							for R in range(RB_ul_wanted):
								for S in range(L_ul_wanted):
									uplinkRB[SC_index + R, S_index + S] = TC_RNTI

							# Allocating physical resources for this device.
							resourceAllocationTable[lookup, 2] = RB_dl_offered # Resources Blocks assigned to the new requester for downlink.
							resourceAllocationTable[lookup, 3] = SC_dl_offered # Subcarriers assigned to the new requester for downlink.
							resourceAllocationTable[lookup, 5] = S_dl_offered # Symbol's index assigned to the new requester for downlink.
							resourceAllocationTable[lookup, 6] = RB_ul_offered # Resources Blocks assigned to the new requester for uplink.
							resourceAllocationTable[lookup, 7] = SC_ul_offered # Subcarriers assigned to the new requester for uplink.
							resourceAllocationTable[lookup, 9] = S_ul_offered # Symbol's index assigned to the new requester for uplink.
							resourceAllocationTable[lookup, 10] = TTL # New TTL.

							# RRC Connection Setup (MSG4) messages that will be send to the requester.
							RRCPendingList.append('TC-RNTI=' + uplinkRRC[channelIndex].split('|')[0] + \
								'|40-bitValue=' + uplinkRRC[channelIndex].split('|')[1] + \
								'|SC_dl=' + SC_dl_offered + \
								'|K0=0|S=' + S_dl_offered + \
								'|L=' + resourceAllocationTable[lookup, 4] + \
								'|SC_ul=' + SC_ul_offered + \
								'|K2=0|S=' + S_ul_offered + \
								'|L=' + resourceAllocationTable[lookup, 8])
							
							# Clears RA-RNTI from resource allocation table after resource assigment.
							resourceAllocationTable[lookup, 0] = 'empty'
		
		# Checks if there are empty spaces in downlinkRRC to send pending RRC Connection Setup messages.
		for channelIndex, channel in enumerate(downlinkRRC):
			# If there is an available channel that was not used for another transmission, an MSG4
			# can be sent in that channel.
			if channel == None:
				
				# If there are pending MSG4...
				if len(RRCPendingList) > 0:
					
					# Sends the MSG4 through the channel.
					downlinkRRC[channelIndex] = RRCPendingList.pop(0)
					
					# Prints the timestamp when the gNB sends the MSG4, and the channel where the MSG4 was sent.
					if printLogs == True: print(env.now, 'us: gNB -> Sending RRC Connection Setup for:', downlinkRRC[channelIndex])

					# Add 'mnPower' to the total energy spent in the mobile network band by the gNB.
					totalEnergyGNB += mnPower

		# If the current mobile network slot has expired...
		if env.now - numerologySlotTimer >= numerologySlot:
			# Resetting the timer.
			numerologySlotTimer = env.now

			# Add 1 to the number of symbols in the mobile network.
			symbolCounterMN += 1
			# After 14 symbols, add 1 to the number of slots in the mobile network.
			if symbolCounterMN > 13: 
				symbolCounterMN = 0
				slotCounterMN += 1

				# After 80 slots of 14 symbols, restart the slots counter in the mobile network.
				if slotCounterMN > 79:
					slotCounterMN = 0

		#############################################
		yield env.timeout(timeResolution) # Time step.
		#############################################

		# Cleaning signals.
		downlinkRRC = [None for i in range(64)] # Cleaning downlink channels for RCC messages.
		commonDownlinkBW = [None for i in range(65)] # Cleaning common downlink channels.
		collisionList = [None for i in range(64)] # Cleaning the collision list in the mobile network band after one time step.
		collisionListBand24 = [None for i in range(totalNetworkSlices)] # Cleaning the collision list in the 2.4GHz band after one time step.
		for bandIndex in range(len(collisionListBand24)):
			collisionListBand24[bandIndex] = [None for i in range(3)] # Cleaning the 3 BLE channels for each slice in the 2.4GHz band.

def device(env, blePower, mnPower, timeResolution=31.25, index=0, status='Requester'\
	, totalNetworkSlices=1, discoverBeforeSIB1=False, backOff=False, framework=False\
	, noClassicRACH=False, maxSym=14, maxSC=3, getFreqFromGNB=False, D2DRandomFreq=False\
	, printLogs=False):
	
	# Device Configuration.
	global commonDownlinkBW
	global commonUplinkBW
	global downlinkRRC
	global uplinkRRC
	global downlinkBW
	global uplinkBW
	global downlinkRB
	global uplinkRB
	global band24
	global symbolCounterMN
	global slotCounterMN
	global numerologySlot
	global collisionList
	global totalCollisionsNetwork
	global totalCollisions24Band
	global totalEnergyNetwork
	global totalEnergy24Band
	global totalTimeForRegistration
	global totalRegisteredDevices
	global totalRegisteredByGNB
	global totalRegisteredByRelay
	global totalDev
	global availableFreq24BandListening
	global registeredDevicesByTimestamp
	global randomValueForAccess
	global fixedR
	global richedRelaysStartTime

	# Sets the network slice (the area where the device will camp on)
	# within the mobile network coverage area: Randomly selected.
	networkSlice = random.randint(0, totalNetworkSlices-1)

	# Informs that the proposed Bluetooth backoff has expired (True) or not (False).
	inquirerBackOffTimeOut = True
	scannerBackOffTimeOut = True

	# Timers.
	inquirerBackOffTimer = 0
	scannerBackOffTimer = 0

	# Time to wait to transmit a response for an advertising message.
	timeToResponse = timeResolution

	# Minimum ra-responseWindow interval {sl1, sl2, sl4, sl8, sl10, sl20, sl40, sl80}.
	RelayFoundTime = numerologySlot

	# This flag is used to clean the transmission in the last timeResolution step.
	# Signal transmitted (True) and not transmitted (False).
	signalTransmitted = False

	# This flag is used to stop listen discovery messages when 
	# there is an already discovered message.
	# Signal received (True) and not received (False).
	signalReceived = False
	
	# Informs that the device needs to set some parameters at the time it starts
	# the relay tasks.
	# Configured (True) and not configured (False).
	deviceConfigured = False
	
	# This flag is used to clean the transmission in the uplinkBW 
	# (when is forwarded to the gNB a discovery message).
	# Message forwarded (True) and not forwarded (False).
	txTogNBByRelay = False
	
	# This flag is used to clean the PRACH (MSG2) transmission in the commonUplinkBW.
	# PRACH transmitted (True) and not transmitted (False).
	prachTransmitted = False
	
	# Informs a preamble was selected (True) or not (False).
	preambleSelected = False

	# Informs that the requester found a RAR and it stopped the Random-Access for now.
	# Stopped PRACH (True) or not (False).
	stopPRACH = False

	# ra-responseWindow interval.
	ra_responseWindow = numerologySlot # (minimum ra-responseWindow interval {sl1, sl2, sl4, sl8, sl10, sl20, sl40, sl80})
	
	# Setting up the variable that will store the timestamp when a preamble is selected. 
	ra_responseWindowTimer = 0

	# Informs an MSG3 was sent by the requester using the traditional Random-Access procedure.
	# MSG3 sent (True) or not (False).
	RRCRequestPRACH = False
	sentRRCRequestPRACH = False

	# Informs an MSG3 was sent by the requester using the Framework RAA procedure.
	# MSG3 sent (True) or not (False).
	RRCRequestRelay = False
	sentRRCRequestRelay = False

	# Number of times that were allocated resources for this device.
	resourcesAllocated = 0

	# This flag is used to stop the D2D discovery procedure.
	# Stop (True) or not (False).
	stopDiscovery = False

	# Sets the index for the channel where starts the block of resources of
	# this device.
	myULResourcesIndex = 0
	
	# Informs that MSG4 was received (True) or not (False).
	rrcReceived = False

	# Informs SIB1 has arrived (True) or not (False).
	sib1Arrived = False

	# Informs that this device acting like a relay should send the requester message to the gNB (True) or not (False).
	forwardMsg = False

	# Informs that the ACB backoff is running (True) or not (False).
	TacbFlag = False

	# Sets the remaining slots the requester will wait to listen to SIB1 messages.
	# In this case, it will listen since the first SIB1 is transmitted.
	noRACH = 0

	# Number of unsuccessful access attempts.
	noRACHStored = 0

	# This parameter is used by the BLE discovery procedure.
	# timeSteps is used to set the start time steps of the discovery procedure.
	timeSteps = 0

	# Informs that a frequency for inquiring has been selected randomly.
	D2DRandomFreqSelected = False

	# Informs that the interval for advertising PDU transmission has not started.
	sendFirstPDU = False

	#################################################################
	# Random start time in the simulation for every device in the
	# range 0ms - 15ms.
	startBackOff = random.randint(0, 480)
	yield env.timeout(timeResolution*startBackOff)
	#################################################################

	# If the device acts like a requester there are some parameters
	# to set up.
	if status == 'Requester':
		# When requester role, the device selects a random RA-RNTI 
		# in the available range where the mobile network allows 
		# RA-RNTIs to be assigned.
		# The RA_RNTIRequester is used for the Framework RAA.
		RA_RNTIRequester = str(random.randint(1, 65523))
		
		# The requester sets the number of subcarriers and symbols 
		# it needs for the application (in this case randomly). In 
		# the case of the subcarriers, the random number selected is
		# multiplied by 12 to point to the first subcarrier of a 
		# Resource Block, which contains 12 subcarriers.
		rsc = str(random.randint(1, maxSC)*12) + '|' + \
			str(random.randint(1, maxSC)*12) + '|' + \
			str(random.randint(1, maxSym)) + '|' + \
			str(random.randint(1, maxSym))

		# Message that the requester will send via D2D communications
		# to find a relay during the discovery procedure. This message
		# will be as destiny the gNB after forwarded by a relay. The
		# message contains: "RA-RNTI (requester) | subcarriers for downlink | subcarriers for uplink | symbols for downlink | symbols for uplink".
		msg = RA_RNTIRequester + '|' + rsc
		
		# Prints logs for the requester including timestamp, device index, 
		# message sent to relays, and slice group the requester belongs to.
		if printLogs == True: print(env.now, 'us: Device (' + str(index) + \
			'): Requirements via Relay:' + msg, '(Slice Group:' + str(networkSlice) + ')')

	RA_RNTI = '' # Clears RA-RNTI to start the procedures.
	TC_RNTIRequester = '' # Clears the TC-RNTI (used for the Framework RAA).
	TC_RNTI = '' # Clears TC-RNTI (used for the traditional Random-Access procedure).

	# While the number of devices registered in the mobile network
	# is less than than the number of total devices attempting to
	# connect to the network, this device will continue executing 
	# its requester or relay tasks.
	while totalRegisteredDevices < totalDev + fixedR:
		if totalRegisteredDevices >= fixedR and richedRelaysStartTime == 0:
			richedRelaysStartTime = env.now
			totalEnergy24Band = 0
			totalEnergyNetwork = 0
			totalEnergyGNB = 0
		
		# Requester tasks.
		if status == 'Requester':
			
			#################### Receives SIB1 (MSG1) ####################
			# Checks SIB1 search space to find SIB1 with the PRACH
			# configuration (MSG1).
			# If the ACB backoff is not running (TacbFlag == False), 
			# the Random-Access (PRACH) is running (stopPRACH == False)
			# and there are no selected preambles outside the ra-responseWindow
			# interval (preambleSelected == False), and there are available 
			# preambles (SIB1 received -> len(commonDownlinkBW[0]) == 64)...
			if TacbFlag == False and stopPRACH == False and \
				preambleSelected == False and commonDownlinkBW[0] != None \
				and len(commonDownlinkBW[0]) == 64:
				
				# ... SIB1 has been found. However, the custom RACH procedure
				# only allows the requester to get a preamble from SIB1 if the
				# number of failed attempts has been reduced to zero (every time
				# the requester find SIB1 it reduces by 1 the number of failed
				# attempts to gain access).
				if noRACH == 0:
					sib1Arrived = True
					discoverBeforeSIB1 = True

					# Executes ACB algorithm. Generates a random uniform number
					# in the range [0, 1].
					# If is used the custom RACH procedure...
					if noClassicRACH == True:
						# ... Gives access always.
						accessNumber = 0
					# In case is used the traditional RACH procedure...
					else:
						accessNumber = random.randint(0, 100)*0.01 # 0.01 resolution.

					# As part of the ACB procedure, the requester only gets one preamble
					# if the calculated access number is less than the access value sent
					# by the gNB in the SIB1 message (accessNumber < randomValueForAccess).
					if accessNumber < randomValueForAccess:
						# Informs a preamble was selected.
						preambleSelected = True
						
						# Sets the starting timestamp of the ra-responseWindow.
						ra_responseWindowTimer = env.now

						# It is randomly selected one of the 64 available preambles.
						preamble = random.choice(commonDownlinkBW[0])

						# Now, some parameters sent by the gNB are used to compute the
						# RA-RNTI that the device will use to communicate with the gNB
						# during the Random-Access procedure.
						s_id = symbolCounterMN
						t_id = slotCounterMN
						f_id = 0 # because is used FDD.
						ul_carrier_id = 0 # no SUL carrier in this case, just pure NR.
						RA_RNTI = str(1 + s_id + 14*t_id + 14*80*f_id + 14*80*8*ul_carrier_id)

						# The message used to inform the gNB about the selected preamble
						# includes the RA-RNTT that indentify this requester and the
						# selected preamble.
						msgRACH = RA_RNTI + '|' + str(preamble)

						# Sends PRACH request (MSG2).
						# If there is occupied the common uplink channel associated with
						# the frequency related to the selected preamble, a collision occurs.
						if commonUplinkBW[preamble] != None or collisionList[preamble] != None:
							# Prints logs for the requester including timestamp, device index, 
							# and the collision preamble frequency.
							if printLogs == True: print(env.now, 'us: Device (' + str(index) + \
								'): Collision in PRACH preamble=' + str(preamble)) # notify collision
							
							commonUplinkBW[preamble] = None # Free the channel.
							
							# Sets the selected preamble frequency as a conflictive channel 
							# (collisions will occur for the rest of the devices).
							collisionList[preamble] = 'collision'
							
							# Increases by one the number of collisions in the mobile network band.
							totalCollisionsNetwork += 1
						# In case the channel associated to the selected preamble is empty, the 
						# requester can send the PRACH message to the gNB.
						else:
							# Prints logs for the requester including timestamp, device index, 
							# the selected preamble, and the PRACH message.
							if printLogs == True: print(env.now, 'us: Device (' + str(index) + \
								') sent PRACH in frequency (preamble)=' + str(preamble), 'PRACH MESSAGE:', msgRACH)

							# Sets the selected preamble frequency (channel) as occupied.
							commonUplinkBW[preamble] = msgRACH
							# Informs a PRACH message was transmitted.
							prachTransmitted = True

						# In any case, the energy spent by the device increases by the transmission power
						# used to send the message to the mobile network.
						totalEnergyNetwork += mnPower

					# In case the calculated access number is greater than the access value sent
					# by the gNB in the SIB1 message (accessNumber > randomValueForAccess), the
					# requester executes the ACB backoff.
					else:
						# Informs an ACB backoff was executed.
						TacbFlag = True
						
						# Sets the starting timestamp of the ACB backoff.
						TacbTimer = env.now
						
						# Computes the ACB backoff.
						Tacb = random.choice([4])#, 8, 16, 32, 64, 128, 256, 512])
						Tbarring = (0.7 + 0.6*random.randint(0, 100)*0.01)*Tacb

						# Prints logs for the requester including timestamp, device index, 
						# the access number, and the computed ACB backoff.
						print(env.now, 'Device (' + str(index) + ') prob=' + str(accessNumber) + \
							' selects Tbarring=' + str(Tbarring) + ' seconds')
				# It is reduced by 1 the number of failed attempts to gain access
				else:
					noRACH -= 1

			# If the ACB backoff is running and the elapsed time is equal or greater than
			# the computed ACB backoff interval, the ACB backoff has expired.
			if TacbFlag == True and env.now - TacbTimer >= Tbarring*10**6: # converts Tbarring from seconds into us
				# Informs that the ACB backoff has stopped.
				TacbFlag = False

				# Prints logs for the requester including timestamp, device index, 
				# and informs that the ACB backoff has expired.
				print(env.now, 'Device (' + str(index) + ') Tbarring END')

			# If the Random-Access (PRACH) is running (stopPRACH == False),
			# and the requester got a preamble from the last SIB1 (preambleSelected == True),
			# and the ra-responseWindow has not expired since the preamble selection, the 
			# requester will check if there are Random-Access Response RAR from the
			# gNB in the common downlink channel.
			if stopPRACH == False and preambleSelected == True and \
				env.now - ra_responseWindowTimer < ra_responseWindow:
				# Checks RAR search space...
				for channelDBWIndex in range(1, len(commonDownlinkBW)):
					# If there are responses in one of the channels of the common downlink channel...
					if commonDownlinkBW[channelDBWIndex] != None:
						# If the RAR sent by the gNB contains this requester RA-RNTI, the identifier '0' to describe
						# that the current RAR has been acquired via PRACH (no Framework RAA), and the preamble sent
						# by this requester, the requester knows the RAR belongs to itself.
						if commonDownlinkBW[channelDBWIndex].split('|')[0].split('_')[0] == RA_RNTI and \
							commonDownlinkBW[channelDBWIndex].split('|')[0].split('_')[1] == '0' and \
							commonDownlinkBW[channelDBWIndex].split('|')[1] == str(preamble):
							# RAR acquired via PRACH.
							
							# Prints logs for the requester including timestamp, device index, 
							# the notification that the RAR was received via PRACH, and the new
							# TC-RNTI offered by the gNB to this requester to be indetified in 
							# the RRC Connection Request message (MSG3).
							if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
								') RAR received via PRACH, new TC_RNTI=' + \
								commonDownlinkBW[channelDBWIndex].split('|')[2])
							
							# Gets the new TC-RNTI for this requester.
							TC_RNTI = commonDownlinkBW[channelDBWIndex].split('|')[2]

							# Randomly computes a 40-bit value to use as an identifier for the gNB
							# if multiple requesters with the same RA-RNTI received the same RAR
							# (the same TC-RNTI will be assigned to all those requesters). In this case,
							# the gNB only reply (MSG4) to the first requester that sent MSG3 (the reply 
							# contains the 40-bit indentifier to identify only the first requester).
							randomValue = random.randint(0, 2**40 - 1) # TC_RNTI|40-bitValue
							
							# Informs that the requester found a RAR and it stopped the Random-Access for now.
							stopPRACH = True

							# Stops looking for RAR responses since one was found.
							break
			# If the Random-Access (PRACH) is running (stopPRACH == False),
			# and the requester got a preamble from the last SIB1 (preambleSelected == True),
			# and the ra-responseWindow has expired since the preamble selection, the 
			# requester will clear the RA-RNTI it computed before and it will clear the
			# selected preamble in order to restart the PRACH procedure again.
			elif stopPRACH == False and preambleSelected == True and \
				env.now - ra_responseWindowTimer >= ra_responseWindow:
				# ra-responseWindow has timed up and it must restart PRACH procedure.

				# Informs a preamble was unselected.
				preambleSelected = False
				# Clears the RA-RNTI identifier.
				RA_RNTI = ''
				# Unselects the preamble.
				preamble = ''

				# Add 1 to the number of unsuccessful Random-Access attempts.
				noRACHStored += 1
				# Randomly selects the number of successive SIB1s that the requester will not listen to.
				# (the requester will not execute the Random-Access procedure).
				noRACH = random.randint(0, noRACHStored)

			# Checks RAR search space (via relay).
			# If the Random-Access procedure is running (stopPRACH == False),
			# and if only SIB1 has been received the requester can listen to RAR 
			# because the device needs SIB1 information to receive RAR.
			if stopPRACH == False and sib1Arrived == True:
				# The requester will search for RAR in the common downlink channels.
				for channelDBWIndex in range(1, len(commonDownlinkBW)):
					# If the requester finds a RAR...
					if commonDownlinkBW[channelDBWIndex] != None:
						# If the message received contains the RA_RNTIRequester identifier of this requester, and
						# the '1' identifier to describe that the current RAR has been acquired via Framework RAA
						# (no traditional Random-Access procedure).
						if commonDownlinkBW[channelDBWIndex].split('|')[0].split('_')[0] == RA_RNTIRequester and \
							commonDownlinkBW[channelDBWIndex].split('|')[0].split('_')[1] == '1':
							# RAR acquired via relay

							# Prints logs for the requester including timestamp, device index, 
							# notification of the RAR reception because of the use of the Framework RAA,
							# and the TC-RNTI offered by the gNB to this requester.
							if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
								') RAR received via RELAY, new TC_RNTI=' + commonDownlinkBW[channelDBWIndex].split('|')[1])
							
							# Sets the TC_RNTI variable with the TC-RNTI offered by the gNB to this requester.
							TC_RNTIRequester = commonDownlinkBW[channelDBWIndex].split('|')[1]

							# Randomly computes a 40-bit value to use as an identifier for the gNB
							# if multiple requesters with the same RA-RNTI received the same RAR
							# (the same TC-RNTI will be assigned to all those requesters). In this case,
							# the gNB only reply (MSG4) to the first requester that sent MSG3 (the reply 
							# contains the 40-bit indentifier to identify only the first requester).
							randomValueRequester = random.randint(0, 2**40 - 1) # TC_RNTI|40-bitValue
							
							# Informs that the requester found a RAR and it stopped the Random-Access for now.
							stopPRACH = True

							# Stops the discovery because an RAR has been found.
							stopDiscovery = True

							# Sets the starting timestamp of the MSG3 transmission.
							afterRelayFoundTimer = env.now

							# Stops looking for empty channels.
							break

			#################### Sends RRC Connection Request (MSG3) ####################
			# If the requester received a RAR by using the Framework RAA 
			# (the TC_RNTIRequester is set -> TC_RNTIRequester != '') and
			# the requester has not sent an RRC Connection Request (MSG3) yet,
			# the requester will send the MSG3.
			if sentRRCRequestRelay == False and TC_RNTIRequester != '':
				# The requester will search for the first empty uplink channel
				# to transmit MSG3, this is equivalent to send MSG3 in the
				# uplink grants given by the gNB to the requester in MSG2.
				for ulRRCIndex, ulRRCChannel in enumerate(uplinkRRC):
					# When the requester finds an empty channel, the requester
					# transmits MSG3.
					if ulRRCChannel == None:
						# Sends the requester TC_RNTIRequester identifier and the 40-bit value to the gNB.
						uplinkRRC[ulRRCIndex] = TC_RNTIRequester + '|' + str(randomValueRequester) # TC_RNTI|40-bitValue.

						# Sets RRCIndexRelay to the index of channel where is transmitted MSG3.
						# This is used to free the channel in the next time step.
						RRCIndexRelay = ulRRCIndex

						# Informs an MSG3 was sent by this requester.
						RRCRequestRelay = True
						sentRRCRequestRelay = True

						# Sets the starting timestamp of the MSG3 transmission.
						afterRelayFoundTimer = env.now

						# Stops looking for empty channels.
						break
			# If the requester received a RAR by using the traditional PRACH 
			# procedure (the TC-RNTI is set -> TC-RNTI != ''), and
			# the requester has not sent an RRC Connection Request (MSG3) yet,
			# and the Random-Access procedure is stopped because an RAR arrived,
			# the requester will send the MSG3.
			if sentRRCRequestPRACH == False and stopPRACH == True and TC_RNTI != '':
				# The requester will search for the first empty uplink channel
				# to transmit MSG3, this is equivalent to send MSG3 in the
				# uplink grants given by the gNB to the requester in MSG2.
				for ulRRCIndex, ulRRCChannel in enumerate(uplinkRRC):
					# When the requester finds an empty channel, the requester
					# transmits MSG3.
					if ulRRCChannel == None:
						# Sends the requester TC-RNTI identifier and the 40-bit value to the gNB.
						uplinkRRC[ulRRCIndex] = TC_RNTI + '|' + str(randomValue) # TC_RNTI|40-bitValue.

						# Sets RRCIndexPRACH to the index of channel where is transmitted MSG3.
						# This is used to free the channel in the next time step.
						RRCIndexPRACH = ulRRCIndex

						# Informs an MSG3 was sent by this requester.
						RRCRequestPRACH = True
						sentRRCRequestPRACH = True

						# Sets the starting timestamp of the MSG3 transmission.
						afterRelayFoundTimer = env.now

						# Stops looking for empty channels.
						break

			#################### Receives RRC Connection Setup (MSG4) ####################
			# If the requester sent MSG3 by using the Framework RAA...
			if sentRRCRequestRelay == True:
				# The requester will search for the first empty downlink channel
				# to receive MSG4.
				for dlRRCIndex, dlRRCChannel in enumerate(downlinkRRC):
					# If there is a downlink channel with info (MSG4) and the info contains the 
					# TC_RNTIRequester of this requester, and the message also contains 
					# the 40-bit value sent before by this requester in MSG3, the requester 
					# will acquire resources from the mobile network via relay.
					if dlRRCChannel != None and \
						dlRRCChannel.split('|')[0].split('=')[1] == TC_RNTIRequester and \
						dlRRCChannel.split('|')[1].split('=')[1] == str(randomValueRequester):

						# If this requester has not resources yet, the requester acquires
						# the resources offered by the gNB.
						if resourcesAllocated == 0:
							# Add 1 to the total number of registered devices via relay 
							# (using the Framework RAA).
							totalRegisteredByRelay += 1
							
							# Add 1 to the total number of registered devices.
							totalRegisteredDevices += 1
							
							# Add 1 to the total number of resources allocated for this device.
							# In this case, 1 is the maximum number of times the resources are allocated.
							resourcesAllocated += 1

							if totalRegisteredDevices > fixedR:
								registeredDevicesByTimestamp.append(env.now - richedRelaysStartTime + timeResolution)
							else:
								# Saves the exact timestamp which this device was registered.
								registeredDevicesByTimestamp.append(env.now)

							# Prints logs for the requester including timestamp, the total number of
							# registered devices, this device index, and the slice where this device
							# belongs to.
							print(env.now, 'us: Registered', totalRegisteredDevices, 'of', totalDev + fixedR, \
								'Device (' + str(index) + ') -> (Slice Group: ' + str(networkSlice) + ')')
							
							# Prints logs for the requester including timestamp, device index, and the
							# number of resources allocated (1), and the channel where the MSG4 was found.
							if printLogs == True: print(env.now, 'us: ' + status + '(' + str(index) + \
								') RESOURCES ALLOCATED (via RELAY) quantity=' + str(resourcesAllocated) + \
								' :', dlRRCChannel)
							
							# Gets the resources allocated by the gNB for this device.
							myULOfferedTotalResources = (int(dlRRCChannel.split('|')[6].split('=')[1].split('-')[1])\
							 - int(dlRRCChannel.split('|')[6].split('=')[1].split('-')[0]) + 1)\
							 *int(dlRRCChannel.split('|')[9].split('=')[1]) # (SC_ul_max - SC_ul_min)*L_ul
							
							# Prints logs for the requester including the allocated resources.
							if printLogs == True: print('UL offered', myULOfferedTotalResources)

							# Sets the index for the channel where starts the block of resources of
							# this device.
							myULResourcesIndex = int(dlRRCChannel.split('|')[6].split('=')[1].split('-')[0])\
							 + int(dlRRCChannel.split('|')[8].split('=')[1])
							
							# This device is not anymore a requester, now it is a new relay and it will
							# behave like that.
							status = 'Relay'

							# Informs that MSG4 was received.
							rrcReceived = True

							if totalRegisteredDevices > fixedR:
								totalTimeForRegistration = env.now - richedRelaysStartTime + timeResolution
							else:
								# Informs the exact timestamp which this device was registered.
								totalTimeForRegistration = env.now

						# Stops looking for MSG4.
						break
			# If the requester sent MSG3 by using the traditional Random-Access approach...
			if sentRRCRequestPRACH == True:
				# The requester will search for the first empty downlink channel
				# to receive MSG4.
				for dlRRCIndex, dlRRCChannel in enumerate(downlinkRRC):
					# If there is a downlink channel with info (MSG4) and the info contains the 
					# TC_RNTI of this requester, and the message also contains 
					# the 40-bit value sent before by this requester in MSG3, the requester 
					# will acquire resources from the mobile network via relay.
					if dlRRCChannel != None and \
						dlRRCChannel.split('|')[0].split('=')[1] == TC_RNTI and \
						dlRRCChannel.split('|')[1].split('=')[1] == str(randomValue):
						
						# If this requester has not resources yet, the requester acquires
						# the resources offered by the gNB.
						if resourcesAllocated == 0:
							# Add 1 to the total number of registered devices via the 
							# traditional Random-Access procedure.
							totalRegisteredByGNB += 1

							# Add 1 to the total number of registered devices.
							totalRegisteredDevices += 1

							# Add 1 to the total number of resources allocated for this device.
							# In this case, 1 is the maximum number of times the resources are allocated.
							resourcesAllocated += 1
							
							if totalRegisteredDevices > fixedR:
								registeredDevicesByTimestamp.append(env.now - richedRelaysStartTime + timeResolution)
							else:
								# Saves the exact timestamp which this device was registered.
								registeredDevicesByTimestamp.append(env.now)

							# Prints logs for the requester including timestamp, the total number of
							# registered devices, this device index, and the slice where this device
							# belongs to.
							print(env.now, 'us: Registered', totalRegisteredDevices, 'of', totalDev + fixedR, \
								'Device (' + str(index) + ') -> (Slice Group: ' + str(networkSlice) + ')')

							# Prints logs for the requester including timestamp, device index, and the
							# number of resources allocated (1), and the channel where the MSG4 was found.
							if printLogs == True: print(env.now, 'us: ' + status + '(' + str(index) + \
								') RESOURCES ALLOCATED (via PRACH) quantity=' + str(resourcesAllocated) + ' :'\
								, dlRRCChannel)

							# Gets the resources allocated by the gNB for this device.
							myULOfferedTotalResources = (int(dlRRCChannel.split('|')[6].split('=')[1].split('-')[1])\
							 - int(dlRRCChannel.split('|')[6].split('=')[1].split('-')[0]) + 1)\
							 *int(dlRRCChannel.split('|')[9].split('=')[1]) # (SC_ul_max - SC_ul_min)*L_ul

							# Prints logs for the requester including the allocated resources.
							if printLogs == True: print('UL offered', myULOfferedTotalResources)

							# Sets the index for the channel where starts the block of resources of
							# this device for uplink and downlink.
							myULResourcesIndex = int(dlRRCChannel.split('|')[6].split('=')[1].split('-')[0])\
							 + int(dlRRCChannel.split('|')[8].split('=')[1])
							myDLResourcesIndex = int(dlRRCChannel.split('|')[2].split('=')[1].split('-')[0])\
							 + int(dlRRCChannel.split('|')[4].split('=')[1])
							
							# This device is not anymore a requester, now it is a new relay and it will
							# behave like that.
							status = 'Relay'

							if totalRegisteredDevices > fixedR:
								totalTimeForRegistration = env.now - richedRelaysStartTime + timeResolution
							else:
								# Informs the exact timestamp which this device was registered.
								totalTimeForRegistration = env.now
						# Stops looking for MSG4.
						break

			# Cleans repeated resource allocation (resources were given twice, from PRACH and from relay).
			if resourcesAllocated > 1:
				# Prints logs for the requester including timestamp, device index, and the
				# notification of the double assigned resources.
				if printLogs == True: print(env.now, 'us: Device (' + str(index) + \
					') CLEANING DOUBLE ASSIGNED RESOURCES')

				# Reduces by 1 the number of resources allocated for this device.
				resourcesAllocated -= 1

				# The resources allocated via the traditional Random-Access procedure are cleaned for
				# both downlink and uplink.
				downlinkRB[myDLResourcesIndex, :] = 0.0
				uplinkRB[myULResourcesIndex, :] = 0.0

			#################### Bluetooth Low Energy (BLE) discovery procedure ####################
			# If the Framework RAA is used (framework == True), and the variable discoverBeforeSIB1 is
			# set to True (this means that the requester can discover relays because there is no need
			# to wait the arriving of SIB1 or SIB1 already arrived), and the traditional Random-Access
			# procedure is running (stopPRACH == False) because the requester has not found an RAR yet,
			# and the inquiring is running (stopDiscovery == False), the requester can start or continue
			# discovering relays.
			if framework == True and discoverBeforeSIB1 == True and stopPRACH == False \
				and stopDiscovery == False and inquirerBackOffTimeOut == True:

				# Advertising.
				# If timeSteps is 0-10 timeSteps or 224-234 timeSteps or 448-458 timeSteps, the requester 
				# sends various advertising PDU parts.
				if (timeSteps >= 0 and timeSteps <= 10) or \
					(timeSteps >= 224 and timeSteps <= 234) or \
					(timeSteps >= 448 and timeSteps <= 458):

					if timeSteps == 0 or timeSteps == 224 or timeSteps == 448:
						# Informs that the interval for advertising PDU transmission has started.
						sendFirstPDU = True

						# Informs that there are no collisions yet.
						advPDUCollisioned = False

						# If the inquiring is done by using a random frequency from 0, 1, and 2.
						if D2DRandomFreq == True:
							freq = random.randint(0, 2)
							D2DRandomFreqSelected = True
						# The channel used for inquiring is 0, 1, or 2 (CH37, CH38, CH39), respectively.
						else:
							if timeSteps == 0: freq = 0 # CH37
							elif timeSteps == 224: freq = 1 # CH38
							elif timeSteps == 448: freq = 2 #CH39

					if sendFirstPDU == True:
						# If the 2.4GHz channel is occupied in the network slice the device belongs to,
						# or if the 2.4GHz channel is stored as a conflictive channel, a collision will
						# occur.
						if band24[networkSlice][freq] != None or collisionListBand24[networkSlice][freq] != None:
							# Informs that a collision occurred for at least one advertising PDU part transmission.
							advPDUCollisioned = True

							# Prints logs for the requester including timestamp, device index, 
							# notification of the collision, and the network slice the device 
							# belongs to.
							if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
								'): Collision freq=' + str(freq) + ' (Slice Group: ' + str(networkSlice) + ')')

							# Free the channel for next transmissions.
							band24[networkSlice][freq] = None

							# Sets the channel as a conflictive channel. Collisions will occur
							# for other devices transmitting at the same time in this channel.
							collisionListBand24[networkSlice][freq] = 'collision'

							# Add 1 to the number of collisions in the 2.4GHz band.
							totalCollisions24Band += 1

							# Informs that occured a collision and no signal was transmitted.
							signalTransmitted = False

							# If the advertising PDU transmission ends, informs that the interval 
							# for advertising PDU transmission has ended.
							if (timeSteps == 10 or timeSteps == 234 or timeSteps == 458): sendFirstPDU = False
						# If the channel is empty, the device can send the inquiring signal.
						else:
							# If the advertising PDU transmission ends, and if there were no collisions, informs
							# that a inquiring packet has been sent.
							if (timeSteps == 10 or timeSteps == 234 or timeSteps == 458) and advPDUCollisioned == False:
								# Prints logs for the requester including timestamp, device index, 
								# notification of the transmission of the inquiring signal, and 
								# the network slice the device belongs to.
								if printLogs == True: print(env.now, 'us: Requester (' + str(index) + \
									'): Inquiry packet sent in frequency: ' + str(freq) + ' (Slice Group: ' \
									+ str(networkSlice) + ')')

								# Sets the channel with the inquiring signal.
								band24[networkSlice][freq] = 'inquiry_' + str(index) + '_' + msg

								# Informs that the interval for advertising PDU transmission has ended.
								sendFirstPDU = False
							# If the advertising PDU transmission ends, and if there were collisions, do nothing.
							# Maybe next time the device finds a relay ;)
							elif (timeSteps == 10 or timeSteps == 234 or timeSteps == 458) and advPDUCollisioned == True:
								# Prints logs for the requester including timestamp, device index, 
								# notification of the NO transmission of the inquiring signal, and 
								# the network slice the device belongs to.
								if printLogs == True: print(env.now, 'us: Requester (' + str(index) + \
									'): Inquiry packet NOT sent in frequency: ' + str(freq) + ' (Slice Group: ' \
									+ str(networkSlice) + ')')

								# Informs that the interval for advertising PDU transmission has ended.
								sendFirstPDU = False
							# If the advertising PDU transmission has not ended, continues transmitting
							# advertising PDU messages.
							else:
								# Prints logs for the requester including timestamp, device index, 
								# notification of the transmission of the advertising PDU part, and 
								# the network slice the device belongs to.
								if printLogs == True: print(env.now, 'us: Requester (' + str(index) + \
									'): Advertising PDU part sent in frequency: ' + str(freq) + ' (Slice Group: ' \
									+ str(networkSlice) + ')')

								# Sets the channel with a part of the advertising PDU.
								band24[networkSlice][freq] = 'advPDU'

							# Informs that an advertising PDU part was transmitted.
							signalTransmitted = True

						# Add 'blePower' to the total energy spent in the 2.4GHz band.
						totalEnergy24Band += blePower

				# Scanning after advertising.
				# If the device is not transmitting inquiring signals, the requester is listening to
				# inquiring responses from nearby relays.
				if (timeSteps > 10 and timeSteps < 224) or \
					(timeSteps > 234 and timeSteps < 448) or \
					(timeSteps > 458):
					
					if D2DRandomFreqSelected == False:
						# Scanning channel 37.
						if timeSteps > 10 and timeSteps < 224:
							freq = 0
						# Scanning channel 38.
						elif timeSteps > 234 and timeSteps < 448:
							freq = 1
						# Scanning channel 39.
						elif timeSteps > 458:
							freq = 2

					# Prints logs for the requester including timestamp, device index, 
					# frequency where the requester is listening to, and 
					# the network slice the device belongs to.
					if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
						') -> Scanning at frequency:', freq, '(Slice Group: ' + str(networkSlice) + ')')
					
					# If the 2.4GHz channel is occupied in the network slice the device belongs to,
					# and the message contains the word 'scan', and the message contains this
					# requester RA_RNTIRequester identifier, the requester knows the message
					# is a response for the inquiring message the requester sent before. Now, 
					# the requester knows a relays has been found and that relay will forward the
					# message containing the requester needed resources to the gNB.
					if band24[networkSlice][freq] != None and \
						band24[networkSlice][freq].split('_')[0] == 'scan' and \
						band24[networkSlice][freq].split('_')[2] == RA_RNTIRequester:

						# Prints logs for the requester including timestamp, device index, 
						# device RA_RNTIRequester, and 
						# the network slice the device belongs to.
						if printLogs == True: print(env.now, 'us: RELAY FOUND (' + \
							band24[networkSlice][freq].split('_')[1] + ') !!! for ' + status + \
							' (' + RA_RNTIRequester + ')' + ' (Slice Group: ' + str(networkSlice) + ')')
						
						# Stops the discovery because a relay has been found.
						stopDiscovery = True

						# Informs that this device needs to set some parameters at the time it starts
						# the relay tasks.
						deviceConfigured = False

						# Informs the exact timestamp which this device found a relay.
						afterRelayFoundTimer = env.now

			# Time to wait for RRC response (MSG4) after relay has been found or RAR is received, 
			# if no RRC response arrives (rrcReceived), restarts discovery.
			if rrcReceived == False and stopDiscovery == True and \
				env.now - afterRelayFoundTimer >= RelayFoundTime:

				# All the parameters are set to default to restart the access procedures.
				sentRRCRequestPRACH = False
				sentRRCRequestRelay = False
				TC_RNTIRequester = ''
				stopDiscovery = False
				stopPRACH = False

				# Computing a new RA_RNTIRequester again for the next access attempt,
				# procedure is speeded up.
				RA_RNTIRequester = str(random.randint(1, 65523))
				msg = RA_RNTIRequester + '|' + rsc # new RA_RNTI included

				# Prints logs for the requester including timestamp, device index, 
				# message sent to relays, and slice group the requester belongs to.
				if printLogs == True: print(env.now, 'us: Device (' + str(index) + \
					'): Requirements via Relay (NEW RA_RNTI): ' + msg)

		# Relay tasks.
		if status == 'Relay':
			if scannerBackOffTimeOut == True:
				# Configures the relay parameters the first time its tasks are executed.
				if deviceConfigured == False:
					# Informs that this device D2D technology has been configured.
					deviceConfigured = True

					# timeSteps is used to set the start of the discovery procedure.
					timeSteps = 0

					# Randomly sets the first frequency the relay will listen to.
					freq = random.choice([0, 1, 2]) # Channels 37 (0), 38 (1), and 39 (2).

					# If the gNB manages the frequency used by the relays...
					if getFreqFromGNB == True:
						# Frequency selected by the gNB where this relay will listen to.
						freq = availableFreq24BandListening[networkSlice]
						if availableFreq24BandListening[networkSlice] == 2:
							# Setting new frequency for the next relay that start listen to.
							availableFreq24BandListening[networkSlice] = 0
						else:
							# Setting new frequency for the next relay that start listen to.
							availableFreq24BandListening[networkSlice] = availableFreq24BandListening[networkSlice] + 1


				if timeSteps <= 800: # scanWindow = 25000 #us -> 25 ms

					# Listens to discovery messages in 2.4GHz band.
					# If this relay has not received a signal and this relay finds a message
					# in its network slice in the 2.4GHz band, and the message contains the
					# word 'inquiry', the relay knows a requester is looking for access.
					if signalReceived == False and band24[networkSlice][freq] != None and \
						band24[networkSlice][freq].split('_')[0] == 'inquiry':

						# Message sent by the requester with the requester resources necessities.
						incomingMsg = band24[networkSlice][freq].split('_')[2]
						
						# Entire message sent by the requester. This message is forwarded by the
						# relay to the gNB.
						msgToForward = band24[networkSlice][freq]

						# Index of the requester that sent the message.
						msgRequester = band24[networkSlice][freq].split('_')[1]

						# Timestamp which the message sent by the requester was received.
						msgTimestamp = env.now
						
						# Prints logs for the relay including timestamp, device index, 
						# the channel where the signal was received, and the slice group the 
						# requester belongs to.
						if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
							') DISCOVERY SIGNAL ARRIVED in freq=' + str(freq) + ', (Slice Group: ' + \
							str(networkSlice) + ')')
						
						# Channel where this requester will send back the acknowledge message to the requester.
						frequencyForResponse = freq

						# Index of this relay to be indentified in order to appear in the logs.
						indexForResponse = index

						# Informs an inquiring signal as arrived (useful to free the channel later).
						signalReceived = True

						# Max number of bits of a discovery message: 
						# 38 bits = 16 bits (RA_RNTI) + 10 bits (cellID) + 2 bits (SC_dl=12,24,36,48) + 2 bits (SC_ul=12,24,36,48) + 4 bits (NoSym_dl) + 4 bits (NoSym_ul)
						if 38/myULOfferedTotalResources < 1:
							numberOfWaitingSlotsToForward = 0
						else:
							numberOfWaitingSlotsToForward = np.ceil(float(38/myULOfferedTotalResources))*timeResolution

						# Informs that this relay should send the requester message to the gNB.
						forwardMsg = True

					# Forwards the discovery message to the gNB after numberOfWaitingSlotsToForward slot, 
					# this is used to simulate that the gNB receives more than one transmission
					# when the number of resources for this relay is less than the number of bits 
					# of the discovery message.
					if forwardMsg == True and env.now - msgTimestamp >= numberOfWaitingSlotsToForward:
						# Informs that there are no more requester messages to forward to the gNB.
						forwardMsg = False

						if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
							') FORWARDED message to gNB for Requester (' + msgRequester + '): ' + msgToForward)
						
						# Forwards the discovery message to the gNB.
						uplinkBW[myULResourcesIndex] = incomingMsg # sending discovery message to gNB
						
						# Informs that the requester message has been forwarded to the gNB.
						txTogNBByRelay = True

						# Add 'mnPower' to the total energy spent in the mobile network band.
						totalEnergyNetwork += mnPower

					# If this relay received an inquiring signal from a requester and the time
					# to wait to reply has expired, the relay will reply with an acknowledge
					# message to the requester.
					if signalReceived == True and env.now - msgTimestamp >= timeToResponse:
						# Informs that there are no more requester messages to handle.
						signalReceived = False

						# This code is not necessary for BLE (D2D technology) because the acknowledge
						# message will collide always due to the few frequencies for D2D communications,
						# in this case there are only 3 available frequencies (channels).
						'''
						# Sends back a response for the discovery message.
						# If the 2.4GHz channel is occupied in the network slice the device belongs to,
						# or if the 2.4GHz channel is stored as a conflictive channel, a collision will
						# occur.
						if band24[networkSlice][freq] != None or collisionListBand24[networkSlice][freq] != None:
							# Prints logs for the relay including timestamp, device index, 
							# notification of the collision, and the network slice the device 
							# belongs to.
							if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
								'): Collision freq=' + str(freq) + ' (Slice Group: ' + str(networkSlice) + ')')

							# Free the channel for next transmissions.
							band24[networkSlice][freq] = None

							# Sets the channel as a conflictive channel. Collisions will occur
							# in the future.
							collisionListBand24[networkSlice][freq] = 'collision'

							# Add 1 to the number of collisions in the 2.4GHz band.
							totalCollisions24Band += 1

							# Informs that occured a collision and no signal was transmitted.
							signalTransmitted = False
						# If the channel is empty, the device can send the acknowledge signal.
						else:
							# Prints logs for the relay including timestamp, device index, 
							# notification of the transmission of the acknowledge signal, and 
							# the network slice the device belongs to.
							if printLogs == True: print(env.now, 'us: Relay (' + str(index) + \
								'): Response packet sent in frequency: ' + str(freq) + ' (Slice Group: ' \
								+ str(networkSlice) + ')')

							# Sets the channel with the acknowledge signal.
							band24[networkSlice][freq] = 'scan_' + str(index) + '_' + msg.split('|')[0]

							# Informs that the acknowledge signal was transmitted.
							signalTransmitted = True

						# Add 'blePower' to the total energy spent in the 2.4GHz band.
						totalEnergy24Band += blePower
						'''
		#############################################
		yield env.timeout(timeResolution) # Time step.
		#############################################

		if status == 'Requester':
			# Increments by 1 the time step.
			timeSteps += 1

			# If the time step is 640, the advertising interval has expired.
			# Therefore, the time step restarts from zero.
			if timeSteps == 640: # advInterval = 20000 #us -> 20 ms
				timeSteps = 0

				if inquirerBackOffTimeOut == True:
					inquirerBackOffTimeOut = False
					inquirerBackOffTimer = env.now

					# As part of the BLE algorithm, a backoff is computed before discovering
					# relays again.
					back_off = random.randint(0, 10)

					# Prints logs for the requester including timestamp, device index, 
					# and backoff.
					if printLogs == True: print(env.now, 'us: ' + status + ' (' + str(index) + \
						') BACKOFF: ' + str(1000*back_off))

					# Applies the computed backoff.
					#yield env.timeout(1000*back_off) # advDelay = 1000 us, 2000 us, ..., 10000 us
					inquirerBackOffInterval = 1000*back_off
		# The inquirer backoff has expired.
		if inquirerBackOffTimeOut == False and env.now - inquirerBackOffTimer >= inquirerBackOffInterval:
			inquirerBackOffTimeOut = True
			inquirerBackOffTimer = env.now

			# Resetting parameters.
			timeSteps = 0

		if status == 'Relay':
			# Increments by 1 the time step.
			timeSteps += 1

			# If the time step is 1600, the scanning interval has expired.
			# Therefore, the time step restarts from zero.
			if timeSteps == 1600: # scanInterval = 50000 #us -> 50 ms
				timeSteps = 0

				# If the gNB is not in charge of the frequencies the
				# relays listen to, the requester computes the next
				# frequency it will listen to.
				if getFreqFromGNB == False:
					# This is the next frequency the relay will listen to.
					freq += 1
					if freq > 2:
						freq = 0

				if scannerBackOffTimeOut == True:
					scannerBackOffTimeOut = False
					scannerBackOffTimer = env.now

					# Backoff applied to avoid collisions when two relays receive an inquiring
					# message in the same frequency at the same time.
					back_off = random.randint(1, 16304) # 10240ms - 50ms (window) = 10190ms -> 10190ms/0.625ms = 16304 (integer)
					#yield env.timeout(625*back_off - timeResolution)

					scannerBackOffInterval = 625*back_off - timeResolution
		# The scanner backoff has expired.
		if scannerBackOffTimeOut == False and env.now - scannerBackOffTimer >= scannerBackOffInterval:
			scannerBackOffTimeOut = True
			scannerBackOffTimer = env.now

			# Resetting parameters.
			timeSteps = 0
			freq = 0

		# Cleaning signals.
		# If it was transmitted a signal in the 2.4GHz band, now
		# is time to empty the used channel.
		if signalTransmitted == True:
			if printLogs == True: print(env.now, 'us: CLEANED by ' + status + \
				' (' + str(index) + ')'  + ' (Slice Group: ' + str(networkSlice) + ')')
			signalTransmitted = False
			band24[networkSlice][freq] = None

		# If the relay forwarded the requester message to the gNB,
		# now is time to empty the used uplink channel.
		if txTogNBByRelay == True:
			if printLogs == True: print(env.now, 'us: CLEANED by ' + status + \
				' (' + str(index) + ')')
			txTogNBByRelay = False
			uplinkBW[myULResourcesIndex] = None

		# If MSG2 was transmitted, empty the used channel.
		if prachTransmitted == True:
			prachTransmitted = False
			commonUplinkBW[preamble] = None

		# If the requester transmitted MSG3 using the traditional
		# Random-Access approach, empty the used uplink channel.
		if RRCRequestPRACH == True:
			RRCRequestPRACH = False
			uplinkRRC[RRCIndexPRACH] = None

		# If the requester transmitted MSG3 using the Framework
		# RAA approach, empty the used uplink channel.
		if RRCRequestRelay == True:
			RRCRequestRelay = False
			uplinkRRC[RRCIndexRelay] = None

def main(fixedRelays, totalDevices, framework, noClassicRACH, discoverBeforeSIB1\
	, totalNetworkSlices, maxSym, maxSC, blePower, mnPower, getFreqFromGNB\
	, D2DRandomFreq, printLogs, seed):

	#################### METRICS FOR COMPARISON. ####################
	# Total number of collisions in the mobile network band experienced by the devices.
	global totalCollisionsNetwork; totalCollisionsNetwork = 0
	# Total number of collisions in the 2.4GHz band experienced by the devices.
	global totalCollisions24Band; totalCollisions24Band = 0
	# Total units of energy spent in the mobile network band by the devices.
	global totalEnergyNetwork; totalEnergyNetwork = 0
	# Total units of energy spent in the 2.4GHz band by the devices.
	global totalEnergy24Band; totalEnergy24Band = 0
	# Total units of energy spent by the gNB.
	global totalEnergyGNB; totalEnergyGNB = 0
	# Total time for all device's registration.
	global totalTimeForRegistration; totalTimeForRegistration = 0
	# Total number of registered devices in the mobile network.
	global totalRegisteredDevices; totalRegisteredDevices = 0
	# Total number of registered devices by using the traditional Random-Access approach.
	global totalRegisteredByGNB; totalRegisteredByGNB = 0
	# Total number of registered devices by using the proposed Framework RAA procedure.
	global totalRegisteredByRelay; totalRegisteredByRelay = 0
	# Timestamps which each device was registered.
	global registeredDevicesByTimestamp; registeredDevicesByTimestamp = []

	# Total number of devices that want to have access to the network.
	global totalDev
	totalDev = totalDevices

	global fixedR
	fixedR = fixedRelays
	global richedRelaysStartTime
	richedRelaysStartTime = 0

	# Maximum number of subcarriers a device can request.
	maxNumberOfSubcarriersForDevice = 36

	# Maximum number of symbols a device can request.
	maxNumberOfSymbolsInFrame = 14

	global numerologySlot
	# 66.67us -> slot time for 15kHz numerology.
	numerologySlot = 66.67

	# Seed used to replicate random numbers. This is useful to replicate results.
	random.seed(seed)
	
	# Time esolution of 31.25us, used for the simulation.
	timeResolution = 31.25

	# Simulation ends after 2*10**6us = 2s.
	endTime = 2*10**6

	# Bandwidth where are transmitted SIB1 (channel 0), and RAR (channels 1-65).
	global commonDownlinkBW; commonDownlinkBW = [None for i in range(65)]

	# Bandwidth where are transmitted Random-Access Requests (channels 0-64).
	global commonUplinkBW; commonUplinkBW = [None for i in range(64)]

	# Bandwidth where are transmitted RRC Connection Setup (MSG4) messages by the gNB.
	# The bandwidth takes into account that there are enough resources to make a transmission
	# for every device without collision (the maximum number of subcarriers and symbols
	# are used to compute the bandwidth).
	global downlinkRRC; downlinkRRC = [None for i in \
	range(totalDevices*maxNumberOfSubcarriersForDevice*maxNumberOfSymbolsInFrame)]

	# Bandwidth where are transmitted RRC Connection Request (MSG3) messages by the devices.
	# The bandwidth takes into account that there are enough resources for a device
	# to transmit MSG3 without collision (the maximum number of subcarriers and symbols
	# are used to compute the bandwidth).
	global uplinkRRC; uplinkRRC = [None for i in \
	range(totalDevices*maxNumberOfSubcarriersForDevice*maxNumberOfSymbolsInFrame)]

	# Bandwidth where are allocated downlink resources.
	# The bandwidth takes into account that there are enough resources to allocate for a device.
	# (the maximum number of subcarriers and symbols are used to compute the bandwidth).
	global downlinkBW; downlinkBW = [None for i in \
	range(totalDevices*maxNumberOfSubcarriersForDevice*maxNumberOfSymbolsInFrame)]
	
	# Bandwidth where are allocated uplink resources.
	# The bandwidth takes into account that there are enough resources to allocate for a device.
	# (the maximum number of subcarriers and symbols are used to compute the bandwidth).
	global uplinkBW; uplinkBW = [None for i in \
	range(totalDevices*maxNumberOfSubcarriersForDevice*maxNumberOfSymbolsInFrame)]

	# Bandwidth where are transmitted 2.4GHz signals.
	global band24; band24 = [None for i in range(totalNetworkSlices)]
	for bandIndex in range(len(band24)):
		# Only 3 channels are available for BLE.
		band24[bandIndex] = [None for i in range(3)]

	# This is the resource's grid used to allocate resources for the devices in the downlink.
	# totalDevices*maxNumberOfSubcarriersForDevice subcarriers and 14 symbols for downlink.
	global downlinkRB
	downlinkRB = np.zeros((totalDevices*maxNumberOfSubcarriersForDevice, 14))

	# This is the resource's grid used to allocate resources for the devices in the downlink.
	# totalDevices*maxNumberOfSubcarriersForDevice subcarriers and 14 symbols for uplink.
	global uplinkRB
	uplinkRB = np.zeros((totalDevices*maxNumberOfSubcarriersForDevice, 14))

	# List used to notify that a channel is occupied in the mobile network band.
	global collisionList 
	collisionList = [None for i in range(64)]

	# List used to notify that a channel is occupied in the 2.4GHz band.
	global collisionListBand24
	collisionListBand24 = [None for i in range(totalNetworkSlices)]
	for bandIndex in range(len(collisionListBand24)):
		# Only 3 channels are available for BLE.
		collisionListBand24[bandIndex] = [None for i in range(3)]
	global availableFreq24BandListening; availableFreq24BandListening = [0 for i in range(totalNetworkSlices)]

	# Building the environment for the simulation.
	env = simpy.Environment()

	# Run every device's task.
	for i in range(fixedRelays + totalDevices):
		env.process(device(env=env, blePower=blePower, mnPower=mnPower, timeResolution=timeResolution\
			, index=i, status='Requester', totalNetworkSlices=totalNetworkSlices, discoverBeforeSIB1=discoverBeforeSIB1\
			, backOff=True, framework=framework, noClassicRACH=noClassicRACH, maxSym=maxSym, maxSC=maxSC\
			, getFreqFromGNB=getFreqFromGNB, D2DRandomFreq=D2DRandomFreq, printLogs=printLogs))
	
	# Run the gNB tasks.
	env.process(gNB(env=env, mnPower=mnPower, timeResolution=timeResolution, totalNetworkSlices=totalNetworkSlices, printLogs=printLogs))

	# Run the simulation.
	env.run()

	# Total resources allocated for downlink and uplink.
	if printLogs == True: print(downlinkRB)
	if printLogs == True: print(uplinkRB)

	# Stores the resource's grid with the assigned resources 
	# for every device in files for downlink and uplink.
	pd.DataFrame(downlinkRB).to_csv('downlinkRB.csv')
	pd.DataFrame(uplinkRB).to_csv('uplinkRB.csv')

	# Counting the total number of resources occupied by all the devices for downlink.
	totalDownlinkSC = 0
	for resourceRow in range(downlinkRB.shape[0]):
		for resourceColumn in range(downlinkRB.shape[1]):
			if downlinkRB[resourceRow][resourceColumn] != 0.0:
				totalDownlinkSC += 1
				break

	# Counting the total number of resources occupied by all the devices for uplink.
	totalUplinkSC = 0
	for resourceRow in range(uplinkRB.shape[0]):
		for resourceColumn in range(uplinkRB.shape[1]):
			if uplinkRB[resourceRow][resourceColumn] != 0.0:
				totalUplinkSC += 1
				break

	#################### COMPARISON RESULTS. ####################
	print('\nTotal collisions in the mobile network band:', totalCollisionsNetwork)
	print('Total collisions in the 2.4 GHz band:', totalCollisions24Band)
	print('TOTAL COLLISIONS:', totalCollisionsNetwork + totalCollisions24Band)
	print('Total energy spent by devices transmitting to the gNB:', totalEnergyNetwork, 'units')
	print('Total energy spent by devices transmitting in the 2.4 GHz band:', totalEnergy24Band, 'units')
	print('TOTAL ENERGY SPENT BY DEVICES:', totalEnergyNetwork + totalEnergy24Band, 'units')
	print('Total energy spent by the gNB:', totalEnergyGNB, 'units')
	print('TOTAL TIME FOR DEVICE REGISTRATION:', totalTimeForRegistration, 'us')
	print('Total devices registered by the gNB:', totalRegisteredByGNB)
	print('Total devices registered by relays:', totalRegisteredByRelay)
	print('TOTAL REGISTERED DEVICES:', totalRegisteredDevices, 'of', totalDevices)
	print('######################################################################\n')
	
	# Builds a string with the timestamps in wich each device was registered.
	histogram = ''
	for timestamp in registeredDevicesByTimestamp:
		histogram = histogram + str(timestamp) + '|'

	# Returns the logs from the simulation.
	return histogram, str(totalDevices) + '|' + str(maxSym) + '|' + str(maxSC) + '|' + \
	str(totalCollisionsNetwork) + '|' + str(totalCollisions24Band) + '|' + \
	str(totalCollisionsNetwork + totalCollisions24Band) + '|' + str(totalEnergyNetwork) + \
	'|' + str(totalEnergy24Band) + '|' + str(totalEnergyNetwork + totalEnergy24Band) + '|' + \
	str(totalEnergyGNB) + '|' + str(totalTimeForRegistration) + '|' + str(totalRegisteredByGNB) + \
	'|' + str(totalRegisteredByRelay) + '|' + str(totalRegisteredDevices) + '|' + str(totalDownlinkSC) + \
	'|' + str(totalUplinkSC)

if __name__ == '__main__':
	# Total number of connected devices before the simulation start.
	fixedRelays = 1000

	# Total number of devices that want to get resources from the network.
	totalDevices = 100

	# FOR COMPARISON.
	# Traditional RACH procedure (False) or using Framework RAA (True).
	framework = True

	# Uses classic RACH (False) or not (True).
	noClassicRACH = True
	
	# Starts discovery after receiving SIB1 (False) or before SIB1 (True).
	discoverBeforeSIB1 = False
	
	# If the D2D power transmission is the Bluetooth power transmission (True)
	# or the Wi-Fi power transmission (False).
	ble = True
	if ble == True:
		# Total number of network slices (number of possible groups of D2D devices).
		totalNetworkSlices = int(4.16*10**5/7853.98) # micro-cell coverage area (hexagonal -> a=400m) / bluetooth coverage area (circular -> radius=50m).

		# Bluetooth transmission power (8dBm=0.0063095734448Watts).
		blePower = int(0.0063095734448/0.0063095734448) # normalizing with 8dBm.
	else:
		totalNetworkSlices = int(4.16*10**5/31415.93) # micro-cell coverage area (hexagonal -> a=400m) / wifi coverage area (circular -> radius=100m).

		# Bluetooth transmission power (8dBm=0.0063095734448Watts).
		blePower = int(0.1/0.0063095734448) # normalizing with 8dBm (wifi=20dBm=100mWatts).

	# Mobile Network transmission power (24dBm=0.25118864315Watts).
	mnPower = int(0.25118864315/0.0063095734448) # normalizing with 8dBm.
	
	# Maximum number of symbols a device can request to the network.
	maxSym = 14

	# Maximum number of subcarriers a device can request to the network.
	maxSC = 3

	# Listens only in the frequency the gNB dictates (True) or the relay (False) during Bluetooth scanning.
	getFreqFromGNB = False

	# Selects a random frequency for D2D inquiring (True) or the traditional frequency generation (False).
	D2DRandomFreq = False

	# Prints logs in console (True) or not (False).
	printLogs = False

	# Seed (random generator).
	seed = 0
	
	# Only for test.
	'''
	histogram, logs = main(fixedRelays, totalDevices, framework, noClassicRACH, discoverBeforeSIB1, totalNetworkSlices, \
		maxSym, maxSC, blePower, mnPower, getFreqFromGNB, D2DRandomFreq, printLogs, seed=seed)
	print(logs)
	print(histogram)
	'''

	for seed in range(10):
		if not os.path.isdir('!logs_for_average'): os.mkdir('!logs_for_average')
		if not os.path.isdir('!logs_for_average/logs' + str(seed)): os.mkdir('!logs_for_average/logs' + str(seed))
		'''
		############ 1 graph ############
		# 1. comparison by number of devices -> 
		# only RACH (without framework), 52 network slices, 14 symbols, 36 subcarriers.		
		for totalD in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
			print('1st graph (1) -> Devices:', totalD)
			histogram, logs = main(fixedRelays, totalD, framework=False, noClassicRACH=False, discoverBeforeSIB1=False, \
				totalNetworkSlices=totalNetworkSlices, maxSym=maxSym, maxSC=maxSC, blePower=blePower, mnPower=mnPower, \
				getFreqFromGNB=False, D2DRandomFreq=False, printLogs=False, seed=seed)
			file = open('!logs_for_average/logs' + str(seed) + '/01.log', 'a')
			file.write(logs + '\n')
			file.write(histogram + '\n')
			file.close()
		
		# 2. comparison by number of devices -> 
		# RACH and framework (after SIB1 discovery), 52 network slices, 14 symbols, 36 subcarriers.
		for totalD in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
			print('1st graph (2) -> Devices:', totalD)
			histogram, logs = main(fixedRelays, totalD, framework=True, noClassicRACH=True, discoverBeforeSIB1=False, \
				totalNetworkSlices=totalNetworkSlices, maxSym=maxSym, maxSC=maxSC, blePower=blePower, mnPower=mnPower, \
				getFreqFromGNB=False, D2DRandomFreq=False, printLogs=False, seed=seed)
			file = open('!logs_for_average/logs' + str(seed) + '/02.log', 'a')
			file.write(logs + '\n')
			file.write(histogram + '\n')
			file.close()
		
		# 3. comparison by number of devices -> 
		# RACH and framework (before SIB1 discovery), 52 network slices, 14 symbols, 36 subcarriers.
		for totalD in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
			print('1st graph (3) -> Devices:', totalD)
			histogram, logs = main(fixedRelays, totalD, framework=True, noClassicRACH=True, discoverBeforeSIB1=True, \
				totalNetworkSlices=totalNetworkSlices, maxSym=maxSym, maxSC=maxSC, blePower=blePower, mnPower=mnPower, \
				getFreqFromGNB=False, D2DRandomFreq=False, printLogs=False, seed=seed)
			file = open('!logs_for_average/logs' + str(seed) + '/03.log', 'a')
			file.write(logs + '\n')
			file.write(histogram + '\n')
			file.close()
		'''
		# 4. comparison by number of devices -> 
		# RACH and framework (after SIB1 discovery) and frequencies for listening
		# offered by the gNB, 52 network slices, 14 symbols, 36 subcarriers.
		for totalD in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
			print('1st graph (4) -> Devices:', totalD)
			histogram, logs = main(fixedRelays, totalD, framework=True, noClassicRACH=True, discoverBeforeSIB1=False, \
				totalNetworkSlices=totalNetworkSlices, maxSym=maxSym, maxSC=maxSC, blePower=blePower, mnPower=mnPower, \
				getFreqFromGNB=True, D2DRandomFreq=False, printLogs=False, seed=seed)
			file = open('!logs_for_average/logs' + str(seed) + '/04.log', 'a')
			file.write(logs + '\n')
			file.write(histogram + '\n')
			file.close()
		'''
		# 5. comparison by number of devices -> 
		# RACH and Framework (before SIB1 discovery) and frequencies for listening
		# offered by the gNB, 52 network slices, 14 symbols, 36 subcarriers.
		for totalD in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
			print('1st graph (5) -> Devices:', totalD)
			histogram, logs = main(fixedRelays, totalD, framework=True, noClassicRACH=True, discoverBeforeSIB1=True, \
				totalNetworkSlices=totalNetworkSlices, maxSym=maxSym, maxSC=maxSC, blePower=blePower, mnPower=mnPower, \
				getFreqFromGNB=True, D2DRandomFreq=False, printLogs=False, seed=seed)
			file = open('!logs_for_average/logs' + str(seed) + '/05.log', 'a')
			file.write(logs + '\n')
			file.write(histogram + '\n')
			file.close()
		#################################
		
		# 6. comparison by maximum number of symbols and number of subcarriers -> 
		# 100 devices, RACH and Framework (before SIB1 discovery), 52 network slices.
		for maxS in [5, 10, 14]:
			for maxSc in [1, 2, 3]:
				print('2nd graph -> Subcarriers:', maxSc*12, 'Symbols:', maxS)
				histogram, logs = main(fixedRelays, 100, framework=True, noClassicRACH=True, discoverBeforeSIB1=True, \
					totalNetworkSlices=totalNetworkSlices, maxSym=maxS, maxSC=maxSc, blePower=blePower, mnPower=mnPower, \
					getFreqFromGNB=False, D2DRandomFreq=False, printLogs=False, seed=seed)
				file = open('!logs_for_average/logs' + str(seed) + '/06.log', 'a')
				file.write(logs + '\n')
				file.write(histogram + '\n')
				file.close()
		#################################

		# 7. comparison by maximum number of symbols and number of subcarriers -> 
		# 1000 devices, RACH and framework (before SIB1 discovery), 52 network slices.
		for maxS in [5, 10, 14]:
			for maxSc in [1, 2, 3]:
				print('3rd graph -> Subcarriers:', maxSc*12, 'Symbols:', maxS)
				histogram, logs = main(fixedRelays, 1000, framework=True, noClassicRACH=True, discoverBeforeSIB1=True, \
					totalNetworkSlices=totalNetworkSlices, maxSym=maxS, maxSC=maxSc, blePower=blePower, mnPower=mnPower, \
					getFreqFromGNB=False, D2DRandomFreq=False, printLogs=False, seed=seed)
				file = open('!logs_for_average/logs' + str(seed) + '/07.log', 'a')
				file.write(logs + '\n')
				file.write(histogram + '\n')
				file.close()
		#################################
		'''