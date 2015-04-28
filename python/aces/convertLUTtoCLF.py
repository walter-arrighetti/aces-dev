#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions to convert different LUT formats into CommonLUT Format (CLF)
Additional READ formats added by Walter Arrighetti PhD (walter.arrighetti@gmail.com):
	* FilmLight Truelight Cube, .cub extension (3D, 1D+3D or 1D LUT)
	* IRIDAS Cube, .cube extension (3D LUT)
	* DaVinci Resolve Cube, .cube extension (3D LUT)
	* Autodesk Lustre 3D LUT, .3dl extension (3D LUT)
	* Digital Vision Nucoda Color Management System (CMS), .cms extension (3D, 1D+3D or 1D LUT)
	* Rohde & Schwarz Clipster 3D LUT, .xml extension (3D LUT)
"""
import array
import math
import sys
import os
import re

import clf

LUTFORMAT_UNKNOWN = 'Unknown'
LUTFORMAT_1D = '1D'
LUTFORMAT_3D = '3D'
UNKNOWN_RGB = True

genericLUText = {
	"TruelightCube":'cub',
	"IridasCube":'cube',
	"DaVinciCube":'cube',
	"Lustre3DL":'3dl',
	"NucodaCMS":'cms',
	"Clipster":'xml'
}
genericLUT = {  # accepted_size  newline  1st.ch  red1st  float.fmt  nextch  commentable?   [ shapersize  newline  1st.ch  float.fmt nextch ]
	"TruelightCube":[ [16,17,33,64],'\n'," ",True,".6f"," ",False,    [0,2,101,257,1024],'\n'" ",".6f"," " ],
	"IridasCube":[ [16,17,32,64],'\r\n',"",UNKNOWN_RGB,".6f"," ",True ],
	"DaVinciCube":[ [21,33],'\r\n',"",UNKNOWN_RGB,".6f"," ",False ],
	"Lustre3DL":[ [17],'\n'," ",True,"-4d"," ",True,    [17],'\n',"","d"," " ],
	"NucodaCMS":[ [8,9,16,17,32,33,66],'\r\n',"",UNKNOWN_RGB,".6f"," ",True,    [8,64,101,1024,4096,65536],'\r\n',"",".6f"," " ],
	"Clipster":[ [17],'\r\n'," ",UNKNOWN_RGB,"d"," ",False ],
}

def parse_TruelightCube(lines):
	cursor, has1D, has3D = 0, False, False
	line, side, size = 0, 0, 0
	lutpns, samples1D, samples3D = [], [], []

	if lines[0].strip().split() != ["#","Truelight","Cube","v2.0"]:	return None
	inheader, n = True, 1
	CubeVer = float(lines[0].strip().split()[3][1:])
	if int(CubeVer) != 2:	return None
	while inheader:
		line = lines[n].strip().split()
		if line[0] != "#":
			inheader = False
			n += 1
			break
		elif line[1] == "lutLength" and line[2].isdigit():
			has1d = True
			size = int(line[2])
		elif line[1] == "iDims" and line[2].isdigit():
			if int(line[2]) != 3:	return None
		elif line[1] == "oDims" and line[2].isdigit():
			if int(line[2]) != 3:	return None
		elif len(line)==5 and line[1] == "width" and line[2].isdigit() and line[3].isdigit() and line[4].isdigit():
			has3d = True
			if int(line[2]) == int(line[3]) == int(line[4]):
				side = int(line[2])
			else:	return None
		elif has1d and line[1]=="InputLUT":	break
		elif has3d and line[1]=="Cube":	break
		n += 1
	inheader, cursor = False, n
	if (not has1d) or (not has3d):	return None

	if has1d:
		if lines[cursor].strip().split() != ["#","InputLUT"]:	return None
		for n in range(cursor+1,cursor+1+size):
			samples1D.extend( map(float,lines[n].strip().split()) )
		cursor += 2+size
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut1d", "lut1d")
		lutpn.setArray(size, samples1D)
		lutpns.append(lutpn)
	if has3d:
		if lines[cursor].strip().split() != ["#","Cube"]:	return None
		for n in range(cursor+1, cursor+1+(side**3)):
			samples3D.extend( map(float,lines[n].strip().split()) )
		cursor += 2+(side**3)
		lutpn = clf.LUT3D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut3d", "lut3d")
		lutpn.setArray([side,side,side], samples3D)
		lutpns.append(lutpn)

	if not lutpns:	return None
	del samples1D, samples3D, lutpn
	return lutpns

def parse_IridasCube(lines):
	samples = []
	line = lines[0].strip().split()
	if not (line[0]=="LUT_3D_SIZE" and line[1].isdigit()):	return None
	resolution = [int(line[1]), int(line[1]), int(line[1])]
	if len(lines)-1 != resolution[0]**3:	return None
	for n in range(1,len(lines)):
		sample = map(float,lines[n].strip().split())
		if len(sample)!=3:	break
		samples.extend(sample)
	if len(samples) != 3*(resolution[0]**3):	return None
	lutpn = clf.LUT3D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut3d", "lut3d")
	lutpn.setArray(resolution, samples)
	del samples
	return [ lutpn ]


def parse_Lustre3DL(lines):
	pass

def parse_NucodaCMS(lines):
	cursor, title, has1D, has3D, range1D, range3D = 0, None, 0, 0, (None,None), (None,None)
	line, side, size = lines[0].strip().split(), 0, 0
	lutpns, samples1D, samples3D = [], [], []

	if not line[0]=="NUCODA_3D_CUBE" and line[1].isdigit()):	return None
	CMSver = int(line[1])
	if not 1 <= CMSver <= 3:	return None

	for n in range(1,len(lines)):
		line = lines[n].strip().split()
		if (not title) and line[0]=="TITLE":
			if line[1].startswith('"') and line[1].endswith('"'):	title = line[1].strip('"')
			elif line[1].startswith("'") and line[1].endswith("'"):	title = line[1].strip("'")
			else:	title = line[1].strip()
			continue
		if line[0]=="LUT_1D_SIZE" and line[1].isdigit():
			has1d = has3d+1
			size = int(line[1])
		elif line[0]=="LUT_3D_SIZE" and line[1].isdigit():
			has3d = has1d+1
			side = int(line[1])
		elif CMSver>=3 and has1d and line[0]=="LUT_1D_INPUT_RANGE" and len(line)==3:
			range1D = (float(line[1]), float(line[2]))
		elif CMSver>=3 and has3d and line[0]=="LUT_3D_INPUT_RANGE" and len(line)==3:
			range3D = (float(line[1]), float(line[2]))
		elif len(line)==3 and line[0][0].isdigit() and line[1][0].isdigit() and line[2][0].isdigit():
			cursor = n
			break
		else:	return None
	if (not cursor) or ((not has1d) and (not has3d)):	return None

	if has1d and ((not has3d) or has1d<has3d):	# 1D LUT is either alone or comes *before* 3D LUT
		for n in range(cursor, cursor+size):
			samples1D.extend( map(float,lines[n].split()) )
		cursor += size
		if range1D not in [(None,None),(1.,0.)]:
			rangepn = clf.Range(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "range", "range")
			rangepn.setMinInValue(range1D[0])
			rangepn.setMaxInValue(range1D[1])
			rangepn.setMinOutValue(0.0)
			rangepn.setMaxOutValue(1.0)
			lutpns.append(rangepn)
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut1d", "lut1d")
		lutpn.setArray(size, samples1D)
		lutpns.append(lutpn)
	if has3d:									# 3D LUT is either alone or comes *before* 1D LUT
		for n in range(cursor, cursor+(side**3)):
			samples3D.extend( map(float,lines[n].split()) )
		cursor += side**3
		lutpn = clf.LUT3D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut3d", "lut3d")
		lutpn.setArray([side,side,side], samples3D)
		lutpns.append(lutpn)
	if has1d and has1d>has3d:					# 1D LUT comes *after* 3D LUT
		for n in range(cursor, cursor+size):
			samples1D.extend( map(float,lines[n].split()) )
		cursor += sidz**3
		if range3D not in [(None,None),(1.,0.)]:
			rangepn = clf.Range(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "range", "range")
			rangepn.setMinInValue(range3D[0])
			rangepn.setMaxInValue(range3D[1])
			rangepn.setMinOutValue(0.0)
			rangepn.setMaxOutValue(1.0)
			lutpns.append(rangepn)
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut1d", "lut1d")
		lutpn.setArray(size, samples3D)
		lutpns.append(lutpn)
	if not lutpns:	return None
	del samples1D, samples3D, lutpn
	return lutpns

def parse_ClipsterXML(lines):
	samples = []
	header = re.match(r"<LUT3D\s+name='(?P<name>.+)'\s+N='(?P<side>\d+)'\s+BitDepth='(?P<depth>\d+)'>", lines[0], re.I)
	if (not header) or lines[1]!="<values>" or lines[-2]!="</values>" or lines[-1]!="</LUT3D>":
		return None
	side, renorm, name = int(header.group("side")), (2**int(header.group("depth")))-1, header.group("name")
	if len(lines)-4 != side**3:	return None
	resolution = [side, side, side]
	for n in range(2,len(lines)-1):
		sample = map(int,lines[n].split())
		if len(sample)!=3:	break
		r, g, b = float(sample[0])/renorm, float(sample[1])/renorm, float(sample[2])/renorm
		samples.extend([r,g,b])
	if len(samples) != 3*(side**3):	return None
	lutpn = clf.LUT3D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut3d", "lut3d")
	lutpn.setArray(resolution, samples)
	del samples
	return [ lutpn ]

def readgenericLUT(lutPath, fileFormat):
	lines = []
	if fileFormat not in genericLUText.keys():	return None
	with open(lutPath,'r') as f:
		rawlines = f.read().splitlines()
	for n in range(len(rawlines)):
		line = rawlines[n].strip()
		if (not line) or line.startswith('#'):	continue
		lines.append(line)
	del rawlines
	if not lines:	return None
	if fileFormat == "TruelightCube":
		print "Reading Truelight Cube"
		return parse_TruelightCube(lines)
	elif fileFormat == "IridasCube":
		print "Reading Iridas Cube"
		return parse_IridasCube(lines)
	elif fileFormat == "DaVinciCube":
		pass
	elif fileFormat == "Lustre3DL":
		print "Reading Autodesk 3D LUT"
		return parse_Lustre3DL(lines)
	elif fileFormat == "NucodaCMS":
		print "Reading Nucoda CMS"
		return parse_NucodaCMS(lines)
	elif fileFormat == "Clipster":
		print "Reading Clipster XML LUT"
		return parse_ClipsterXML(lines)
	else:	return None

# Generate an inverse 1D LUT by evaluating and resampling the original 1D LUT
# Inverse LUTs will have 2x the base LUT's number of entries. This may not be 
# enough to characterize the inverse function well.
def generateLUT1DInverseResampled(resolution, samples, minInputValue, maxInputValue):
	lutpns = []

	# Invert happens in 3 stages
	# 1. Range values down from the min and max output values to 0-1
	# 2. Generate the inverse LUT for these newly reranged values
	# 3. Range the value up to the range defined by minInputValue and maxInputValue
	# This is similar to how .CSP preluts are turned into ProcessNodes

	# Get the resolution of the prelut
	inputResolution = resolution[0]
	channels = resolution[1]

	# XXX
	# Given that we're resampling, we should probably increase the
	# resolution of the resampled lut relative to the source
	outputResolution = inputResolution
	outputResolution *= 2

	# Find the minimum and maximum input
	# XXX
	# We take the min and max of all three preluts because the Range node
	# only takes single value, not RGB triples. If the prelut ranges are 
	# very different, this could introduce some artifacting
	minOutputValue = samples[0]
	for c in range(channels):
		minOutputValue = min(minOutputValue, samples[c])

	maxOutputValue = samples[-channels]
	for c in range(channels):
		maxOutputValue = max(maxOutputValue, samples[-channels+c])

	#print( inputResolution, minInputValue, maxInputValue, minOutputValue, maxOutputValue )

	# Create a Range node to normalize data from the range [minOut, maxOut]
	rangepn1 = clf.Range(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "inverse_1d_range_1", "inverse_1d_range_1")
	rangepn1.setMinInValue(minOutputValue)
	rangepn1.setMaxInValue(maxOutputValue)
	rangepn1.setMinOutValue(0.0)
	rangepn1.setMaxOutValue(1.0)

	lutpns.append(rangepn1)

	# Generate inverse 1d LUT by running values through the
	# - inverse normalization [0,1] back to [minOut, maxOut]
	# - the inverse of the original LUT
	inverseSamples = [0.0]*outputResolution*channels

	for inverseLutIndex in range(outputResolution):

		# Normalized LUT input
		inputValue = float(inverseLutIndex)/(outputResolution-1)

		# Invert the normalization
		rangedValue = inputValue*(maxOutputValue - minOutputValue) + minOutputValue

		inverseSample = [0.0]*channels

		# For each channel
		for channel in range(channels):
			# Find the location of the de-normalized value in the lut
			for lutIndex in range(inputResolution):
				sampleIndex = lutIndex*channels + channel
				if samples[sampleIndex] > rangedValue:
					break

			# Get the interpolation value
			lutIndexLow = max(0, lutIndex-1)
			lutIndexHigh = min(inputResolution-1, lutIndex)
			sampleIndexLow = lutIndexLow*channels + channel
			sampleIndexHigh = lutIndexHigh*channels + channel

			if lutIndexLow == lutIndexHigh:
				lutInterp = 0.0
			else:
				lutInterp = (rangedValue - samples[sampleIndexLow])/(
					samples[sampleIndexHigh] - samples[sampleIndexLow])

			# Find the output value
			outputInterpolated = (lutInterp + lutIndexLow)/(inputResolution-1)

			inverseSample[channel] = outputInterpolated

		inverseSamples[inverseLutIndex*channels:(inverseLutIndex+1)*channels] = inverseSample

	# Create a 1D LUT with generated sample values
	lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "inverse_1d_lut", "inverse_1d_lut")
	lutpn.setArray(channels, inverseSamples)

	lutpns.append(lutpn)

	# Create a Range node to expaand from [0,1] to [minIn, maxIn]
	rangepn2 = clf.Range(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "inverse_1d_range_2", "inverse_1d_range_2")
	rangepn2.setMinInValue(0.0)
	rangepn2.setMaxInValue(1.0)
	rangepn2.setMinOutValue(minInputValue)
	rangepn2.setMaxOutValue(maxInputValue)

	lutpns.append(rangepn2)

	return lutpns

# Generate an inverse 1D LUT by evaluating all possible half-float values.
# All inverse LUTs will be 65536 entries, but you don't have to worry about resolution issues
def generateLUT1DInverseHalfDomain(resolution, samples, minInputValue, maxInputValue, rawHalfs=False):
	lutpns = []

	# Invert happens in 1 stages
	# 1. Generate the inverse LUT for each possible half-float value

	# Get the resolution of the prelut
	inputResolution = resolution[0]
	channels = resolution[1]

	# For this inversion approach, we will always use 64k entries
	outputResolution = 65536

	#print( inputResolution, minInputValue, maxInputValue, minOutputValue, maxOutputValue )

	# Generate inverse 1d LUT by running values through the
	# - the inverse of the original LUT
	inverseSamples = [0.0]*outputResolution*channels

	for inverseLutIndex in range(outputResolution):

		# LUT input
		inputValue = clf.uint16ToHalf(inverseLutIndex)

		inverseSample = [0.0]*channels

		# For each channel
		for channel in range(channels):
			if math.isnan(inputValue):
				outputInterpolated = inputValue
			elif math.isinf(inputValue):
				outputInterpolated = inputValue
			else:
				# Find the location of the value in the lut
				for lutIndex in range(inputResolution):
					sampleIndex = lutIndex*channels + channel
					if samples[sampleIndex] > inputValue:
						break

				# Get the interpolation value
				lutIndexLow = max(0, lutIndex-1)
				lutIndexHigh = min(inputResolution-1, lutIndex)
				sampleIndexLow = lutIndexLow*channels + channel
				sampleIndexHigh = lutIndexHigh*channels + channel

				if lutIndexLow == lutIndexHigh:
					lutInterp = 0.0
				else:
					lutInterp = (inputValue - samples[sampleIndexLow])/(
						samples[sampleIndexHigh] - samples[sampleIndexLow])

				# Find the output value
				outputInterpolated = (lutInterp + lutIndexLow)/(inputResolution-1)

			if rawHalfs:
				outputInterpolated = clf.halfToUInt16(outputInterpolated)

			inverseSample[channel] = outputInterpolated

		inverseSamples[inverseLutIndex*channels:(inverseLutIndex+1)*channels] = inverseSample

	# Create a 1D LUT with generated sample values
	if rawHalfs:
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], 
			"inverse_1d_lut", "inverse_1d_lut", rawHalfs=rawHalfs, halfDomain=True)
	else:
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], 
			"inverse_1d_lut", "inverse_1d_lut", halfDomain=True)
	lutpn.setArray(channels, inverseSamples)

	lutpns.append(lutpn)

	return lutpns

# Generate an inverse 1D LUT using an IndexMap to hold the mapping directly.
def generateLUT1DInverseIndexMap(resolution, samples, minInputValue, maxInputValue):
	lutpns = []

	# Invert happens in 3 stages
	# 1. Create the index map that goes from LUT output values to index values
	# 2. Create a LUT that maps from index values to [0,1]
	# 3. Create a Range node to remap from [0,1] to [minInput,maxInput]

	# Get the resolution of the prelut
	inputResolution = resolution[0]
	channels = resolution[1]

	#print( inputResolution, minInputValue, maxInputValue )

	# Index Maps for the inverse LUT
	indexMaps = []
	for c in range(channels):
		indexMapInput = [0.0]*inputResolution
		for i in range(inputResolution):
			indexMapInput[i] = samples[i*channels + c]

		indexMapOutput = range(inputResolution)

		indexMaps.append([indexMapInput, indexMapOutput])

	# Sample values for the LUT - output is [0,1]
	inverseSamples = [0.0]*inputResolution*channels

	for i in range(inputResolution):
		v = float(i)/(inputResolution-1)
		for c in range(channels):
			inverseSamples[i*channels + c] = v

	# Create a 1D LUT with generated index map and sample values
	lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], 
		"inverse_1d_lut", "inverse_1d_lut")

	if channels == 3:
		lutpn.setIndexMaps(indexMaps[0], indexMaps[1], indexMaps[2])
	else:
		lutpn.setIndexMaps(indexMaps[0])

	lutpn.setArray(channels, inverseSamples)

	lutpns.append(lutpn)

	# Create a Range node to expaand from [0,1] to [minIn, maxIn]
	if minInputValue != 0.0 or maxInputValue != 1.0:
		rangepn2 = clf.Range(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "inverse_1d_range_1", "inverse_1d_range_1")
		rangepn2.setMinInValue(0.0)
		rangepn2.setMaxInValue(1.0)
		rangepn2.setMinOutValue(minInputValue)
		rangepn2.setMaxOutValue(maxInputValue)

		lutpns.append(rangepn2)

	return lutpns

def readSPI1D(lutPath, 
			  direction='forward', 
			  interpolation='linear',
			  inversesUseIndexMaps=True, 
			  inversesUseHalfDomain=True):
	with open(lutPath) as f:
		lines = f.read().splitlines()

	#
	# Read LUT data
	#
	dataFormat = LUTFORMAT_1D
	resolution = [0, 0]
	samples = []
	indexMap = []
	minInputValue = 0.0
	maxInputValue = 1.0

	for line in lines:
		#print( "line : %s" % line )
		tokens = line.split()

		if tokens[0] == "Version":
			version = int(tokens[1])
			if version != 1:
				break
		elif tokens[0] == "From":
			minInputValue = float(tokens[1])
			maxInputValue = float(tokens[2])
		elif tokens[0] == "Length":
			resolution[0] = int(tokens[1])
		elif tokens[0] == "Components":
			resolution[1] = int(tokens[1])
		elif tokens[0] in ["{", "}"]:
			continue
		else:
			samples.extend(map(float, tokens))
		#else:
		#    print( "Skipping line : %s" % tokens )

	#
	# Create ProcessNodes
	#
	lutpns = []

	# Forward transform, pretty straightforward
	if direction == 'forward':
		# Remap input range
		if minInputValue != 0.0 or maxInputValue != 1.0:
			rangepn = clf.Range(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "range", "range")
			rangepn.setMinInValue(minInputValue)
			rangepn.setMaxInValue(maxInputValue)
			rangepn.setMinOutValue(0.0)
			rangepn.setMaxOutValue(1.0)

			lutpns.append(rangepn)

		# LUT node
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut1d", "lut1d", interpolation=interpolation)
		lutpn.setArray(resolution[1], samples)

		lutpns.append(lutpn)

	# Inverse transform, LUT has to be resampled
	else:
		if inversesUseIndexMaps:
			print( "Generating inverse of 1D LUT using Index Maps")
			lutpnInverses = generateLUT1DInverseIndexMap(resolution, samples, minInputValue, maxInputValue)
		elif inversesUseHalfDomain:
			print( "Generating full half-domain inverse of 1D LUT")
			lutpnInverses = generateLUT1DInverseHalfDomain(resolution, samples, minInputValue, maxInputValue, rawHalfs=True)
		else:
			print( "Generating resampled inverse of 1D LUT")
			lutpnInverses = generateLUT1DInverseResampled(resolution, samples, minInputValue, maxInputValue)            
		lutpns.extend(lutpnInverses)

	#print (dataFormat, resolution, samples, indexMap, minInputValue, maxInputValue)
	return lutpns

def readSPI3D(lutPath, direction='forward', interpolation='linear'):
	with open(lutPath) as f:
		lines = f.read().splitlines()

	# Translate between different names for the same interpolation if necessary
	ocioToCLFInterpolation = {'linear':'trilinear'}
	if interpolation in ocioToCLFInterpolation:
		interpolation = ocioToCLFInterpolation[interpolation]

	dataFormat = LUTFORMAT_3D
	resolution = [0, 0, 0]
	samples = []
	indexMap = []
	minInputValue = 0.0
	maxInputValue = 1.0

	tokens = lines[0].split()
	if tokens[0] == "SPILUT":
		version = float(tokens[1])
		if version == 1.0:
			tokens = map(int, lines[1].split())
			if tokens[0] == 3 or tokens[1] == 3:
				tokens = map(int, lines[2].split())
				resolution = tokens

				# This assumes that the value are printed in order
				# with blue incrementing fastest like the CLF specification
				# This is generally true, but should be generalized at some
				# point to take into account the indices printed as the 
				# first three values on each line
				for line in lines[3:]:
					tokens = map(float, line.split())
					samples.extend(tokens[3:]) 

	#
	# Create ProcessNodes
	#
	lutpn = clf.LUT3D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut3d", "lut3d", interpolation=interpolation)
	lutpn.setArray(resolution, samples)

	#return (dataFormat, resolution, samples, indexMap, minInputValue, maxInputValue)
	return [lutpn]

def generateCLFPrelut(cspPreluts):
	prelutpns = []

	# Get the individual preluts
	(prelutR, prelutG, prelutB) = cspPreluts

	# Get the resolution of the prelut
	inputResolution = max(len(prelutR[0]), len(prelutG[0]), len(prelutB[0]))

	# XXX
	# Given that we're resampling, we should probably increase the
	# resolution of the resampled lut relative to the source
	outputResolution = inputResolution

	# If the prelut only affects the range, skip this step
	if inputResolution > 2:
		outputResolution *= 2

	# Find the minimum and maximum input
	# XXX
	# We take the min and max of all three preluts because the Range node
	# only takes single value, not RGB triples. If the prelut ranges are 
	# very different, this could introduce some artifacting
	minInputValue = min(prelutR[0][0], prelutG[0][0], prelutB[0][0])
	maxInputValue = max(prelutR[0][-1], prelutG[0][-1], prelutB[0][-1])

	#print( inputResolution, minInputValue, maxInputValue )

	# Create a Range node to normalize data from that range [min, max]
	rangepn = clf.Range(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "prelut_range", "prelut_range")
	rangepn.setMinInValue(minInputValue)
	rangepn.setMaxInValue(maxInputValue)
	rangepn.setMinOutValue(0.0)
	rangepn.setMaxOutValue(1.0)

	prelutpns.append(rangepn)

	# If the prelut only affects the range, skip generating a lut to represent it
	if inputResolution > 2:
		# Generate 1d LUT by running values through the
		# - inverse normalization
		# - the cspprelut
		samples = [0.0]*outputResolution*3

		for i in range(outputResolution):
			# Normalized LUT input
			inputValue = float(i)/(outputResolution-1)

			# Invert the normalization
			rangedValue = inputValue*(maxInputValue - minInputValue) + minInputValue

			sample = [0.0, 0.0, 0.0]

			# For each channel
			for channel in range(len(cspPreluts)):
				# Find the location of the de-normalized value in the prelut
				for prelutIndex in range(inputResolution):
					if cspPreluts[channel][0][prelutIndex] > rangedValue:
						break

				# Get the interpolation value
				prelutIndexLow = max(0, prelutIndex-1)
				prelutIndexHigh = min(inputResolution-1, prelutIndex)
				prelutInterp = (rangedValue - cspPreluts[channel][0][prelutIndexLow])/(
					cspPreluts[channel][0][prelutIndexHigh] - cspPreluts[channel][0][prelutIndexLow])

				# Find the output value
				outputInterpolationRange = (cspPreluts[channel][1][prelutIndexHigh] - cspPreluts[channel][1][prelutIndexLow])
				outputInterpolated = prelutInterp*outputInterpolationRange + cspPreluts[channel][1][prelutIndexLow]

				sample[channel] = outputInterpolated

			samples[i*3:(i+1)*3] = sample

		# Create a 1D LUT with generated sample values
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "prelut_lut1d", "prelut_lut1d")
		lutpn.setArray(len(cspPreluts), samples)

		prelutpns.append(lutpn)

	return prelutpns

def readCSP(lutPath, direction='forward', interpolation='linear'):
	with open(lutPath) as f:
		lines = f.read().splitlines()

	dataFormat = LUTFORMAT_UNKNOWN
	resolution = []
	samples = []
	prelut = []
	minInputValue = 0.0
	maxInputValue = 1.0

	tokens = lines[0].split()
	if tokens[0] == "CSPLUTV100":
		format = lines[1].split()[0]

		if format in ["1D", "3D"]:
			# Find the first line of metadata
			metaStart = 2
			while lines[metaStart].rstrip() != "BEGIN METADATA":
				metaStart += 1
			metaStart += 1

			metadata = ""
			dataStart = metaStart
			for line in lines[metaStart:]:
				dataStart += 1
				if line.rstrip() == "END METADATA":
					break
				else:
					metadata += (line.rstrip())
					#print( "metadata line : %s" % line.rstrip() )

			#print( "metadata : %s" % metadata)
			dataStart += 1
			
			while lines[dataStart].rstrip() == '':
				#print( "blank line")
				dataStart += 1

			#print( "Prelut data starts on line : %d" % dataStart )

			#
			# Read Index Maps
			#

			# Red Index Map
			prelutRedResolution = int(lines[dataStart+0])
			prelutRedInput = map(float, lines[dataStart+1].split())
			prelutRedOutput = map(float, lines[dataStart+2].split())

			# Green Index Map
			prelutGreenResolution = int(lines[dataStart+3])
			prelutGreenInput = map(float, lines[dataStart+4].split())
			prelutGreenOutput = map(float, lines[dataStart+5].split())

			# Blue Index Map
			prelutBlueResolution = int(lines[dataStart+6])
			prelutBlueInput = map(float, lines[dataStart+7].split())
			prelutBlueOutput = map(float, lines[dataStart+8].split())

			prelut = [[prelutRedInput, prelutRedOutput],
				[prelutGreenInput, prelutGreenOutput],
				[prelutBlueInput, prelutBlueOutput]]

			#
			# Read LUT data
			#
			dataStart = dataStart+10

			while lines[dataStart].rstrip() == '':
				#print( "blank line")
				dataStart += 1

			resolution = map(int, lines[dataStart].split())
			dataStart += 1

			#print( "lut data starts on line : %d" % dataStart )

			# 3D LUT data
			if format == "3D":
				dataFormat = LUTFORMAT_3D

				# CSP incremements LUT samples red, then green, then blue
				# CLF incremements LUT samples blue, then green, then red 
				# so we need to move samples around
				samples = [0.0]*resolution[0]*resolution[1]*resolution[2]*3
				cspIndex = 0
				for line in lines[dataStart:]:
					# Convert from sample number to LUT index, CSP-style
					indexR = cspIndex%resolution[0]
					indexG = (cspIndex/resolution[0])%resolution[1]
					indexB = cspIndex/(resolution[0]*resolution[1])

					# Convert from LUT index to sample number, CLF-style
					clfIndex = (indexR*resolution[0] + indexG)*resolution[1] + indexB
					clfIndex *= 3

					# Convert from text to float
					cspSamples = map(float, line.split())

					# Add to the sample values array
					if len(cspSamples) == 3:
						#print( "csp sample %d -> lut index : %d, %d, %d -> clf index : %d" % 
						#    (cspIndex, indexR, indexG, indexB, clfIndex))
						#print( "lut value : %3.6f %3.6f %3.6f" % (cspSamples[0], cspSamples[1], cspSamples[2]) )

						samples[clfIndex:clfIndex+3] = cspSamples

					cspIndex += 1

			# 1D LUT data
			elif format == "1D":
				dataFormat = LUTFORMAT_1D

				samples = [0.0]*resolution[0]*3

				cspIndex = 0
				for line in lines[dataStart:]:
					clfIndex = cspIndex
					clfIndex *= 3

					# Convert from text to float
					cspSamples = map(float, line.split())

					# Add to the sample values array
					if len(cspSamples) == 3:
						#print( "csp sample %d -> clf index : %d" % 
						#    (cspIndex, clfIndex/3))
						#print( "lut value : %3.6f %3.6f %3.6f" % (cspSamples[0], cspSamples[1], cspSamples[2]) )

						samples[clfIndex:clfIndex+3] = cspSamples

					cspIndex += 1

	#
	# Create ProcessNodes
	#
	lutpns = []

	if prelut != []:
		print( "Generating prelut")
		prelutpns = generateCLFPrelut(prelut)
		if prelutpns != []:
			for prelutpn in prelutpns:
				lutpns.append(prelutpn)

	if format == "3D":
		lutpn = clf.LUT3D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut3d", "lut3d")
		lutpn.setArray(resolution, samples)
		lutpns.append(lutpn)
	elif format == "1D":
		lutpn = clf.LUT1D(clf.bitDepths["FLOAT16"], clf.bitDepths["FLOAT16"], "lut1d", "lut1d")
		lutpn.setArray(3, samples)
		lutpns.append(lutpn)

	#print (dataFormat, resolution, samples, prelut, minInputValue, maxInputValue)
	return lutpns

def getLUTFileFormat(lutPath):
	fileFormat = os.path.split(lutPath)[1].split('.')[-1]
	return fileFormat.lower()

def convertLUTToProcessNode(lutPath, 
							direction='forward', 
							interpolation='linear', 
							inversesUseIndexMaps=False,
							inversesUseHalfDomain=True):
	dataFormat = LUTFORMAT_UNKNOWN
	resolution = [0, 0]
	samples = []
	indexMap = []
	minInputValue = 0.0
	maxInputValue = 1.0
	lutpns = []

	# Read LUTs here
	print( "Reading LUT : %s" % lutPath )

	fileFormat = getLUTFileFormat(lutPath)

	if fileFormat in genericLUText.values():
		for key in genericLUText.keys():
				if fileFormat == genericLUText[key]:
					fileFormat = key
					break
		print "Reading %s LUT"%fileFormat
		lutpns = readgenericLUT(lutPath, fileFormat)
	elif fileFormat == "spi1d":
		print "Reading Sony 1D LUT"
		#(dataFormat, resolution, samples, indexMap, minInputValue, maxInputValue) = readSPI1D(lutPath)
		lutpns = readSPI1D(lutPath, direction, interpolation, inversesUseIndexMaps, inversesUseHalfDomain)
	elif fileFormat == "spi3d":
		print "Reading Sony 3D LUT"
		lutpns = readSPI3D(lutPath, direction, interpolation)
	elif fileFormat == "csp":
		print "Reading CSP 1D shaper + 1D or 3D LUT"
		lutpns = readCSP(lutPath, direction, interpolation)
	else:
		print "Unsupported LUT format : %s" % fileFormat

	return lutpns

def convertLUTtoCLF(lutPath, 
					clfPath,
					inverse=False,
					inversesUseIndexMaps=False, 
					inversesUseHalfDomain=True):
	direction = 'forward'
	if inverse:
		direction = 'inverse'

	# Load the LUT and convert to a CLF ProcessNode
	lutpns = convertLUTToProcessNode(lutPath,
		direction=direction, 
		inversesUseIndexMaps=inversesUseIndexMaps, 
		inversesUseHalfDomain=inversesUseHalfDomain)

	# Create a CLF ProcessList and populate contents
	if lutpns != []:
		pl = clf.ProcessList()

		# Populate
		pl.setID('Converted lut')
		pl.setCompCLFversion(1.0)
		pl.setName('Converted lut')

		for lutpn in lutpns:
			pl.addProcess(lutpn)

		# Write CLF to disk
		pl.writeFile(clfPath)        

def main():
	import optparse

	p = optparse.OptionParser(description='Convert a LUT to the Common LUT Format',
								prog='convertLUTtoCLF',
								version='0.01',
								usage='%prog [options]')

	p.add_option('--lut', '-l', default=None)
	p.add_option('--clf', '-c', default=None)
	p.add_option('--inverse', '', action='store_true', default=False)
	p.add_option('--inversesUseIndexMaps', '', action='store_true', default=False)
	p.add_option('--inversesUseHalfDomain', '', action='store_true', default=False)

	options, arguments = p.parse_args()

	#
	# Get options
	# 
	lutPath = options.lut
	clfPath = options.clf
	inverse = options.inverse
	inversesUseIndexMaps = options.inversesUseIndexMaps
	inversesUseHalfDomain = options.inversesUseHalfDomain

	try:
		argsStart = sys.argv.index('--') + 1
		args = sys.argv[argsStart:]
	except:
		argsStart = len(sys.argv)+1
		args = []

	print "command line : \n%s\n" % " ".join(sys.argv)
 
	#
	# Run 
	#
	if lutPath != None and clfPath != None:
		convertLUTtoCLF(lutPath, 
			clfPath, 
			inverse=inverse,
			inversesUseIndexMaps=inversesUseIndexMaps,
			inversesUseHalfDomain=inversesUseHalfDomain)

# main

if __name__ == '__main__':
	main()
