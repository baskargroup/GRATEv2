## Files Description:
### acrossDSpacingInfo.csv:
- Contains following data columns:
	- **Metric Distance**: In nm, The [[Documentation#Metric distance]] is evaluated. 
	- **Direct Distance**: In nm, The direct centroid to centroid distance is evaluated. 
	- **Realtive Angle**: Between 0 to 90 degree, The angle difference between the two crystal pattern orientation.
- Data is evaluated using the following procedure:
	- For each image, all combinations of crystal pairs, one in 1.9nm D-spacing range and other in 0.7nm D-spacing range are considered.
	- For each pair, we calculate the data.

### sameDSpacingInfo.csv: 
- There are two such files one for each type of crystals (1.9nm and 0.7nm).
- Contains the same data as acrossDSpacingInfo.csv, but evalutaed between the crystals of same D-spacing.

### overall.csv:
- There are two such files one for each type of crystals (1.9nm and 0.7nm).
- Data is evaluated for each crystal.
- Contains the following data columns:
	1. **Image name**
	2. **Centroid**: Centroid location of crystal in the image.  
	3. **Crystal Area (nm^2)**: Shrink wrap area of the crystal. 
	4.  **Crystal Angle (zero at X-axis and clockwise positive)**: Angle of the pattern in the crystals. 
	5.  **D-Spacing(FFT, nm)**: D-spacing evaluated from the FFT.
	6.  **crystalMajorAxis_length (nm)**: Length of the major axis of the crystal.
	7.  **crystalMinorAxis_length (nm)** : Length of the minor axis of the crystal.
	8.  **MajorAxisAngle** : In degrees, Angle of the major axis of the crystal. Angle is zero at X-axis and clockwise positive.
	9.  **angleDifference**: In degrees (between 0 to 90), Difference between the angle of crystal pattern (point 4) and crystal major axis angle (point 8).
            
**Note**: We filter out crystals with area smaller than:
	- 7*Dspacing^2: for 1.9nm dspacing crystals
	- 10*Dspacing^2: for 0.7nm dspacing crystals
## Metric distance:
$Metric Distance$ = $\frac{C1C2}{r1+r2}$
-   C1C2 : Distance between center 1 to center 2.
-   r1:  Radius of crystal 1
-   r2: Radius of crystal 2