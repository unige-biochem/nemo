// Extra FIJI script for NEMO, the Nematics & Morphology Toolkit.
// Purpose: Embryo Flattening for Magdalena Schindler (EMBL Heidelberg, Petridou Lab).
// Author: Konstantinos Andreadis
getDimensions(width, height, channels, slices, frames);
outputPath = "/Users/andreadi/Downloads/resss/debug/arc_lengths.csv";
open("/Users/andreadi/Downloads/resss/debug/coordinate_metadata.tiff");
run("Make Substack...", "channels=1,2,3 stack");
run("Close");

run("ROI Manager...");
waitForUser("Draw multiple line ROIs and click 'Add' for each one in the ROI Manager, then press OK");

setBatchMode(true);

n = roiManager("count");
totalArcLength = 0;
csv = "ROI,ArcLength(um)\n"; // CSV header
print("Found " + n + "lines in ROI !");
for (r = 0; r < n; r++) {
    roiManager("Select", r);
    Stack.getPosition(c, z, t);
    print(c, z, t);
    getSelectionCoordinates(x, y);
    makeLine(x[0], y[0], x[1], y[1]);
    run("Interpolate", "interval=1");
    getSelectionCoordinates(x, y);
    arcLength = 0;

    for (i = 0; i < x.length - 1; i++) {
        x1 = x[i];
        y1 = y[i];
        x2 = x[i + 1];
        y2 = y[i + 1];
        
		phiIndex   = (z - 1) * channels + 1;
        thetaIndex = (z - 1) * channels + 2;
        RIndex     = (z - 1) * channels + 3;
        
        setSlice(phiIndex);   phi1 = getPixel(x1, y1);
        setSlice(thetaIndex); theta1 = getPixel(x1, y1);
        setSlice(RIndex);     R1 = getPixel(x1, y1);

        setSlice(phiIndex);   phi2 = getPixel(x2, y2);
        setSlice(thetaIndex); theta2 = getPixel(x2, y2);
        setSlice(RIndex);     R2 = getPixel(x2, y2);

        phi1 = PI * phi1 / 180;
        theta1 = PI * theta1 / 180;
        phi2 = PI * phi2 / 180;
        theta2 = PI * theta2 / 180;

        deltaTheta = Math.abs(theta2 - theta1);
        deltaPhi = Math.abs(phi2 - phi1);

        dS = R1 * Math.sqrt(Math.pow(deltaTheta, 2) + Math.pow(Math.sin(theta1) * deltaPhi, 2));
        arcLength += dS;
    }

    print("Arc Length of ROI " + r + ": " + arcLength + "um");
    csv += "" + r + "," + arcLength + "\n";
    totalArcLength += arcLength;
}
print("Total Arc Length from all ROIs: " + totalArcLength + "um");
File.saveString(csv, outputPath);