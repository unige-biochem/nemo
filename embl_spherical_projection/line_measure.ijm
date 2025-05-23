// Extra FIJI script for NEMO, the Nematics & Morphology Toolkit.
// Purpose: Embryo Flattening for Magdalena Schindler (EMBL Heidelberg, Petridou Lab).
// Author: Konstantinos Andreadis

getDimensions(width, height, channels, slices, frames);
open("..../coordinate_metadata.tiff"); // [!] Specify Path for input metadata TIFF [!]
rename("MetadataTIFF");
outputPath = "..../arc_lengths.csv"; // [!] Specify Path for output CSV [!]
run("Make Substack...", "channels=1,2,3 stack");

run("ROI Manager...");
waitForUser("Draw multiple line ROIs and click 'Add' for each one in the ROI Manager, then press OK");

setBatchMode(true);

n = roiManager("count");
totalArcLength = 0;
csv = "ROI,ArcLength(um)\n"; // CSV header
print("Found " + n + " lines in ROI!");

for (r = 0; r < n; r++) {
    roiManager("Select", r);
    Stack.getPosition(c, z, t); // get T and Z of where line was drawn
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

        // Calculate correct metadata slice indices (1-based)
        phiSlice   = ((t - 1) * slices * 3) + ((z - 1) * 3) + 1;
        thetaSlice = phiSlice + 1;
        RSlice     = phiSlice + 2;

        selectWindow("MetadataTIFF");
        setSlice(phiSlice);   phi1 = getPixel(x1, y1);
        setSlice(thetaSlice); theta1 = getPixel(x1, y1);
        setSlice(RSlice);     R1 = getPixel(x1, y1);

        setSlice(phiSlice);   phi2 = getPixel(x2, y2);
        setSlice(thetaSlice); theta2 = getPixel(x2, y2);
        setSlice(RSlice);     R2 = getPixel(x2, y2);

        phi1 = PI * phi1 / 180;
        theta1 = PI * theta1 / 180;
        phi2 = PI * phi2 / 180;
        theta2 = PI * theta2 / 180;

        deltaTheta = Math.abs(theta2 - theta1);
        deltaPhi = Math.abs(phi2 - phi1);

        dS = R1 * Math.sqrt(Math.pow(deltaTheta, 2) + Math.pow(Math.sin(theta1) * deltaPhi, 2));
        arcLength += dS;
    }

    print("Arc Length of ROI " + r + ": " + arcLength + " um");
    csv += "" + r + "," + arcLength + "\n";
    totalArcLength += arcLength;
}

print("Total Arc Length from all ROIs: " + totalArcLength + " um");
File.saveString(csv, outputPath);