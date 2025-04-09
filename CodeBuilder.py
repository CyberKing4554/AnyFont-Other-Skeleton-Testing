# This file is gonna be messy. To say the least. So I'm keeping it in a separate file.
import numpy as np

# PROGMEM requires restructuring the getLetter method and only works with AVR boards
#  I should use a def for whether it's an AVR and compile-time change which getLetter method it uses
# useProgMem = "PROGMEM"
# useProgMem = ""
useProgMem = "MEM_TYPE"


structCode = """

//######################################################################################
//######################################################################################
//######################################################################################
//######################################          ######################################
//###################################### WARNING: ######################################
//######################################          ######################################
//######################################################################################
//######################################################################################
//###                                                                                ###
//### This code file is automatically generated.                                     ###
//###  Any modifications to this file will be erased next generation.                ###
//###                                                                                ###
//### If you want to modify the code, modify .\\arduinotemplate\\arduinotemplate.ino   ###
//###                                                                                ###
//######################################################################################
//######################################################################################
//######################################################################################









// AVR boards like the Nano have to explicitly be told to leave variables in progmem
//  Whereas SAMD boards do this by default on constant data - so the "MEM_TYPE" keyword is replaced with nothing on SAMD boards.

// About "SAMD_CONST", due to the differences in how PROGMEM and SRAM work, we have to change whether they are const or not to save the most space

#ifdef __AVR__
#define MEM_TYPE PROGMEM
#define SAMD_CONST
#else
#define MEM_TYPE
#define SAMD_CONST const
#endif


#include <stdint.h>

struct Point {
    SAMD_CONST uint8_t x;
    SAMD_CONST uint8_t y;
};

struct Contour {
    SAMD_CONST Point* points;
    SAMD_CONST uint8_t pointCount;
};

struct Letter {
    SAMD_CONST char letter;
    SAMD_CONST Contour* contours;
    SAMD_CONST uint8_t contourCount;
};
"""

# pointsDefinition = "" # Example below
"""
Point A_contour1_points[] = {
    {0, 0}, 
    {1, 1}, 
    {2, 0}
};

Point A_contour2_points[] = {
    {1, 1}, 
    {1, 2}
};

Point B_contour1_points[] = {
    {0, 0}, 
    {0, 2}, 
    {1, 1}, 
    {0, 2}
};
"""


# contourDefinition = "" # Example below
"""
Contour A_contours[] = {
    {A_contour1_points, 3},
    {A_contour2_points, 2}
};

Contour B_contours[] = {
    {B_contour1_points, 4}
};
"""

# letterDefinition = "" # Example below
"""
Letter letters[] = {
    {'A', A_contours, 2},
    {'B', B_contours, 1},
};
"""

# Then this shows how to use it
"""
const int letterCount = sizeof(letters) / sizeof(letters[0]);

Letter* getLetter(char letter) {
    for (int i = 0; i < letterCount; ++i) {
        if (letters[i].letter == letter) {
            return &letters[i];
        }
    }
    return nullptr;
}

void setup() {
    Serial.begin(9600);

    // Example usage: access the letter 'A'
    Letter* letterA = getLetter('A');
    if (letterA != nullptr) {
        for (int i = 0; i < letterA->contourCount; ++i) {
            Contour& contour = letterA->contours[i];
            Serial.print("Contour ");
            Serial.print(i);
            Serial.println(":");
            for (int j = 0; j < contour.pointCount; ++j) {
                Point& point = contour.points[j];
                Serial.print("Point ");
                Serial.print(j);
                Serial.print(": (");
                Serial.print(point.x);
                Serial.print(", ");
                Serial.print(point.y);
                Serial.println(")");
            }
        }
    } else {
        Serial.println("Letter not found");
    }
}

void loop() {
    // Your main code here
}

"""

def buildCode(lettersJson, ignore_height):
    pointsDefinition = ""
    contourDefinition = ""
    letterDefinition = ""

    # One last thing, we're gonna scan through the letters here and store the largest Y value, for the sake of scaling.
    maxXSize = 0
    maxYSize = 0

    # The idea here is pretty simple, it's just tedious. 
    #  Reference the above strings for how the data structure works
    # The basic idea is loop through all the letters, 
    #  exporting points, then contours at the next level down, then finally letters

    # WAIT "PROGMEM" EXISTS?? Now I have like 26x more space to work with

    # Start letter array
    letterDefinition += "const Letter letters[] {mem} = {{\n".format(mem=useProgMem)

    for letterStr in lettersJson:
        letterArray = lettersJson[letterStr]

        # Add letter
        letterDefinition += "\t{{'{letter}', {letter}_contours, {lLen}}},\n".format(letter=letterStr, lLen=len(letterArray))

        contourNum = 1
        localMaxYSize = 0

        # Start this contour array
        contourDefinition += "const Contour {letter}_contours[] {mem} = {{\n".format(letter=letterStr, mem=useProgMem)
        
        for contour in letterArray:
            # Add contour
            contourDefinition += "\t{{{letter}_contour{num}_points, {cLen}}},\n".format(letter=letterStr, num=contourNum, cLen=len(contour))

            # Start point array
            pointsDefinition += "const Point {letter}_contour{num}_points[] {mem} = {{\n".format(letter=letterStr, num=contourNum, mem=useProgMem)
            contourNum += 1

            for point in contour:
                pointsDefinition += "\t{{{x}, {y}}},\n".format(x=point[0], y=point[1])
                maxXSize = max(point[0], maxXSize)
                localMaxYSize = max(point[1], localMaxYSize)
                # maxYSize = max(point[1], maxYSize)

            # Remove comma from final point, close
            pointsDefinition = pointsDefinition[:-2] + "\n};\n"

        # For comic sans, Q goes much lower than everything else, pushing it lower than ideal
        print("DEBUG: " + letterStr + " height is " + str(localMaxYSize))
        if letterStr not in ignore_height:
            maxYSize = max(localMaxYSize, maxYSize)

        # Remove comma from final contour, close
        contourDefinition = contourDefinition[:-2] + "\n};\n"
        
    # Remove comma, close letter array
    letterDefinition = letterDefinition[:-2] + "\n};\n"

    lettersDefs =                        \
        structCode.strip()  + "\n\n" +   \
        pointsDefinition    + "\n" +     \
        contourDefinition   + "\n" +     \
        letterDefinition    + "\n" +     \
        "#define MaxYSize " + str(maxYSize)

        # "#define MaxXSize " + str(maxXSize) + "\n" \
        # "#define MinOfMaxLens " + str(min(maxYSize, maxXSize)) + " # The largest x/y (whatever is lower) of any char,  \n"


    # Read code template
    codeTemplateFile = open("arduinotemplate\\arduinotemplate.ino")
    codeTemplate = codeTemplateFile.read()
    codeTemplateFile.close()
    codeTemplate = codeTemplate[codeTemplate.index("//END LETTERS"):]

    finalCode = lettersDefs + "\n" + codeTemplate

    

    return finalCode

 
