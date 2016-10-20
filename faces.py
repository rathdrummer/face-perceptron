import sys, random

LEARNING_RATE=0.000011 # Keep ~0.00001. 0.000011 gives the highest value if re-testing the existing results

HAPPY=1
SAD=2
MISCHEVIOUS=3
MAD=4

def openFile(file):
    """Does as it says on the tin"""
    with open(file) as f:
        return f.readlines()

def parseFaces(file):
    """Returns a list of faces; each face is a list of rows, each row a list of integers (the pixels)"""
    lines=openFile(file)

    currentImage = 0
    faces=list()

    for line in lines:
        line=line.strip()
        if line != "" and line[0] != '#':
            if line[:5] == 'Image':
                currentImage = int(line[5:])-1
                faces.append(list())
            else:
                row=line.split(" ")
                row = [ int(x) for x in row ] 
                faces[currentImage].append(row)

    return faces

def parseExpressions(file):
    """Returns the corresponding expressions as a list of integers. 1:Happy, 2:Sad, 3:Mischevious, 4:Mad"""
    lines=openFile(file)

    currentExpression=0
    expressions=list()

    for line in lines:
        line=line.strip()
        if line[:5] == "Image":
            parts=line.split(" ")
            currentExpression = int(parts[0][5:])-1
            expressions.append(int(parts[1]))
            
    return expressions

def outputNodeValue(inputList,weightList):
    """Calculates the output for the node produced by weightList as per the input. This is the neuron activation function"""
    outputValue=0
    for i in range(20):
        for j in range(20):
            outputValue += (inputList[i][j]*weightList[i][j])
    return outputValue

if __name__ == "__main__":

    # -- Establish Neural Network -- #
    
    weights=list()

    for i in range(5):
        weights.append(list())

    # Create weights for each of the four output nodes    
    for row in range(20):
        weights[HAPPY].append(list())
        weights[SAD].append(list())
        weights[MISCHEVIOUS].append(list())
        weights[MAD].append(list())
        for node in range(20):
            weights[HAPPY][row].append(0)
            weights[SAD][row].append(0)
            weights[MISCHEVIOUS][row].append(0)
            weights[MAD][row].append(0)

    # Neural network established.


    # -- Training -- #

    trainingFaces=parseFaces(sys.argv[1])
    trainingExpressions=parseExpressions(sys.argv[2])

    for i in range(len(trainingFaces)):
        # First calculate the output for the face and its expression
        currentExpression=trainingExpressions[i]
        face=trainingFaces[i]

        output=[0,0,0,0,0]
        error=[0,0,0,0,0]

        # Get the errors from the desired outputs (desired outputs are for the current corresponding expression, 0 for others)
        for expression in [HAPPY,SAD,MISCHEVIOUS,MAD]:
            output[expression]=outputNodeValue(face,weights[expression])
            error[expression]= 0 - output[expression]
        error[currentExpression] = 1 - output[currentExpression]

        # Which we use to adjust each weight for the expression        
        for row in range(20):
            for node in range(20):
                for expression in [HAPPY,SAD,MISCHEVIOUS,MAD]:
                    weightDifference = LEARNING_RATE*error[expression]*face[row][node]
                    weights[expression][row][node] += float(weightDifference)

    # Training complete, weights defined.



    # -- Testing -- #

    testFaces=parseFaces(sys.argv[3])

    for i in len(testFaces):
        output=[0,0,0,0,0]
        result=0

        for expression in [HAPPY,SAD,MISCHEVIOUS,MAD]:
            output[expression]=outputNodeValue(testFaces[i],weights[expression])
            if output[expression]>output[result]:
                result=expression
        
        print("Image",i," ",expression)
    
