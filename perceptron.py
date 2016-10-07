
LEARNING_RATE=1
TRAINING_FACES="training.txt"
TRAINING_EXPRESSIONS="training-facit.txt"
TEST_FACES=""

HAPPY=1
SAD=2
MISCHEVIOUS=3
MAD=4

def openFile(file):
    with open(file) as f:
        return f.readlines()

def parseFaces(file):
    
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
                faces[currentImage].append(row)

    return faces

def parseExpressions(file):
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

    outputValue=0
    for i in range(len(inputList)):
        outputValue+=inputList(i)*weightList(i)
    return outputValue

# Note: don't print all the faces at once, as this crashes idle.
# Do something along the lines of "for face in faces: print face"

if __name__ == "__main__":


    # -- Establish Neural Network -- #
    
    weights=list()

    for i in range(5):
        weights.append(list())

    # Create weights for each of the four output nodes
    for i in range(400):
        weights[HAPPY].append(1)
        weights[SAD].append(1)
        weights[MISCHEVIOUS].append(1)
        weights[MAD].append(1)

    # Neural network established.


    # -- Training -- #

    trainingFaces=parseFaces(TRAINING_FACES)
    trainingExpressions=parseExpressions(TRAINING_EXPRESSIONS)
    
    for i in range(len(trainingFaces)):
        # First calculate the output for the face and its expression
        currentExpression=trainingExpressions[i]
        face=trainingFaces[i]

        output=[0,0,0,0,0]
        error=[0,0,0,0,0]

        # Get the errors from the desired outputs (1 for the current expression, 0 for others)
        for expression in [HAPPY,SAD,MISCHEVIOUS,MAD]:

            output[expression]=outputNodeValue(face,weights[expression])
            error[expression]=-output[expression]
            
        error[currentExpression] = 1 - output[currentExpression]

        
        # Which we use to adjust each weight for the expression        
        for node in range(400):
            for expression in [HAPPY,SAD,MISCHEVIOUS,MAD]:
                weightDifference = LEARNING_RATE*error[expression]*face[node]
                weights[expression][node] += weightDifference

    # Training complete, weights defined.



    # -- Testing -- #

    testFaces=parseFaces(TEST_FACES)

    for face in testFaces
        output=[0,0,0,0,0]
        result=0

        for expression in [HAPPY,SAD,MISCHEVIOUS,MAD]:
            output[expression]=outputNodeValue(face,weights[expression])
            if output[expression]>output[result]:
                result=expression

        if result == HAPPY:
            print("Happy")
        elif result == SAD:
            print("Sad")
        elif result == MISCHEVIOUS:
            print("Mischevious")
        elif result == MAD:
            print("Mad")
        else:
            print("Error: result is value ",result)
        
            
            
        
