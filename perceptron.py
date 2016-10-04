def openFile(file):
    with open(file) as f:
        return f.readlines()

def parseFaces(file):
    
    lines=openFile("file")

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
    



parseFaces("training.txt")

