
#copyUtils.py - by lganic
#faster copy functions for nested lists

"""copyUtils.py - by lganic
faster copy functions for nested lists"""# for help() page


def fastCopyMatrix(mat):
    "Return copy of 2D matrix"
    outMat=[]
    for row in mat:
        outMat.append(row[:])
    return outMat

def fastDeepCopy(matrix3D):
    "Return copy of 3D matrix"
    output=[]
    for item in matrix3D:
        output.append(fastCopyMatrix(item))
    return output


#fastCopyMatrix.__doc__=

#fastDeepCopy.__doc__=
