
def getDomain(inEmail):
    return inEmail[inEmail.find('@') + 1:]

#Run this block when this file is executed direct i.e. as a the main call

if __name__ == '__main__':

    #a change that I don't care about
    theEmail = 'campbell.bg@gmail.com'

    print(getDomain(theEmail))