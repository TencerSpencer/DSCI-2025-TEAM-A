



def getMembersOfClass(clazz):
    return [attr for attr in dir(clazz) if not callable(getattr(clazz, attr)) and not attr.startswith("__")]

def getAttrValue(clazz, member):
    clazz.__getattribute__(member)