from faker import Faker


class Example(object):
    bool143 = True
    bool2 = True
    blah = False
    foo = True
    foobar2000 = False

example = Example()
members = [attr for attr in dir(example) if not callable(getattr(example, attr)) and not attr.startswith("__")]
print(members)   

# to extract the value use: example.__getattribute__(members[0])