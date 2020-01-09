class Person:
    population = 100

    def __init__(self, name):
        # instance variable
        self.name = name
        Person.population += 1


pobj = Person("Nima")
pobj2 = Person("Ali")
print(pobj.population)
print(pobj.population)
