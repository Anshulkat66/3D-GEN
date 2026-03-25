#strings
#slicing
a= "Anant Bedi"
print(a[0:8:1])#start:stop:step
print(a[0:8:2]) 
b="Hello I am data scientist"
print(b[0:5:1])
print(b[11:15:1])
print(b[16:25:1])
c=56
c=46
print(c)
d="Hello"
d="hiii"
print(d)#latest or last value will be printed
print(d[1])
#d[0]="H" #string is immutable we cannot change the value of string
age= 20
Des= "Data scientist"
print(f"Hello My age is {age} and my designation is {Des}")
#escape sequences
print(f"Hello My age is {age}\nmy designation is {Des}")
print(f"Hello My age is {age} and\b my designation is {Des}")
print(f"Hello My age is {age}\tmy designation is {Des}")
#type conversion
h= 100
h= str(h)
print(h)
print(type(h))
g= "200"
g= int(g)
print(g)
print(type(g))
#string alphabets cannot be converted to integer
'''
i= "Anant"
i= int(i)
print(i)
this will show error because string cannot be converted to integer
'''
#boolean
#0,0.0,False," ",[],(),{} are considered as false so when they are converted to boolean they will return false value
print(bool(0))
print(bool(0.0))
#rest all will return True value
print(bool(1))
print(bool(-1))
print(bool("Hello"))
print(bool([1, 2, 3]))
#input
name= input("Enter your name: ")
print(f"Hello {name} welcome to python programming")
ages= input("Enter your age: ")
print(type(ages))#it will show string so we will convert the data type to int
ages1=int(input("Enter your age:"))
print(type(ages1))


