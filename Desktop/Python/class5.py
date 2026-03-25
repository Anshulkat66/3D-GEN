#comparison operators
#==, !=, > , < , >=, <=
a= 12
b=34
c=12
print(a==b) #false
print(a!=b) #true
print(a>b) #false
print(a<b) #true
print(a>c)#false
print(a>=b) #false
print(a<=b) #true
print(a>=c)#true
print((12==12)==True) #true
print((12==12)==False) #false

#logical operators
#and, or, not
print(12>10 and 12<20) #false
print(12>10 or 12<20) #true
print(not(12>10)) #false   #not operator is used to reverse the result of a condition