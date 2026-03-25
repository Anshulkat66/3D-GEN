age= int(input("Enter your age:"))
if age >= 18:
    print("You can vote.")
#if we want that nothing should be printed even when condition is truw than instead of print we will write pass after condition
    #pass
else:
    print("You cannot vote.")

money= int(input("Enter the amount of money you have:"))
if money == 10:
    print("I will eat chochobar")
elif money == 20:
    print("I will eat Ice cream")
elif money == 30:
    print("I will eat cake")
else:
    print("I will eat something else")

a= 12
b= 17
c=18
if a>b and a>c:
    print("a is greatest")
elif b>a and b>c:
    print("b is greatest")
else:
    print("c is greatest")
