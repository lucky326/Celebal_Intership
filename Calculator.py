while(True):
    ch=input("Enter your choice + , - , * , / any other key to exit ")
    x=int(input("Enter First Number "))
    y=int(input("Enter Second Number "))
    if(ch=='+'):
        print(f"Addition of {x} and {y} is: {x+y}")
    elif(ch=='-'):
        print(f"substraction of {x} and {y} is: {x-y}")
    elif(ch=="*"):
        print(f"multiplication of {x} and {y} is: {x*y}")
    elif(ch=='/'):
        print(f"division of {x} and {y} is: {x/y}")
    else:
        break