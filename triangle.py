
n=int(input("Enter the size for lower triangle"))

for i in range(n):
    for j in range(i+1):
            print("*",end=" ")
    print()

print("\n\n\n")
n=int(input("Enter the size for upper triangle"))

for i in range(n,0,-1):
    for j in range(i):
            print("*",end=" ")
    print()

print("\n\n\n")
n=int(input("Enter the size for pyramid"))

for i in range(n):
    for j in range(n*2):
        if(j<n-i or j>n+i):
            print(" ",end=" ")
        else:
            print("*",end=" ")
    print()
